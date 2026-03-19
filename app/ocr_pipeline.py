"""OCR pipeline using PaddleOCR with document classification.

Usage: Call process_image(doc_id, s3_key) to run OCR on an uploaded image.
"""
import asyncio
import io
import os
import logging
from typing import Dict, List, Any

import numpy as np
from PIL import Image, ImageOps
from paddleocr import PaddleOCR

from app.s3 import download_from_s3
from app.vlm_client import refine_ocr_with_vlm, extract_document_fields


logger = logging.getLogger(__name__)

# Limit concurrent OCR processing - configurable for different instance sizes
# t3.large (8GB): 2, t3.xlarge (16GB): 4, t3.2xlarge (32GB): 6-8
MAX_CONCURRENT_OCR = int(os.getenv("MAX_CONCURRENT_OCR", "2"))
_processing_semaphore = asyncio.Semaphore(MAX_CONCURRENT_OCR)
logger.info(f"OCR concurrency limit set to {MAX_CONCURRENT_OCR}")

# Lazy-load OCR model
_ocr_model = None


def get_ocr_model() -> PaddleOCR:
    """Get or initialize the PaddleOCR model."""
    global _ocr_model
    if _ocr_model is None:
        logger.info("Initializing PaddleOCR model...")
        _ocr_model = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)
        logger.info("PaddleOCR model loaded")
    return _ocr_model


def preprocess_image(image_bytes: bytes) -> bytes:
    """Preprocess image for OCR - invert if predominantly dark."""
    img = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB if needed (handles PNG with alpha, grayscale, etc.)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Check if image is predominantly dark (mean brightness < 127)
    grayscale = img.convert("L")
    mean_brightness = np.array(grayscale).mean()

    if mean_brightness < 127:
        # Dark image - invert colors for better OCR
        img = ImageOps.invert(img)
        logger.info(f"Inverted dark image (brightness={mean_brightness:.1f})")

    # Convert back to bytes
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def classify_doc_type(ocr_text: str) -> str:
    """Classify document type based on OCR text using heuristics."""
    text_lower = ocr_text.lower()

    # Invoice indicators (check before receipt - invoices often contain "total" too)
    if any(kw in text_lower for kw in ["invoice", "bill to", "remit to", "amount due", "purchase order"]):
        return "invoice"

    # Subscription indicators (check before receipt - subscriptions often contain "total" too)
    if any(kw in text_lower for kw in ["renew", "subscription", "billing", "monthly", "annual"]):
        return "subscription"

    # Receipt indicators
    if any(kw in text_lower for kw in ["total", "subtotal", "tax", "cash", "visa", "mastercard", "payment"]):
        return "receipt"

    # Warranty indicators
    if any(kw in text_lower for kw in ["warranty", "serial number", "guarantee", "coverage"]):
        return "warranty"

    # Insurance indicators
    if any(kw in text_lower for kw in ["policy", "premium", "coverage", "insured", "beneficiary"]):
        return "insurance"

    return "other"


def calculate_confidence(ocr_result: List) -> float:
    """Calculate average confidence from OCR results."""
    if not ocr_result or not ocr_result[0]:
        return 0.0

    confidences = []
    for line in ocr_result[0]:
        if len(line) >= 2 and isinstance(line[1], tuple) and len(line[1]) >= 2:
            confidences.append(line[1][1])

    return sum(confidences) / len(confidences) if confidences else 0.0


def extract_text_and_boxes(ocr_result: List) -> tuple[str, List[Dict[str, Any]]]:
    """Extract text and bounding boxes from OCR result."""
    if not ocr_result or not ocr_result[0]:
        return "", []

    texts = []
    boxes = []

    for line in ocr_result[0]:
        bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        text_info = line[1]  # (text, confidence)

        text = text_info[0] if isinstance(text_info, tuple) else str(text_info)
        confidence = text_info[1] if isinstance(text_info, tuple) and len(text_info) > 1 else 1.0

        texts.append(text)
        boxes.append({
            "text": text,
            "confidence": confidence,
            "bbox": bbox,
        })

    return " ".join(texts), boxes


async def run_ocr_pipeline(image_bytes: bytes) -> Dict[str, Any]:
    """Run OCR pipeline on image bytes and return results."""
    # Preprocess image (invert if dark)
    processed_bytes = preprocess_image(image_bytes)

    # Run PaddleOCR
    ocr = get_ocr_model()
    result = ocr.ocr(processed_bytes, cls=True)

    # Extract text and bounding boxes
    doc_text, ocr_blocks = extract_text_and_boxes(result)
    confidence = calculate_confidence(result)

    # Mark if no text was detected (so status doesn't stay "Processing")
    if not doc_text:
        doc_text = "[No text detected]"
    elif confidence < 0.7:
        # Only use VLM refinement if we have actual OCR text to refine
        try:
            doc_text = await refine_ocr_with_vlm(image_bytes, doc_text)
        except Exception as e:
            logger.warning(f"VLM refinement failed: {e}")

    # Classify document type
    doc_type = classify_doc_type(doc_text)

    # Extract structured fields for classified documents
    extraction = None
    if doc_type in ("receipt", "subscription", "invoice"):
        try:
            extraction = await extract_document_fields(image_bytes, doc_type)
        except Exception as e:
            logger.warning(f"{doc_type} extraction failed: {e}")

    return {
        "doc_text": doc_text,
        "doc_type": doc_type,
        "ocr_blocks": ocr_blocks,
        "confidence": confidence,
        "extraction": extraction,
    }


async def process_image(doc_id: int, s3_key: str) -> None:
    """
    Process an image through the OCR pipeline.
    Uses semaphore to limit concurrent processing (configurable via MAX_CONCURRENT_OCR).
    """
    async with _processing_semaphore:
        logger.info(f"Processing doc_id={doc_id}, s3_key={s3_key}")
        await _process_image_internal(doc_id, s3_key)


async def process_batch_and_crawl(docs: List[Dict[str, Any]], user_id: str) -> None:
    """
    Process a batch of documents, then run radar crawler.
    Uses asyncio.gather to process all docs (semaphore limits concurrency).
    Crawler runs once after all processing completes.
    """
    from app.radar_crawler import crawl_documents

    # Process all images concurrently (semaphore limits actual parallelism)
    await asyncio.gather(*[
        process_image(doc["doc_id"], doc["s3_key"])
        for doc in docs
    ], return_exceptions=True)

    # All done, now crawl for event dates
    try:
        stats = await crawl_documents(user_id=user_id, limit=len(docs) + 5)
        logger.info(f"Radar crawl after batch: {stats}")
    except Exception as e:
        logger.warning(f"Radar crawl failed after batch: {e}")


async def _process_image_internal(doc_id: int, s3_key: str) -> None:
    """Internal processing logic."""
    from app.db import update_document, init_db
    from app.extraction import save_extraction

    try:
        # Download image from S3
        image_bytes = await download_from_s3(s3_key)

        # Run OCR pipeline
        result = await run_ocr_pipeline(image_bytes)

        # Get database connection and update
        pool = await init_db()
        async with pool.acquire() as db:
            # Update document in database
            await update_document(
                db,
                doc_id,
                doc_text=result["doc_text"],
                doc_type=result["doc_type"],
                ocr_blocks=result["ocr_blocks"],
            )

            # Save extraction if present
            if result["extraction"]:
                await save_extraction(db, doc_id, result["doc_type"], result["extraction"])

        logger.info(f"Completed processing doc_id={doc_id}")

    except Exception as e:
        logger.error(f"Failed to process doc_id={doc_id}: {e}")
        # Mark document as failed so it doesn't stay in "Processing" forever
        try:
            pool = await init_db()
            async with pool.acquire() as db:
                await update_document(
                    db,
                    doc_id,
                    doc_text="[Processing failed]",
                    doc_type="other",
                )
            logger.info(f"Marked doc_id={doc_id} as failed")
        except Exception as update_err:
            logger.error(f"Failed to mark doc_id={doc_id} as failed: {update_err}")
