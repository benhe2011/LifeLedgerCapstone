"""OCR pipeline using PaddleOCR with document classification."""
import io
from typing import Dict, List, Any

from PIL import Image
from paddleocr import PaddleOCR

from app.s3 import download_from_s3
from app.db import update_document
from app.vlm_client import refine_ocr_with_vlm, extract_receipt_fields
from app.extraction import save_extraction


# Lazy-load OCR model
_ocr_model = None


def get_ocr_model() -> PaddleOCR:
    """Get or initialize the PaddleOCR model."""
    global _ocr_model
    if _ocr_model is None:
        _ocr_model = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)
    return _ocr_model


def classify_doc_type(ocr_text: str) -> str:
    """Classify document type based on OCR text using heuristics."""
    text_lower = ocr_text.lower()

    # Receipt indicators
    if any(kw in text_lower for kw in ["total", "subtotal", "tax", "cash", "visa", "mastercard", "payment"]):
        return "receipt"

    # Subscription indicators
    if any(kw in text_lower for kw in ["renew", "subscription", "billing", "monthly", "annual"]):
        return "subscription"

    # Warranty indicators
    if any(kw in text_lower for kw in ["warranty", "serial number", "guarantee", "coverage"]):
        return "warranty"

    # Insurance indicators
    if any(kw in text_lower for kw in ["policy", "premium", "coverage", "insured", "beneficiary"]):
        return "insurance"

    return "unknown"


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


async def process_image(db, doc_id: int, s3_key: str) -> None:
    """Process an image through the OCR pipeline."""
    # Download image from S3
    image_bytes = await download_from_s3(s3_key)
    image = Image.open(io.BytesIO(image_bytes))

    # Run PaddleOCR
    ocr = get_ocr_model()
    result = ocr.ocr(image_bytes, cls=True)

    # Extract text and bounding boxes
    doc_text, ocr_blocks = extract_text_and_boxes(result)
    confidence = calculate_confidence(result)

    # Use VLM refinement if confidence is low
    if confidence < 0.7 and doc_text:
        doc_text = await refine_ocr_with_vlm(image_bytes, doc_text)

    # Classify document type
    doc_type = classify_doc_type(doc_text)

    # Update document in database
    await update_document(
        db,
        doc_id,
        doc_text=doc_text,
        doc_type=doc_type,
        ocr_blocks=ocr_blocks,
    )

    # Extract structured fields for receipts
    if doc_type == "receipt":
        fields = await extract_receipt_fields(image_bytes)
        if fields:
            await save_extraction(db, doc_id, doc_type, fields)
