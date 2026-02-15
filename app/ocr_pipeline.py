"""OCR pipeline using PaddleOCR with document classification.

Supports two flows:
1. Direct upload: process_image() - when backend handles upload
2. Frontend upload: process_image_from_s3() - when frontend uploads to S3 directly
"""
import io
import os
import logging
from typing import Dict, List, Any, Optional

from PIL import Image
from paddleocr import PaddleOCR

from app.s3 import download_from_s3
from app.vlm_client import refine_ocr_with_vlm, extract_receipt_fields


logger = logging.getLogger(__name__)

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


async def process_image_from_s3(
    s3_key: str,
    row_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process an image from S3 through the OCR pipeline.

    This is called after frontend uploads to S3 and confirms in its DB.
    Updates the document record via RDS Data API (same as frontend uses).

    Args:
        s3_key: S3 key of the uploaded image
        row_id: Frontend's document ID (row_id in documents table)
        user_id: User ID for logging/verification

    Returns:
        Dict with processing results
    """
    try:
        logger.info(f"Processing image: {s3_key}")

        # Download image from S3
        image_bytes = await download_from_s3(s3_key)

        # Run OCR and get results
        result = await run_ocr_pipeline(image_bytes)

        # Update document in database via RDS Data API
        if row_id:
            await update_document_rds(
                row_id=row_id,
                doc_text=result["doc_text"],
                doc_type=result["doc_type"],
                ocr_blocks=result["ocr_blocks"],
            )

        logger.info(f"Processed {s3_key}: type={result['doc_type']}, confidence={result['confidence']:.2f}")

        return result

    except Exception as e:
        logger.error(f"Error processing {s3_key}: {e}")
        raise


async def run_ocr_pipeline(image_bytes: bytes) -> Dict[str, Any]:
    """Run OCR pipeline on image bytes and return results."""
    # Run PaddleOCR
    ocr = get_ocr_model()
    result = ocr.ocr(image_bytes, cls=True)

    # Extract text and bounding boxes
    doc_text, ocr_blocks = extract_text_and_boxes(result)
    confidence = calculate_confidence(result)

    # Use VLM refinement if confidence is low
    if confidence < 0.7 and doc_text:
        try:
            doc_text = await refine_ocr_with_vlm(image_bytes, doc_text)
        except Exception as e:
            logger.warning(f"VLM refinement failed: {e}")

    # Classify document type
    doc_type = classify_doc_type(doc_text)

    # Extract structured fields for receipts
    extraction = None
    if doc_type == "receipt":
        try:
            extraction = await extract_receipt_fields(image_bytes)
        except Exception as e:
            logger.warning(f"Receipt extraction failed: {e}")

    return {
        "doc_text": doc_text,
        "doc_type": doc_type,
        "ocr_blocks": ocr_blocks,
        "confidence": confidence,
        "extraction": extraction,
    }


async def update_document_rds(
    row_id: str,
    doc_text: str,
    doc_type: str,
    ocr_blocks: List[Dict[str, Any]],
) -> None:
    """Update document in Aurora via RDS Data API (matches frontend pattern)."""
    import json
    import boto3

    rds_client = boto3.client(
        "rds-data",
        region_name=os.getenv("AWS_REGION", "us-west-2"),
    )

    cluster_arn = os.getenv("AWS_RDS_CLUSTER_ARN")
    secret_arn = os.getenv("AWS_RDS_SECRET_ARN")
    database = os.getenv("AWS_RDS_DATABASE")

    if not all([cluster_arn, secret_arn, database]):
        logger.warning("RDS Data API not configured, skipping DB update")
        return

    sql = """
        UPDATE documents
        SET doc_text = :docText,
            doc_type = :docType,
            ocr_blocks = :ocrBlocks
        WHERE row_id = :rowId
    """

    rds_client.execute_statement(
        resourceArn=cluster_arn,
        secretArn=secret_arn,
        database=database,
        sql=sql,
        parameters=[
            {"name": "docText", "value": {"stringValue": doc_text}},
            {"name": "docType", "value": {"stringValue": doc_type}},
            {"name": "ocrBlocks", "value": {"stringValue": json.dumps(ocr_blocks)}},
            {"name": "rowId", "value": {"stringValue": row_id}},
        ],
    )

    logger.info(f"Updated document {row_id} in database")


async def process_image(doc_id: int, s3_key: str) -> None:
    """
    Process an image through the OCR pipeline (direct DB connection version).
    Used when backend handles both upload and DB writes.
    Acquires its own database connection for use in background tasks.
    """
    from app.db import update_document, init_db
    from app.extraction import save_extraction

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
