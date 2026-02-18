"""FastAPI application with endpoints for LifeLedger.

Integration notes:
- Frontend handles uploads via presigned URLs (S3 direct) + confirms to its own API
- Backend provides: OCR processing, semantic search, agent queries
- Frontend calls /process after upload to trigger OCR pipeline
"""
import os
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.auth import get_current_user
from app.s3 import upload_to_s3, generate_presigned_url, download_from_s3, delete_many_from_s3
from app.db import get_db, create_document, search_documents, get_user_documents, get_document_by_id, update_document, delete_documents, get_upcoming_events, get_similar_documents
from app.ocr_pipeline import process_image, process_batch_and_crawl
from app.agent import ask_agent
from app.radar_crawler import crawl_documents

# Configure logging for our app modules (uvicorn already configured root)
import logging
import sys

_log_handler = logging.StreamHandler(sys.stdout)
_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

for _logger_name in ['app.ocr_pipeline', 'app.radar_crawler', 'app.vlm_client', 'app.db']:
    _logger = logging.getLogger(_logger_name)
    _logger.setLevel(logging.INFO)
    _logger.addHandler(_log_handler)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: pre-warm OCR model to avoid cold start on first request
    from app.ocr_pipeline import get_ocr_model
    logger.info("Pre-warming PaddleOCR model...")
    get_ocr_model()
    logger.info("PaddleOCR model ready")
    yield
    # Shutdown: cleanup if needed


app = FastAPI(
    title="LifeLedger API",
    description="Backend API for LifeLedger document processing (OCR, search, agent)",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS configuration - allow frontend origins
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class ProcessItem(BaseModel):
    """Single item for processing."""
    s3_key: str
    row_id: str


class ProcessRequest(BaseModel):
    """Request to process documents uploaded via frontend."""
    items: Optional[List[ProcessItem]] = None  # List for multi-item support
    # Backwards compatible: also accept single item
    s3_key: Optional[str] = None
    row_id: Optional[str] = None


class ProcessResponse(BaseModel):
    status: str
    s3_key: str
    doc_type: Optional[str] = None
    message: str


class SearchRequest(BaseModel):
    query: str
    limit: int = 10


class AskRequest(BaseModel):
    question: str


class DeleteRequest(BaseModel):
    document_ids: List[int]


class ReviewRequest(BaseModel):
    note: str = ""  # Optional, max 500 chars


class LineItem(BaseModel):
    description: str
    qty: Optional[str] = None
    unitPrice: Optional[str] = None
    amount: str


class DocumentResponse(BaseModel):
    """Response matching frontend Document interface."""
    id: str
    type: str  # Receipt, Subscription, Invoice, Fine, Form
    fileUrl: str
    status: str  # Processing, Needs Review, Done
    primaryEntity: str
    secondaryEntity: Optional[str] = None
    primaryDate: str
    secondaryDate: Optional[str] = None
    totalValue: str
    lineItems: Optional[List[LineItem]] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchResult(BaseModel):
    answer: str
    documents: List[DocumentResponse]
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: List[str]  # Document IDs


class UploadAndProcessResponse(BaseModel):
    """Response for combined upload and process endpoint."""
    uploaded: List[Dict[str, Any]]
    count: int
    message: str


# Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "lifeledger-api"}


@app.post("/process", response_model=ProcessResponse)
async def process_document(
    request: ProcessRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
):
    """
    Trigger OCR processing for documents uploaded via frontend.
    Supports single item (s3_key/row_id) or multiple items (items list).
    """
    items_to_process = []

    # Handle new multi-item format
    if request.items:
        items_to_process = [{"s3_key": item.s3_key, "row_id": item.row_id} for item in request.items]
    # Handle legacy single-item format
    elif request.s3_key and request.row_id:
        items_to_process = [{"s3_key": request.s3_key, "row_id": request.row_id}]

    # Single background task for batch + crawler
    docs = [{"doc_id": int(item["row_id"]), "s3_key": item["s3_key"]} for item in items_to_process]
    background_tasks.add_task(process_batch_and_crawl, docs, user_id)

    return ProcessResponse(
        status="processing",
        s3_key=items_to_process[0]["s3_key"] if items_to_process else "",
        message=f"OCR processing started for {len(items_to_process)} document(s)",
    )


@app.post("/upload")
async def upload_images(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    user_id: str = Depends(get_current_user),
    db=Depends(get_db),
):
    """
    Direct upload endpoint (alternative to frontend presigned URL flow).
    Use this for testing or if frontend prefers server-side upload.
    """
    uploaded = []

    for file in files:
        content = await file.read()
        s3_key = await upload_to_s3(content, user_id, file.filename)
        doc_id = await create_document(db, user_id, s3_key)
        uploaded.append({"doc_id": doc_id, "s3_key": s3_key, "filename": file.filename})

    return {"uploaded": uploaded, "count": len(uploaded)}


@app.post("/uploadAndProcess", response_model=UploadAndProcessResponse)
async def upload_and_process(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    user_id: str = Depends(get_current_user),
    db=Depends(get_db),
):
    """
    Upload images and immediately start OCR processing.
    Combines /upload and /process into a single call.
    Supports multiple files with safe concurrency (configurable via MAX_CONCURRENT_OCR).
    """
    uploaded = []

    for file in files:
        # Upload to S3
        content = await file.read()
        s3_key = await upload_to_s3(content, user_id, file.filename)

        # Create DB record
        doc_id = await create_document(db, user_id, s3_key)

        uploaded.append({
            "doc_id": doc_id,
            "s3_key": s3_key,
            "filename": file.filename,
            "status": "processing"
        })

    # Single background task for batch + crawler
    background_tasks.add_task(process_batch_and_crawl, uploaded, user_id)

    return UploadAndProcessResponse(
        uploaded=uploaded,
        count=len(uploaded),
        message=f"Uploaded and started processing {len(uploaded)} file(s)"
    )


@app.post("/search", response_model=SearchResult)
async def search(
    request: SearchRequest,
    user_id: str = Depends(get_current_user),
    db=Depends(get_db),
):
    """
    Search documents using semantic similarity + agent answer.
    Returns AI-generated answer + matching documents in frontend format.
    """
    # Get semantic search results
    docs = await search_documents(db, user_id, request.query, request.limit)

    # Generate agent answer
    agent_result = await ask_agent(db, user_id, request.query)

    # Transform to frontend Document format
    frontend_docs = []
    for doc in docs:
        file_url = await generate_presigned_url(doc["s3_key"])
        frontend_docs.append(DocumentResponse(
            id=str(doc["id"]),
            type=_normalize_doc_type(doc.get("doc_type", "unknown")),
            fileUrl=file_url,
            status=_get_status(doc),
            primaryEntity=_extract_primary_entity(doc),
            secondaryEntity=None,
            primaryDate=doc.get("created_at", "")[:10],
            totalValue=_extract_total_value(doc),
            metadata=doc.get("metadata"),
        ))

    return SearchResult(
        answer=agent_result["answer"],
        documents=frontend_docs,
        query=request.query,
    )


@app.get("/documents")
async def list_documents(
    user_id: str = Depends(get_current_user),
    db=Depends(get_db),
    limit: int = 50,
    offset: int = 0,
):
    """List user's processed documents in frontend format."""
    docs = await get_user_documents(db, user_id, limit, offset)

    frontend_docs = []
    for doc in docs:
        file_url = await generate_presigned_url(doc["s3_key"])
        # Use extracted date if available, otherwise fall back to created_at
        primary_date = doc.get("extracted_date") or doc.get("created_at", "")[:10]
        frontend_docs.append({
            "id": str(doc["id"]),
            "type": _normalize_doc_type(doc.get("doc_type", "unknown")),
            "fileUrl": file_url,
            "status": _get_status(doc),
            "primaryEntity": _extract_primary_entity(doc),
            "primaryDate": primary_date,
            "totalValue": _extract_total_value(doc),
        })

    return frontend_docs


@app.get("/documents/{doc_id}")
async def get_document(
    doc_id: str,
    user_id: str = Depends(get_current_user),
    db=Depends(get_db),
):
    """Get a specific document with full details."""
    doc = await get_document_by_id(db, int(doc_id), user_id)

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    file_url = await generate_presigned_url(doc["s3_key"])

    return {
        "id": str(doc["id"]),
        "type": _normalize_doc_type(doc.get("doc_type", "unknown")),
        "fileUrl": file_url,
        "status": _get_status(doc),
        "primaryEntity": _extract_primary_entity(doc),
        "primaryDate": doc.get("created_at", "")[:10],
        "totalValue": _extract_total_value(doc),
        "ocr_blocks": doc.get("ocr_blocks", []),
        "doc_text": doc.get("doc_text"),
    }


@app.get("/documents/{doc_id}/related")
async def get_related_documents(
    doc_id: str,
    limit: int = 4,
    user_id: str = Depends(get_current_user),
    db=Depends(get_db),
):
    """Get documents similar to the given document using vector similarity."""
    docs = await get_similar_documents(db, int(doc_id), user_id, limit)

    related = []
    for doc in docs:
        file_url = await generate_presigned_url(doc["s3_key"])
        related.append({
            "id": str(doc["id"]),
            "type": _normalize_doc_type(doc.get("doc_type", "other")),
            "fileUrl": file_url,
            "status": _get_status(doc),
            "primaryEntity": _extract_primary_entity(doc),
            "primaryDate": doc["created_at"][:10] if doc.get("created_at") else "",
            "totalValue": "",
            "similarity": round(doc.get("similarity", 0) * 100),
        })

    return related


@app.delete("/documents")
async def delete_documents_endpoint(
    request: DeleteRequest,
    user_id: str = Depends(get_current_user),
    db=Depends(get_db),
):
    """Delete multiple documents and their associated S3 objects.

    Only deletes documents owned by the authenticated user.
    """
    if not request.document_ids:
        raise HTTPException(status_code=400, detail="No document IDs provided")

    # Delete from database and get S3 keys
    s3_keys = await delete_documents(db, request.document_ids, user_id)

    if not s3_keys:
        raise HTTPException(status_code=404, detail="No documents found to delete")

    # Delete from S3
    s3_result = await delete_many_from_s3(s3_keys)

    return {
        "deleted_count": len(s3_keys),
        "s3_deleted": s3_result["deleted"],
        "s3_errors": s3_result["errors"],
        "message": f"Deleted {len(s3_keys)} document(s)",
    }


@app.patch("/documents/{doc_id}/review")
async def review_document(
    doc_id: str,
    request: ReviewRequest,
    user_id: str = Depends(get_current_user),
    db=Depends(get_db),
    background_tasks: BackgroundTasks = None,
):
    """Submit manual review for a document. One-time only.

    Updates doc_text with the user's note, re-classifies doc_type,
    and optionally extracts receipt fields or triggers radar crawler.
    """
    # Validate note length
    if len(request.note) > 500:
        raise HTTPException(status_code=400, detail="Note must be 500 characters or less")

    # Get document and verify ownership
    doc = await get_document_by_id(db, int(doc_id), user_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Only allow review for "Needs Review" documents
    current_text = doc.get("doc_text", "")
    if current_text not in ("[Processing failed]", "[No text detected]", ""):
        raise HTTPException(status_code=400, detail="Document not in reviewable state")

    # Update doc_text with user's note (or "[Reviewed]" if empty)
    new_text = request.note.strip() if request.note.strip() else "[Reviewed]"

    # Re-classify doc_type based on user's text
    from app.ocr_pipeline import classify_doc_type
    new_doc_type = classify_doc_type(new_text)

    # Update document with embedding generation (reuse existing ocr_blocks)
    await update_document(
        db,
        int(doc_id),
        doc_text=new_text,
        doc_type=new_doc_type,
        ocr_blocks=doc.get("ocr_blocks", []),
    )

    # If it looks like a receipt, extract fields from text (no image needed)
    if new_doc_type == "receipt" and request.note.strip():
        from app.vlm_client import extract_receipt_from_text
        from app.extraction import save_extraction
        try:
            extraction = await extract_receipt_from_text(request.note)
            if extraction:
                await save_extraction(db, int(doc_id), "receipt", extraction)
        except Exception as e:
            logger.warning(f"Text extraction failed for doc {doc_id}: {e}")

    # Optionally trigger crawler if note has date patterns/keywords
    if request.note.strip() and background_tasks:
        from app.radar_crawler import has_event_keywords, has_date_pattern
        if has_event_keywords(request.note) or has_date_pattern(request.note):
            # Reset radar_processed so crawler picks it up
            await db.execute(
                "UPDATE documents SET radar_processed = FALSE WHERE id = $1",
                int(doc_id)
            )
            # Run crawler for this user in background
            background_tasks.add_task(crawl_documents, user_id=user_id, limit=5)

    return {"status": "ok", "message": "Review submitted"}


@app.get("/radar")
async def get_radar(
    user_id: str = Depends(get_current_user),
    db=Depends(get_db),
    days: int = 30,
):
    """Get upcoming events/deadlines from user documents within the next N days."""
    events = await get_upcoming_events(db, user_id, days_ahead=days)

    radar_items = []
    for event in events:
        file_url = await generate_presigned_url(event["s3_key"])
        radar_items.append({
            "id": str(event["id"]),
            "type": _normalize_doc_type(event.get("doc_type", "unknown")),
            "fileUrl": file_url,
            "primaryEntity": event.get("event_description") or event.get("event_entity") or event.get("merchant") or "Unknown",
            "date": event.get("event_date"),
            "description": event.get("event_description") or "",
            "totalValue": f"${event['total_amount']:.2f}" if event.get("total_amount") else "",
        })

    return {"events": radar_items, "count": len(radar_items)}


@app.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    user_id: str = Depends(get_current_user),
    db=Depends(get_db),
):
    """Ask analytical questions about documents using the agent."""
    result = await ask_agent(db, user_id, request.question)
    return AskResponse(answer=result["answer"], sources=[str(s) for s in result["sources"]])


@app.post("/internal/crawl-radar")
async def trigger_radar_crawl(
    limit: int = 50,
    user_id: str = Depends(get_current_user),
):
    """Trigger radar crawler to process documents for event dates.

    The crawler:
    1. Finds documents with text but not yet radar-processed
    2. Uses keyword filter to skip docs without deadline-related words
    3. Calls text-only LLM to extract event dates from remaining docs
    4. Updates event_date column for radar display

    Can also be called via cron for background processing.
    """
    stats = await crawl_documents(user_id=user_id, limit=limit)
    return {"status": "completed", **stats}


# Helper functions for frontend format conversion
def _get_status(doc: dict) -> str:
    """Determine document status from doc_text."""
    text = doc.get("doc_text", "")
    if not text:
        return "Processing"
    if text in ("[Processing failed]", "[No text detected]"):
        return "Needs Review"
    return "Done"


def _normalize_doc_type(doc_type: str) -> str:
    """Convert internal doc_type to frontend format."""
    mapping = {
        "receipt": "Receipt",
        "subscription": "Subscription",
        "invoice": "Invoice",
        "fine": "Fine",
        "form": "Form",
        "warranty": "Form",
        "insurance": "Form",
        "other": "Other",
        "unknown": "Other",
    }
    return mapping.get(doc_type.lower(), "Form")


def _extract_primary_entity(doc: dict) -> str:
    """Extract primary entity (merchant name, etc.) from document."""
    # Try extraction data first
    if doc.get("merchant"):
        return doc["merchant"]

    # Fall back to first line of OCR text
    text = doc.get("doc_text", "")
    if text:
        # Handle failed processing
        if text == "[Processing failed]":
            return "Processing Failed"
        first_line = text.split("\n")[0].strip()
        return first_line[:50] if first_line else "Unknown Document"

    return "Processing..."


def _extract_total_value(doc: dict) -> str:
    """Extract total value from document."""
    if doc.get("total_amount"):
        return f"${doc['total_amount']:.2f}"
    return ""
