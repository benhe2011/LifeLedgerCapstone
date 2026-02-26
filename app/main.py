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
from app.db import get_db, create_tables, create_document, search_documents, get_user_documents, get_document_by_id, update_document, get_documents_for_delete, delete_document_records, get_upcoming_events, get_similar_documents, create_regenerate_session, log_regenerate_attempt, count_unmined_sessions
from app.ocr_pipeline import process_image, process_batch_and_crawl
from app.agent import ask_agent
from app.radar_crawler import crawl_documents
from app.content_safety import moderate_image, moderate_user_text
from app.extraction import (
    get_spending_by_merchant,
    detect_recurring_costs,
    detect_trips,
)

# Configure logging for our app modules (uvicorn already configured root)
import logging
import sys

_log_handler = logging.StreamHandler(sys.stdout)
_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

for _logger_name in ['app.ocr_pipeline', 'app.radar_crawler', 'app.vlm_client', 'app.db', 'app.content_safety', 'app.agent']:
    _logger = logging.getLogger(_logger_name)
    _logger.setLevel(logging.INFO)
    _logger.addHandler(_log_handler)

logger = logging.getLogger(__name__)

# Auto-trigger constraint mining after this many unmined regeneration sessions accumulate
MINING_THRESHOLD = int(os.getenv("MINING_THRESHOLD", "20"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure all tables exist (uses IF NOT EXISTS, safe to run every time)
    await create_tables()
    # Pre-warm OCR model to avoid cold start on first request
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


class SafetyInfo(BaseModel):
    """Content safety metadata when a response is blocked."""
    strategy: str
    message: str
    detail: Optional[str] = None


class GroundednessInfo(BaseModel):
    """Groundedness warning metadata."""
    ungrounded_pct: float
    message: str


class SearchResult(BaseModel):
    answer: str
    documents: List[DocumentResponse]
    query: str
    session_id: int  # Feedback loop session ID for regeneration tracking
    safety: Optional[SafetyInfo] = None
    groundedness: Optional[GroundednessInfo] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[str]  # Document IDs
    safety: Optional[SafetyInfo] = None
    groundedness: Optional[GroundednessInfo] = None


class RegenerateRequest(BaseModel):
    session_id: int


class RegenerateResult(BaseModel):
    answer: str
    safety: Optional[SafetyInfo] = None
    groundedness: Optional[GroundednessInfo] = None

class RejectedFile(BaseModel):
    """A file rejected by content safety moderation."""
    filename: str
    message: str

class UploadAndProcessResponse(BaseModel):
    """Response for combined upload and process endpoint."""
    uploaded: List[Dict[str, Any]]
    count: int
    message: str
    rejected: List[RejectedFile] = []


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
    rejected = []

    for file in files:
        content = await file.read()

        # ── Content Safety: image moderation ──
        gate_result = await moderate_image(content)
        if not gate_result.is_safe:
            logger.warning("Image rejected for %s: %s", file.filename, gate_result.message)
            rejected.append({"filename": file.filename, "message": gate_result.message})
            continue

        s3_key = await upload_to_s3(content, user_id, file.filename)
        doc_id = await create_document(db, user_id, s3_key)
        uploaded.append({"doc_id": doc_id, "s3_key": s3_key, "filename": file.filename})

    return {"uploaded": uploaded, "count": len(uploaded), "rejected": rejected}


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
    rejected = []

    for file in files:
        # Upload to S3
        content = await file.read()

        # ── Content Safety: image moderation ──
        gate_result = await moderate_image(content)
        if not gate_result.is_safe:
            logger.warning("Image rejected for %s: %s", file.filename, gate_result.message)
            rejected.append(RejectedFile(filename=file.filename, message=gate_result.message))
            continue

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
        message=f"Uploaded and started processing {len(uploaded)} file(s)",
        rejected=rejected,
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

    # Create feedback loop session (attempt 0 = initial search response)
    session_id = await create_regenerate_session(
        db, user_id, request.query,
        agent_result["answer"],
        agent_result.get("tool_trace", []),
    )

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

    safety = None
    if agent_result.get("safety"):
        safety = SafetyInfo(**agent_result["safety"])

    groundedness = None
    if agent_result.get("groundedness"):
        groundedness = GroundednessInfo(**agent_result["groundedness"])

    return SearchResult(
        answer=agent_result["answer"],
        documents=frontend_docs,
        query=request.query,
        session_id=session_id,
        safety=safety,
        groundedness=groundedness,
    )


@app.post("/regenerate", response_model=RegenerateResult)
async def regenerate(
    request: RegenerateRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
    db=Depends(get_db),
):
    """Regenerate AI answer and log the attempt for the feedback loop."""
    # Look up the original query from the session
    session = await db.fetchrow(
        "SELECT query_text FROM regenerate_sessions WHERE id = $1 AND user_id = $2",
        request.session_id, user_id,
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Rate limit: max 3 regenerates per session in last 60 seconds
    recent = await db.fetchval(
        """SELECT COUNT(*) FROM regenerate_attempts
           WHERE session_id = $1 AND created_at > NOW() - INTERVAL '60 seconds'""",
        request.session_id,
    )
    if recent and recent >= 3:
        raise HTTPException(status_code=429, detail="Too many regenerate requests. Please wait a minute.")

    agent_result = await ask_agent(db, user_id, session["query_text"])

    # Log this attempt
    await log_regenerate_attempt(
        db, request.session_id,
        agent_result["answer"],
        agent_result.get("tool_trace", []),
    )

    # Auto-trigger constraint mining when threshold is reached
    unmined = await count_unmined_sessions(db)
    if unmined >= MINING_THRESHOLD:
        logger.info("Mining threshold reached (%d >= %d), triggering background mining", unmined, MINING_THRESHOLD)
        background_tasks.add_task(_run_mining_background, db)

    safety = None
    if agent_result.get("safety"):
        safety = SafetyInfo(**agent_result["safety"])

    groundedness = None
    if agent_result.get("groundedness"):
        groundedness = GroundednessInfo(**agent_result["groundedness"])

    return RegenerateResult(
        answer=agent_result["answer"],
        safety=safety,
        groundedness=groundedness,
    )


async def _run_mining_background(db):
    """Background task wrapper for constraint mining."""
    try:
        from app.prompt_optimizer import run_mining_job
        stats = await run_mining_job(db)
        logger.info("Background mining complete: %s", stats)
    except Exception as e:
        logger.error("Background mining failed: %s", e)


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
    Deletes S3 first, then DB to avoid orphaned files.
    """
    if not request.document_ids:
        raise HTTPException(status_code=400, detail="No document IDs provided")

    # 1. Get S3 keys (query only, no delete yet)
    valid_ids, s3_keys = await get_documents_for_delete(db, request.document_ids, user_id)

    if not valid_ids:
        # Idempotent: return success even if already deleted
        return {"deleted_count": 0, "s3_deleted": 0, "s3_errors": 0, "message": "No documents found"}

    # 2. Delete from S3 FIRST (so DB record exists if this fails)
    s3_result = await delete_many_from_s3(s3_keys)

    # 3. Delete from DB (extractions, then documents)
    await delete_document_records(db, valid_ids, user_id)

    return {
        "deleted_count": len(valid_ids),
        "s3_deleted": s3_result["deleted"],
        "s3_errors": s3_result["errors"],
        "message": f"Deleted {len(valid_ids)} document(s)",
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

    # ── Content Safety: moderate review note ──
    if request.note.strip():
        gate_result = await moderate_user_text(request.note)
        if not gate_result.is_safe:
            raise HTTPException(status_code=400, detail=gate_result.message)

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

    safety = None
    if result.get("safety"):
        safety = SafetyInfo(**result["safety"])

    groundedness = None
    if result.get("groundedness"):
        groundedness = GroundednessInfo(**result["groundedness"])

    return AskResponse(
        answer=result["answer"],
        sources=[str(s) for s in result["sources"]],
        safety=safety,
        groundedness=groundedness,
    )


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


@app.post("/internal/mine-constraints")
async def trigger_constraint_mining(
    user_id: str = Depends(get_current_user),
    db=Depends(get_db),
):
    """Run the constraint mining batch job.

    Processes unmined regeneration sessions to extract behavioral
    constraints from rejected/accepted response pairs. Mined constraints
    are automatically injected into the agent's system prompt.

    Can be triggered manually or via a scheduled job.
    """
    from app.prompt_optimizer import run_mining_job
    stats = await run_mining_job(db)
    return {"status": "completed", **stats}


@app.get("/analytics/spending")
async def analytics_spending(
    months: int = 6,
    user_id: str = Depends(get_current_user),
    db=Depends(get_db),
):
    """Get spending analytics: by merchant and by month."""
    from datetime import datetime as dt, timedelta
    end_date = dt.now().date()
    start_date = end_date - timedelta(days=30 * months)
    by_merchant = await get_spending_by_merchant(
        db, user_id,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        limit=10,
    )

    # Monthly totals
    rows = await db.fetch(
        """SELECT TO_CHAR(e.date, 'YYYY-MM') as month, SUM(e.total_amount) as total
           FROM extractions e JOIN documents d ON e.doc_id = d.id
           WHERE d.user_id = $1 AND e.total_amount IS NOT NULL AND e.date IS NOT NULL
             AND e.date >= CURRENT_DATE - make_interval(months => $2)
           GROUP BY month ORDER BY month""",
        user_id, months,
    )
    by_month = [{"month": r["month"], "total": float(r["total"])} for r in rows]

    grand_total = sum(m["total"] for m in by_month)

    return {
        "by_merchant": by_merchant,
        "by_month": by_month,
        "total": round(grand_total, 2),
    }


@app.get("/analytics/recurring")
async def analytics_recurring(
    user_id: str = Depends(get_current_user),
    db=Depends(get_db),
):
    """Get detected recurring subscriptions and charges."""
    recurring = await detect_recurring_costs(db, user_id)
    total_monthly = sum(r["monthly_estimate"] for r in recurring)
    total_annual = sum(r["annual_estimate"] for r in recurring)

    return {
        "recurring": recurring,
        "total_monthly": round(total_monthly, 2),
        "total_annual": round(total_annual, 2),
        "count": len(recurring),
    }


@app.get("/analytics/trips")
async def analytics_trips(
    user_id: str = Depends(get_current_user),
    db=Depends(get_db),
):
    """Get detected travel trips grouped by date proximity."""
    trips = await detect_trips(db, user_id)
    total_trip_spending = sum(t["total_cost"] for t in trips)

    return {
        "trips": trips,
        "total_trip_spending": round(total_trip_spending, 2),
        "count": len(trips),
    }


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
