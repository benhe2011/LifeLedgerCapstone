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
from app.s3 import upload_to_s3, generate_presigned_url, download_from_s3
from app.db import get_db, create_document, search_documents, get_user_documents, get_document_by_id, update_document
from app.ocr_pipeline import process_image
from app.agent import ask_agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize OCR model (lazy load on first use)
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
class ProcessRequest(BaseModel):
    """Request to process a document that was uploaded via frontend."""
    s3_key: str
    row_id: Optional[str] = None  # Frontend's document ID


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
    Trigger OCR processing for a document uploaded via frontend.

    Frontend flow:
    1. Frontend gets presigned URL from its API
    2. Frontend uploads directly to S3
    3. Frontend confirms upload to its API (writes to DB)
    4. Frontend calls this endpoint to trigger OCR processing
    """
    # Add OCR processing to background tasks
    if request.row_id:
        background_tasks.add_task(
            process_image,
            int(request.row_id),
            request.s3_key,
        )

    return ProcessResponse(
        status="processing",
        s3_key=request.s3_key,
        message="OCR processing started in background",
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
            status="Done" if doc.get("doc_text") else "Processing",
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
        frontend_docs.append({
            "id": str(doc["id"]),
            "type": _normalize_doc_type(doc.get("doc_type", "unknown")),
            "fileUrl": file_url,
            "status": "Done" if doc.get("doc_text") else "Processing",
            "primaryEntity": _extract_primary_entity(doc),
            "primaryDate": doc.get("created_at", "")[:10],
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
        "status": "Done" if doc.get("doc_text") else "Processing",
        "primaryEntity": _extract_primary_entity(doc),
        "primaryDate": doc.get("created_at", "")[:10],
        "totalValue": _extract_total_value(doc),
        "ocr_blocks": doc.get("ocr_blocks", []),
        "doc_text": doc.get("doc_text"),
    }


@app.get("/radar")
async def get_radar(
    user_id: str = Depends(get_current_user),
    db=Depends(get_db),
):
    """Get upcoming deadlines from user documents."""
    # TODO: Implement deadline extraction and radar
    return {"deadlines": [], "message": "Radar feature coming soon"}


@app.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    user_id: str = Depends(get_current_user),
    db=Depends(get_db),
):
    """Ask analytical questions about documents using the agent."""
    result = await ask_agent(db, user_id, request.question)
    return AskResponse(answer=result["answer"], sources=[str(s) for s in result["sources"]])


# Helper functions for frontend format conversion
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
        "unknown": "Form",
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
        first_line = text.split("\n")[0].strip()
        return first_line[:50] if first_line else "Unknown Document"

    return "Unknown Document"


def _extract_total_value(doc: dict) -> str:
    """Extract total value from document."""
    if doc.get("total_amount"):
        return f"${doc['total_amount']:.2f}"
    return ""
