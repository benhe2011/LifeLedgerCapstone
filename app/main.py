"""FastAPI application with endpoints for LifeLedger."""
import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.auth import get_current_user
from app.s3 import upload_to_s3, generate_presigned_url
from app.db import get_db, create_document, search_documents, get_user_documents
from app.ocr_pipeline import process_image
from app.agent import ask_agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize OCR model (lazy load on first use)
    yield
    # Shutdown: cleanup if needed


app = FastAPI(
    title="LifeLedger API",
    description="Backend API for LifeLedger document processing",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS configuration
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class SearchRequest(BaseModel):
    query: str
    limit: int = 10


class AskRequest(BaseModel):
    question: str


class DocumentResponse(BaseModel):
    id: int
    s3_key: str
    doc_type: str
    doc_text: str | None
    created_at: str
    presigned_url: str | None = None


class SearchResult(BaseModel):
    documents: List[DocumentResponse]
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: List[int]


# Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "lifeledger-api"}


@app.post("/upload")
async def upload_images(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    user_id: str = Depends(get_current_user),
    db=Depends(get_db),
):
    """Upload images for processing."""
    uploaded = []

    for file in files:
        # Read file content
        content = await file.read()

        # Upload to S3
        s3_key = await upload_to_s3(content, user_id, file.filename)

        # Create document record (processing happens in background)
        doc_id = await create_document(db, user_id, s3_key)

        # Process image in background
        background_tasks.add_task(process_image, db, doc_id, s3_key)

        uploaded.append({"doc_id": doc_id, "s3_key": s3_key, "filename": file.filename})

    return {"uploaded": uploaded, "count": len(uploaded)}


@app.post("/search", response_model=SearchResult)
async def search(
    request: SearchRequest,
    user_id: str = Depends(get_current_user),
    db=Depends(get_db),
):
    """Search documents using semantic + keyword search."""
    docs = await search_documents(db, user_id, request.query, request.limit)

    # Add presigned URLs for image access
    for doc in docs:
        doc["presigned_url"] = await generate_presigned_url(doc["s3_key"])

    return SearchResult(documents=docs, query=request.query)


@app.get("/documents", response_model=List[DocumentResponse])
async def list_documents(
    user_id: str = Depends(get_current_user),
    db=Depends(get_db),
    limit: int = 50,
    offset: int = 0,
):
    """List user's processed documents."""
    docs = await get_user_documents(db, user_id, limit, offset)

    for doc in docs:
        doc["presigned_url"] = await generate_presigned_url(doc["s3_key"])

    return docs


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
    return AskResponse(answer=result["answer"], sources=result["sources"])
