"""PostgreSQL database operations with pgvector support."""
import os
import json
from typing import List, Dict, Any
from contextlib import asynccontextmanager

import asyncpg
from sentence_transformers import SentenceTransformer


# Connection pool
_pool: asyncpg.Pool | None = None

# Embedding model (lazy loaded)
_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Get or initialize embedding model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


async def init_db() -> asyncpg.Pool:
    """Initialize database connection pool."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            os.getenv("DATABASE_URL"),
            min_size=2,
            max_size=10,
        )
    return _pool


async def get_db():
    """Dependency to get database connection."""
    pool = await init_db()
    async with pool.acquire() as conn:
        yield conn


async def create_tables():
    """Create database tables if they don't exist."""
    pool = await init_db()
    async with pool.acquire() as conn:
        # Enable pgvector extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        # Create documents table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                s3_key TEXT NOT NULL,
                doc_text TEXT,
                text_vector vector(384),
                doc_type TEXT DEFAULT 'unknown',
                ocr_blocks JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        # Create index on user_id
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id)
        """)

        # Create vector index for similarity search
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_vector
            ON documents USING ivfflat (text_vector vector_cosine_ops)
            WITH (lists = 100)
        """)

        # Create extractions table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS extractions (
                id SERIAL PRIMARY KEY,
                doc_id INTEGER REFERENCES documents(id),
                doc_type TEXT,
                merchant TEXT,
                date DATE,
                total_amount DECIMAL,
                address TEXT
            )
        """)


async def create_document(conn, user_id: str, s3_key: str) -> int:
    """Create a new document record."""
    query = """
        INSERT INTO documents (user_id, s3_key)
        VALUES ($1, $2)
        RETURNING id
    """
    return await conn.fetchval(query, user_id, s3_key)


async def update_document(
    conn,
    doc_id: int,
    doc_text: str,
    doc_type: str,
    ocr_blocks: List[Dict[str, Any]],
) -> None:
    """Update document with OCR results and embedding."""
    # Generate embedding
    model = get_embedding_model()
    embedding = model.encode(doc_text).tolist()

    query = """
        UPDATE documents
        SET doc_text = $2,
            doc_type = $3,
            ocr_blocks = $4,
            text_vector = $5
        WHERE id = $1
    """
    await conn.execute(
        query,
        doc_id,
        doc_text,
        doc_type,
        json.dumps(ocr_blocks),
        embedding,
    )


async def search_documents(
    conn,
    user_id: str,
    query: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Search documents using semantic similarity."""
    # Generate query embedding
    model = get_embedding_model()
    query_embedding = model.encode(query).tolist()

    sql = """
        SELECT id, s3_key, doc_type, doc_text, created_at,
               1 - (text_vector <=> $3) as similarity
        FROM documents
        WHERE user_id = $1 AND text_vector IS NOT NULL
        ORDER BY text_vector <=> $3
        LIMIT $2
    """
    rows = await conn.fetch(sql, user_id, limit, query_embedding)

    return [
        {
            "id": row["id"],
            "s3_key": row["s3_key"],
            "doc_type": row["doc_type"],
            "doc_text": row["doc_text"],
            "created_at": row["created_at"].isoformat(),
            "similarity": float(row["similarity"]),
        }
        for row in rows
    ]


async def get_user_documents(
    conn,
    user_id: str,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Get all documents for a user."""
    query = """
        SELECT id, s3_key, doc_type, doc_text, created_at
        FROM documents
        WHERE user_id = $1
        ORDER BY created_at DESC
        LIMIT $2 OFFSET $3
    """
    rows = await conn.fetch(query, user_id, limit, offset)

    return [
        {
            "id": row["id"],
            "s3_key": row["s3_key"],
            "doc_type": row["doc_type"],
            "doc_text": row["doc_text"],
            "created_at": row["created_at"].isoformat(),
        }
        for row in rows
    ]


async def get_document_by_id(conn, doc_id: int, user_id: str) -> Dict[str, Any] | None:
    """Get a specific document by ID (with user validation)."""
    query = """
        SELECT id, s3_key, doc_type, doc_text, ocr_blocks, created_at
        FROM documents
        WHERE id = $1 AND user_id = $2
    """
    row = await conn.fetchrow(query, doc_id, user_id)

    if not row:
        return None

    return {
        "id": row["id"],
        "s3_key": row["s3_key"],
        "doc_type": row["doc_type"],
        "doc_text": row["doc_text"],
        "ocr_blocks": json.loads(row["ocr_blocks"]) if row["ocr_blocks"] else [],
        "created_at": row["created_at"].isoformat(),
    }
