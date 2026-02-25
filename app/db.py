"""PostgreSQL database operations with pgvector support."""
import os
import ssl
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


def get_ssl_context():
    """Create SSL context for Aurora connection."""
    ssl_context = ssl.create_default_context(cafile="/certs/global-bundle.pem")
    ssl_context.check_hostname = True
    ssl_context.verify_mode = ssl.CERT_REQUIRED
    return ssl_context


async def init_db() -> asyncpg.Pool:
    """Initialize database connection pool."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            os.getenv("DATABASE_URL"),
            min_size=2,
            max_size=10,
            ssl=get_ssl_context(),
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

        # Feedback loop tables: track agent responses across regeneration sessions
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS regenerate_sessions (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                query_text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS regenerate_attempts (
                id SERIAL PRIMARY KEY,
                session_id INTEGER NOT NULL REFERENCES regenerate_sessions(id),
                attempt_number INTEGER NOT NULL,
                answer TEXT NOT NULL,
                tool_trace JSONB,
                created_at TIMESTAMP DEFAULT NOW()
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
            text_vector = $5::vector
        WHERE id = $1
    """
    await conn.execute(
        query,
        doc_id,
        doc_text,
        doc_type,
        json.dumps(ocr_blocks),
        str(embedding),
    )


async def search_documents(
    conn,
    user_id: str,
    query: str,
    limit: int = 10,
    min_similarity: float = 0.3,
) -> List[Dict[str, Any]]:
    """Search documents using semantic similarity.

    Args:
        min_similarity: Minimum similarity threshold (0.0-1.0). Default 0.3 (30%).
    """
    # Generate query embedding
    model = get_embedding_model()
    query_embedding = model.encode(query).tolist()

    sql = """
        SELECT id, s3_key, doc_type, doc_text, created_at,
               1 - (text_vector <=> $3::vector) as similarity
        FROM documents
        WHERE user_id = $1
          AND text_vector IS NOT NULL
          AND 1 - (text_vector <=> $3::vector) >= $4
          AND doc_text NOT IN ('[No text detected]', '[Processing failed]')
        ORDER BY text_vector <=> $3::vector
        LIMIT $2
    """
    rows = await conn.fetch(sql, user_id, limit, str(query_embedding), min_similarity)

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


async def get_similar_documents(
    conn,
    doc_id: int,
    user_id: str,
    limit: int = 4,
) -> List[Dict[str, Any]]:
    """Get documents similar to the given document using vector similarity."""
    # First get the source document's vector
    source = await conn.fetchrow(
        "SELECT text_vector FROM documents WHERE id = $1 AND user_id = $2",
        doc_id, user_id
    )
    if not source or not source["text_vector"]:
        return []

    source_vector = source["text_vector"]

    # Find similar documents (excluding the source document)
    rows = await conn.fetch(
        """
        SELECT id, s3_key, doc_type, doc_text, created_at,
               1 - (text_vector <=> $1::vector) as similarity
        FROM documents
        WHERE user_id = $2
          AND id != $3
          AND text_vector IS NOT NULL
        ORDER BY text_vector <=> $1::vector
        LIMIT $4
        """,
        str(source_vector), user_id, doc_id, limit
    )

    return [
        {
            "id": row["id"],
            "s3_key": row["s3_key"],
            "doc_type": row["doc_type"],
            "doc_text": row["doc_text"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
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
    """Get all documents for a user with extraction data."""
    query = """
        SELECT d.id, d.s3_key, d.doc_type, d.doc_text, d.created_at,
               e.merchant, e.date as extracted_date, e.total_amount
        FROM documents d
        LEFT JOIN extractions e ON d.id = e.doc_id
        WHERE d.user_id = $1
        ORDER BY d.created_at DESC
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
            "merchant": row["merchant"],
            "extracted_date": row["extracted_date"].isoformat() if row["extracted_date"] else None,
            "total_amount": float(row["total_amount"]) if row["total_amount"] else None,
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


async def get_upcoming_events(
    conn,
    user_id: str,
    days_ahead: int = 30,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Get documents with event dates in the upcoming period (for radar).

    Uses event_date column on documents table (populated by radar crawler)
    to find upcoming deadlines, renewals, expirations, etc.
    Works for ANY document type, not just receipts.
    """
    from datetime import date, timedelta

    today = date.today()
    end_date = today + timedelta(days=days_ahead)

    query = """
        SELECT d.id, d.s3_key, d.doc_type, d.doc_text,
               d.event_date, d.event_description, d.event_entity,
               e.merchant, e.total_amount
        FROM documents d
        LEFT JOIN extractions e ON d.id = e.doc_id
        WHERE d.user_id = $1
          AND d.event_date IS NOT NULL
          AND d.event_date >= $2
          AND d.event_date <= $3
        ORDER BY d.event_date ASC
        LIMIT $4
    """
    rows = await conn.fetch(query, user_id, today, end_date, limit)

    return [
        {
            "id": row["id"],
            "s3_key": row["s3_key"],
            "doc_type": row["doc_type"],
            "doc_text": row["doc_text"],
            "event_date": row["event_date"].isoformat() if row["event_date"] else None,
            "event_description": row["event_description"],
            "merchant": row["event_entity"] or row["merchant"],  # Prefer event_entity, fallback to extraction merchant
            "total_amount": float(row["total_amount"]) if row["total_amount"] else None,
        }
        for row in rows
    ]


async def get_documents_for_delete(
    conn,
    doc_ids: List[int],
    user_id: str,
) -> tuple[List[int], List[str]]:
    """Get valid doc IDs and S3 keys for deletion.

    Returns (valid_ids, s3_keys) for documents owned by the specified user_id.
    """
    if not doc_ids:
        return [], []

    query = """
        SELECT id, s3_key FROM documents
        WHERE id = ANY($1::int[]) AND user_id = $2
    """
    rows = await conn.fetch(query, doc_ids, user_id)

    if not rows:
        return [], []

    valid_ids = [row["id"] for row in rows]
    s3_keys = [row["s3_key"] for row in rows]
    return valid_ids, s3_keys


async def delete_document_records(
    conn,
    doc_ids: List[int],
    user_id: str,
) -> None:
    """Delete extractions and documents from DB.

    Call this AFTER deleting from S3 to avoid orphaned files.
    """
    if not doc_ids:
        return

    # Delete extractions first (child table with FK constraint)
    await conn.execute(
        "DELETE FROM extractions WHERE doc_id = ANY($1::int[])",
        doc_ids
    )

    # Delete documents
    await conn.execute(
        "DELETE FROM documents WHERE id = ANY($1::int[]) AND user_id = $2",
        doc_ids,
        user_id
    )


async def create_regenerate_session(conn, user_id: str, query_text: str, answer: str, tool_trace: list) -> int:
    """Create a new regenerate session and log the initial (attempt 0) response.

    Called by /search to start tracking. Returns the session_id.
    """
    session_id = await conn.fetchval(
        """
        INSERT INTO regenerate_sessions (user_id, query_text)
        VALUES ($1, $2)
        RETURNING id
        """,
        user_id, query_text,
    )
    await conn.execute(
        """
        INSERT INTO regenerate_attempts (session_id, attempt_number, answer, tool_trace)
        VALUES ($1, 0, $2, $3::jsonb)
        """,
        session_id, answer, json.dumps(tool_trace),
    )
    return session_id


async def log_regenerate_attempt(conn, session_id: int, answer: str, tool_trace: list) -> int:
    """Log a regeneration attempt for an existing session.

    Auto-increments attempt_number. Returns the new attempt number.
    """
    attempt_number = await conn.fetchval(
        """
        INSERT INTO regenerate_attempts (session_id, attempt_number, answer, tool_trace)
        VALUES ($1, (SELECT COALESCE(MAX(attempt_number), -1) + 1 FROM regenerate_attempts WHERE session_id = $1), $2, $3::jsonb)
        RETURNING attempt_number
        """,
        session_id, answer, json.dumps(tool_trace),
    )
    return attempt_number
