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

        # Add phash column for upload-time deduplication (safe if already exists)
        await conn.execute("""
            ALTER TABLE documents ADD COLUMN IF NOT EXISTS phash TEXT
        """)

        # Add radar crawler columns (safe if already exists)
        await conn.execute("""
            ALTER TABLE documents ADD COLUMN IF NOT EXISTS radar_processed BOOLEAN DEFAULT FALSE
        """)
        await conn.execute("""
            ALTER TABLE documents ADD COLUMN IF NOT EXISTS event_date DATE
        """)
        await conn.execute("""
            ALTER TABLE documents ADD COLUMN IF NOT EXISTS event_description TEXT
        """)
        await conn.execute("""
            ALTER TABLE documents ADD COLUMN IF NOT EXISTS event_entity TEXT
        """)

        # Create index on user_id
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id)
        """)

        # Create index for phash deduplication lookups
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_user_phash ON documents(user_id, phash)
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
                mined BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        # Add mined column if table already existed without it
        try:
            await conn.execute("""
                ALTER TABLE regenerate_sessions ADD COLUMN IF NOT EXISTS mined BOOLEAN DEFAULT FALSE
            """)
        except Exception:
            pass  # Column already exists or table just created with it

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

        # Mined prompt constraints from feedback loop
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS prompt_constraints (
                id SERIAL PRIMARY KEY,
                rule_text TEXT NOT NULL,
                rule_type TEXT NOT NULL,
                source_session_ids INTEGER[],
                active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        # Conversation memory tables
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_user
            ON conversations(user_id)
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id SERIAL PRIMARY KEY,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                documents JSONB,
                session_id INTEGER REFERENCES regenerate_sessions(id),
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conv_msgs
            ON conversation_messages(conversation_id)
        """)


async def create_document(conn, user_id: str, s3_key: str, phash: str | None = None) -> int:
    """Create a new document record."""
    query = """
        INSERT INTO documents (user_id, s3_key, phash)
        VALUES ($1, $2, $3)
        RETURNING id
    """
    return await conn.fetchval(query, user_id, s3_key, phash)


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

    # Remove deleted doc IDs from conversation_messages JSONB arrays
    id_strings = [str(did) for did in doc_ids]
    await conn.execute("""
        UPDATE conversation_messages
        SET documents = (
            SELECT COALESCE(jsonb_agg(elem), '[]'::jsonb)
            FROM jsonb_array_elements(documents) AS elem
            WHERE elem::text NOT IN (SELECT unnest($1::text[]))
        )
        WHERE documents IS NOT NULL
          AND documents != '[]'::jsonb
    """, id_strings)

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


async def count_unmined_sessions(conn) -> int:
    """Count sessions with 2+ attempts that haven't been mined yet."""
    return await conn.fetchval("""
        SELECT COUNT(*) FROM regenerate_sessions s
        WHERE s.mined = FALSE
          AND (SELECT COUNT(*) FROM regenerate_attempts WHERE session_id = s.id) >= 2
    """)


async def get_unmined_sessions(conn) -> List[Dict[str, Any]]:
    """Get sessions with 2+ attempts that haven't been mined yet."""
    rows = await conn.fetch("""
        SELECT s.id, s.query_text
        FROM regenerate_sessions s
        WHERE s.mined = FALSE
          AND (SELECT COUNT(*) FROM regenerate_attempts WHERE session_id = s.id) >= 2
        ORDER BY s.created_at
    """)
    return [{"id": row["id"], "query_text": row["query_text"]} for row in rows]


async def get_session_attempts(conn, session_id: int) -> List[Dict[str, Any]]:
    """Get all attempts for a session, ordered by attempt number."""
    rows = await conn.fetch("""
        SELECT attempt_number, answer, tool_trace
        FROM regenerate_attempts
        WHERE session_id = $1
        ORDER BY attempt_number
    """, session_id)
    return [
        {
            "attempt_number": row["attempt_number"],
            "answer": row["answer"],
            "tool_trace": json.loads(row["tool_trace"]) if row["tool_trace"] else [],
        }
        for row in rows
    ]


async def save_prompt_constraints(conn, constraints: List[Dict[str, Any]], session_ids: List[int]) -> int:
    """Save mined constraints and mark sessions as mined. Returns count saved."""
    count = 0
    for c in constraints:
        # Skip exact duplicates
        existing = await conn.fetchval(
            "SELECT id FROM prompt_constraints WHERE rule_text = $1 AND active = TRUE",
            c["rule"],
        )
        if existing:
            continue
        await conn.execute(
            """
            INSERT INTO prompt_constraints (rule_text, rule_type, source_session_ids)
            VALUES ($1, $2, $3)
            """,
            c["rule"], c["type"], session_ids,
        )
        count += 1

    # Mark sessions as mined
    if session_ids:
        await conn.execute(
            "UPDATE regenerate_sessions SET mined = TRUE WHERE id = ANY($1::int[])",
            session_ids,
        )
    return count


MAX_PROMPT_CONSTRAINTS = int(os.getenv("MAX_PROMPT_CONSTRAINTS", "10"))


async def get_active_constraints(conn) -> List[Dict[str, Any]]:
    """Get active prompt constraints, limited to most recent."""
    rows = await conn.fetch("""
        SELECT rule_text, rule_type
        FROM prompt_constraints
        WHERE active = TRUE
        ORDER BY created_at DESC
        LIMIT $1
    """, MAX_PROMPT_CONSTRAINTS)
    return [{"rule": row["rule_text"], "type": row["rule_type"]} for row in rows]


# --- Conversation Memory ---

CONVERSATION_HISTORY_LIMIT = int(os.getenv("CONVERSATION_HISTORY_LIMIT", "6"))


async def create_conversation(conn, user_id: str, title: str) -> int:
    """Create a new conversation and return its ID."""
    return await conn.fetchval(
        """
        INSERT INTO conversations (user_id, title)
        VALUES ($1, $2)
        RETURNING id
        """,
        user_id, title[:100],
    )


async def add_conversation_message(
    conn, conversation_id: int, role: str, content: str,
    documents: list = None, session_id: int = None,
) -> int:
    """Add a message to a conversation. Returns the message ID."""
    return await conn.fetchval(
        """
        INSERT INTO conversation_messages (conversation_id, role, content, documents, session_id)
        VALUES ($1, $2, $3, $4::jsonb, $5)
        RETURNING id
        """,
        conversation_id, role, content,
        json.dumps(documents) if documents else None,
        session_id,
    )


async def get_conversation_history(
    conn, conversation_id: int, user_id: str,
) -> list:
    """Get recent messages for a conversation, oldest first.

    Limit controlled by CONVERSATION_HISTORY_LIMIT env var (default 6).
    Validates ownership via user_id.
    """
    owner = await conn.fetchval(
        "SELECT user_id FROM conversations WHERE id = $1",
        conversation_id,
    )
    if owner != user_id:
        return []

    rows = await conn.fetch(
        """
        SELECT role, content FROM (
            SELECT role, content, created_at
            FROM conversation_messages
            WHERE conversation_id = $1
            ORDER BY created_at DESC
            LIMIT $2
        ) sub
        ORDER BY created_at ASC
        """,
        conversation_id, CONVERSATION_HISTORY_LIMIT,
    )
    return [{"role": row["role"], "content": row["content"]} for row in rows]


async def update_conversation_timestamp(conn, conversation_id: int):
    """Touch the updated_at timestamp."""
    await conn.execute(
        "UPDATE conversations SET updated_at = NOW() WHERE id = $1",
        conversation_id,
    )
