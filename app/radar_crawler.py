"""Background crawler to extract event dates for radar.

This module provides an async crawler that processes documents to identify
upcoming dates (due dates, renewals, deadlines, etc.) without slowing down
image upload processing.

Event data is stored directly on the DOCUMENTS table so ANY document type
can have an event date, not just receipts.

Flow:
1. Get documents with doc_text but not yet radar-processed
2. Keyword/date filter: skip docs without deadline words or date patterns
3. Text-only LLM call for docs with keywords or dates
4. Save event_date to documents table, mark as processed
"""
import logging
import re
from typing import List, Dict, Any
from datetime import datetime

from app.db import init_db
from app.vlm_client import extract_event_from_text


logger = logging.getLogger(__name__)

# Keywords that suggest a document might have an upcoming date
EVENT_KEYWORDS = [
    # Explicit deadline words
    "due", "expires", "expiration", "renew", "renewal", "deadline",
    "payment due", "bill", "invoice", "subscription", "appointment", "appt",
    "scheduled", "reminder", "by date", "valid until", "ends",
    "before", "overdue", "past due", "balance due",
    # Relative date words
    "tomorrow", "next week", "next month", "next year",
    "in a week", "in a month", "in a day", "in 2 days", "in 3 days",
    "this week", "this month", "this friday", "this monday",
    "this tuesday", "this wednesday", "this thursday", "this saturday", "this sunday",
    "next monday", "next tuesday", "next wednesday", "next thursday", "next friday",
    "upcoming", "soon", "later today", "tonight",
]


def has_event_keywords(text: str) -> bool:
    """Check if text contains keywords suggesting an upcoming event."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in EVENT_KEYWORDS)


# Date patterns to detect dates even without keywords
DATE_PATTERNS = [
    r'\d{1,2}/\d{1,2}/\d{2,4}',  # M/D/YY or MM/DD/YYYY
    r'\d{1,2}-\d{1,2}-\d{2,4}',  # M-D-YY or MM-DD-YYYY
    r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}',  # "March 2", "Jan 15th"
]


def has_date_pattern(text: str) -> bool:
    """Check if text contains date-like patterns."""
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in DATE_PATTERNS)


def parse_date(date_str: str) -> "datetime.date | None":
    """Parse date string to datetime.date, returns None if invalid."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


async def get_unprocessed_documents(db, user_id: str | None = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Get documents with text but not yet processed for radar.

    Args:
        db: Database connection
        user_id: Optional user ID to filter (None = all users)
        limit: Max documents to fetch
    """
    if user_id:
        query = """
            SELECT id, user_id, doc_text
            FROM documents
            WHERE user_id = $1
              AND doc_text IS NOT NULL
              AND doc_text != ''
              AND doc_text != '[Processing failed]'
              AND (radar_processed = FALSE OR radar_processed IS NULL)
            ORDER BY created_at DESC
            LIMIT $2
        """
        rows = await db.fetch(query, user_id, limit)
    else:
        query = """
            SELECT id, user_id, doc_text
            FROM documents
            WHERE doc_text IS NOT NULL
              AND doc_text != ''
              AND doc_text != '[Processing failed]'
              AND (radar_processed = FALSE OR radar_processed IS NULL)
            ORDER BY created_at DESC
            LIMIT $1
        """
        rows = await db.fetch(query, limit)

    return [dict(row) for row in rows]


async def update_document_event(
    db,
    doc_id: int,
    event_date: str,
    event_description: str | None,
    event_entity: str | None,
) -> None:
    """Update document with event date and mark as processed."""
    parsed_date = parse_date(event_date)
    if not parsed_date:
        # Invalid date, just mark as processed
        await db.execute("UPDATE documents SET radar_processed = TRUE WHERE id = $1", doc_id)
        return

    await db.execute("""
        UPDATE documents
        SET event_date = $1, event_description = $2, event_entity = $3, radar_processed = TRUE
        WHERE id = $4
    """, parsed_date, event_description, event_entity, doc_id)


async def mark_processed(db, doc_id: int) -> None:
    """Mark document as radar-processed (no event found)."""
    await db.execute("UPDATE documents SET radar_processed = TRUE WHERE id = $1", doc_id)


async def crawl_documents(user_id: str | None = None, limit: int = 50) -> Dict[str, int]:
    """Crawl unprocessed documents for event dates.

    Args:
        user_id: Optional user ID to filter (None = all users, useful for cron)
        limit: Max documents to process in one crawl

    Returns:
        Stats dict with processed, events_found, skipped, errors counts
    """
    pool = await init_db()

    stats = {"processed": 0, "events_found": 0, "skipped": 0, "errors": 0}

    async with pool.acquire() as db:
        docs = await get_unprocessed_documents(db, user_id, limit)
        logger.info(f"Radar crawler: found {len(docs)} unprocessed documents")

        for doc in docs:
            try:
                doc_text = doc["doc_text"]

                # Skip if no event keywords AND no date patterns (fast filtering)
                if not has_event_keywords(doc_text) and not has_date_pattern(doc_text):
                    await mark_processed(db, doc["id"])
                    stats["skipped"] += 1
                    continue

                # Call text-only LLM
                logger.info(f"Radar crawler: extracting events from doc {doc['id']}")
                result = await extract_event_from_text(doc_text)

                if result and result.get("event_date"):
                    await update_document_event(
                        db,
                        doc["id"],
                        result["event_date"],
                        result.get("event_description"),
                        result.get("entity"),
                    )
                    stats["events_found"] += 1
                    logger.info(f"Radar crawler: found event date {result['event_date']} for doc {doc['id']}")
                else:
                    await mark_processed(db, doc["id"])

                stats["processed"] += 1

            except Exception as e:
                logger.error(f"Radar crawler: error processing doc {doc['id']}: {e}")
                stats["errors"] += 1

    logger.info(f"Radar crawler complete: {stats}")
    return stats
