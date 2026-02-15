"""Field extraction and storage for structured document data."""
from typing import Dict, Any
from datetime import datetime, date


def parse_date(date_str: str) -> date | None:
    """Parse date string to datetime.date, returns None if invalid."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


async def save_extraction(db, doc_id: int, doc_type: str, fields: Dict[str, Any]) -> int:
    """Save extracted fields to the extractions table."""
    # Parse date if present
    date_value = None
    if fields.get("date"):
        try:
            date_value = datetime.strptime(fields["date"], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            pass

    # Parse total amount
    total_amount = None
    if fields.get("total_amount"):
        try:
            total_amount = float(fields["total_amount"])
        except (ValueError, TypeError):
            pass

    query = """
        INSERT INTO extractions (doc_id, doc_type, merchant, date, total_amount, address)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id
    """

    result = await db.fetchval(
        query,
        doc_id,
        doc_type,
        fields.get("merchant"),
        date_value,
        total_amount,
        fields.get("address"),
    )

    return result


async def get_extractions_by_doc(db, doc_id: int) -> Dict[str, Any] | None:
    """Get extractions for a document."""
    query = """
        SELECT id, doc_type, merchant, date, total_amount, address
        FROM extractions
        WHERE doc_id = $1
    """
    row = await db.fetchrow(query, doc_id)

    if not row:
        return None

    return {
        "id": row["id"],
        "doc_type": row["doc_type"],
        "merchant": row["merchant"],
        "date": row["date"].isoformat() if row["date"] else None,
        "total_amount": float(row["total_amount"]) if row["total_amount"] else None,
        "address": row["address"],
    }


async def get_user_receipts(db, user_id: str, limit: int = 100) -> list:
    """Get all receipts for a user with extracted fields."""
    query = """
        SELECT e.id, e.merchant, e.date, e.total_amount, e.address, d.s3_key
        FROM extractions e
        JOIN documents d ON e.doc_id = d.id
        WHERE d.user_id = $1 AND e.doc_type = 'receipt'
        ORDER BY e.date DESC NULLS LAST
        LIMIT $2
    """
    rows = await db.fetch(query, user_id, limit)

    return [
        {
            "id": row["id"],
            "merchant": row["merchant"],
            "date": row["date"].isoformat() if row["date"] else None,
            "total_amount": float(row["total_amount"]) if row["total_amount"] else None,
            "address": row["address"],
            "s3_key": row["s3_key"],
        }
        for row in rows
    ]


async def get_total_spending(db, user_id: str, start_date: str = None, end_date: str = None) -> dict:
    """Get total spending in date range."""
    sql = """
        SELECT COALESCE(SUM(e.total_amount), 0) as total, COUNT(*) as count
        FROM extractions e
        JOIN documents d ON e.doc_id = d.id
        WHERE d.user_id = $1
          AND ($2::date IS NULL OR e.date >= $2::date)
          AND ($3::date IS NULL OR e.date <= $3::date)
    """
    row = await db.fetchrow(sql, user_id, parse_date(start_date), parse_date(end_date))
    return {"total": float(row["total"]), "receipt_count": row["count"]}


async def get_spending_by_merchant(db, user_id: str, start_date: str = None, end_date: str = None, limit: int = 10) -> list:
    """Get spending grouped by merchant."""
    sql = """
        SELECT e.merchant, SUM(e.total_amount) as total, COUNT(*) as count
        FROM extractions e
        JOIN documents d ON e.doc_id = d.id
        WHERE d.user_id = $1 AND e.merchant IS NOT NULL
          AND ($2::date IS NULL OR e.date >= $2::date)
          AND ($3::date IS NULL OR e.date <= $3::date)
        GROUP BY e.merchant
        ORDER BY total DESC
        LIMIT $4
    """
    rows = await db.fetch(sql, user_id, parse_date(start_date), parse_date(end_date), limit)
    return [{"merchant": r["merchant"], "total": float(r["total"]), "count": r["count"]} for r in rows]


async def get_receipts_by_merchant(db, user_id: str, merchant: str, limit: int = 20) -> list:
    """Get receipts from a specific merchant."""
    sql = """
        SELECT e.id, e.merchant, e.date, e.total_amount, d.id as doc_id
        FROM extractions e
        JOIN documents d ON e.doc_id = d.id
        WHERE d.user_id = $1 AND e.merchant ILIKE $2
        ORDER BY e.date DESC
        LIMIT $3
    """
    rows = await db.fetch(sql, user_id, f"%{merchant}%", limit)
    return [
        {
            "id": r["id"],
            "merchant": r["merchant"],
            "date": r["date"].isoformat() if r["date"] else None,
            "total": float(r["total_amount"]) if r["total_amount"] else None,
            "doc_id": r["doc_id"]
        }
        for r in rows
    ]


async def get_receipts_by_date_range(db, user_id: str, start_date: str, end_date: str, limit: int = 50) -> list:
    """Get receipts in a date range."""
    sql = """
        SELECT e.id, e.merchant, e.date, e.total_amount, d.id as doc_id
        FROM extractions e
        JOIN documents d ON e.doc_id = d.id
        WHERE d.user_id = $1 AND e.date BETWEEN $2::date AND $3::date
        ORDER BY e.date DESC
        LIMIT $4
    """
    rows = await db.fetch(sql, user_id, parse_date(start_date), parse_date(end_date), limit)
    return [
        {
            "id": r["id"],
            "merchant": r["merchant"],
            "date": r["date"].isoformat() if r["date"] else None,
            "total": float(r["total_amount"]) if r["total_amount"] else None,
            "doc_id": r["doc_id"]
        }
        for r in rows
    ]
