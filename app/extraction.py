"""Field extraction and storage for structured document data."""
from typing import Dict, Any
from datetime import datetime


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
