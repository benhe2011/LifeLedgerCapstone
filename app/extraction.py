"""Field extraction and storage for structured document data."""
from typing import Dict, Any, List
from datetime import datetime, date, timedelta
from collections import defaultdict


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
          AND ($2::date IS NULL OR COALESCE(e.date, d.created_at::date) >= $2::date)
          AND ($3::date IS NULL OR COALESCE(e.date, d.created_at::date) <= $3::date)
    """
    row = await db.fetchrow(sql, user_id, parse_date(start_date), parse_date(end_date))
    return {"total": float(row["total"]), "receipt_count": row["count"]}


async def get_spending_by_merchant(db, user_id: str, start_date: str = None, end_date: str = None, limit: int = 10) -> list:
    """Get spending grouped by merchant."""
    sql = """
        SELECT e.merchant, COALESCE(SUM(e.total_amount), 0) as total, COUNT(*) as count
        FROM extractions e
        JOIN documents d ON e.doc_id = d.id
        WHERE d.user_id = $1 AND e.merchant IS NOT NULL
          AND ($2::date IS NULL OR COALESCE(e.date, d.created_at::date) >= $2::date)
          AND ($3::date IS NULL OR COALESCE(e.date, d.created_at::date) <= $3::date)
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


async def get_all_receipt_texts(db, user_id: str, limit: int = 50) -> list:
    """Get all receipt documents with their full OCR text for broad reasoning queries."""
    sql = """
        SELECT e.merchant, e.date, e.total_amount, d.id as doc_id, d.doc_text
        FROM extractions e
        JOIN documents d ON e.doc_id = d.id
        WHERE d.user_id = $1
          AND d.doc_text NOT IN ('[No text detected]', '[Processing failed]')
        ORDER BY e.date DESC NULLS LAST
        LIMIT $2
    """
    rows = await db.fetch(sql, user_id, limit)
    return [
        {
            "merchant": r["merchant"],
            "date": r["date"].isoformat() if r["date"] else None,
            "total": float(r["total_amount"]) if r["total_amount"] else None,
            "doc_id": r["doc_id"],
            "items_text": r["doc_text"],
        }
        for r in rows
    ]


async def detect_recurring_costs(db, user_id: str) -> List[dict]:
    """Detect recurring charges by analyzing purchase intervals per merchant."""
    sql = """
        SELECT e.merchant, e.date, e.total_amount
        FROM extractions e
        JOIN documents d ON e.doc_id = d.id
        WHERE d.user_id = $1 AND e.merchant IS NOT NULL AND e.date IS NOT NULL
        ORDER BY e.merchant, e.date
    """
    rows = await db.fetch(sql, user_id)

    # Group by merchant
    by_merchant: Dict[str, list] = defaultdict(list)
    for r in rows:
        by_merchant[r["merchant"]].append({
            "date": r["date"],
            "amount": float(r["total_amount"]) if r["total_amount"] else 0,
        })

    results = []
    for merchant, transactions in by_merchant.items():
        if len(transactions) < 3:
            continue

        dates = [t["date"] for t in transactions]
        amounts = [t["amount"] for t in transactions if t["amount"]]
        intervals = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]

        if not intervals:
            continue

        avg_interval = sum(intervals) / len(intervals)
        avg_amount = sum(amounts) / len(amounts) if amounts else 0

        is_monthly = 28 <= avg_interval <= 31
        is_annual = 350 <= avg_interval <= 380

        if not (is_monthly or is_annual):
            continue

        interval_days = round(avg_interval)
        last_date = dates[-1]
        next_renewal = last_date + timedelta(days=interval_days)

        if is_monthly:
            monthly_est = avg_amount
            annual_est = avg_amount * 12
        else:
            monthly_est = avg_amount / 12
            annual_est = avg_amount

        results.append({
            "merchant": merchant,
            "is_recurring": True,
            "interval_days": interval_days,
            "monthly_estimate": round(monthly_est, 2),
            "annual_estimate": round(annual_est, 2),
            "next_renewal_date": next_renewal.isoformat(),
            "last_date": last_date.isoformat(),
            "transaction_count": len(transactions),
        })

    return results


_TRAVEL_KEYWORDS = {
    "hotel", "flight", "airport", "booking", "reservation", "airline",
    "departure", "arrival", "boarding", "lodging", "accommodation",
    "travel", "itinerary", "check-in", "checkin", "check-out", "checkout",
    "airbnb", "hostel", "resort", "terminal", "gate",
}


def _is_travel_doc(doc_text: str, merchant: str) -> bool:
    """Check if a document is travel-related based on text content."""
    text_lower = (doc_text or "").lower() + " " + (merchant or "").lower()
    return any(kw in text_lower for kw in _TRAVEL_KEYWORDS)


async def detect_trips(db, user_id: str, proximity_days: int = 3) -> List[dict]:
    """Cluster travel-related documents into trips by date proximity."""
    sql = """
        SELECT e.doc_id, e.merchant, e.date, e.total_amount, e.address,
               d.doc_type, d.doc_text
        FROM extractions e
        JOIN documents d ON e.doc_id = d.id
        WHERE d.user_id = $1 AND e.date IS NOT NULL
        ORDER BY e.date
    """
    rows = await db.fetch(sql, user_id)

    # Filter to travel-related documents
    travel_docs = []
    for r in rows:
        if _is_travel_doc(r["doc_text"], r["merchant"]):
            travel_docs.append({
                "doc_id": r["doc_id"],
                "merchant": r["merchant"],
                "date": r["date"],
                "amount": float(r["total_amount"]) if r["total_amount"] else 0,
                "address": r["address"],
            })

    if not travel_docs:
        return []

    # Cluster by date proximity
    trips = []
    current_trip = [travel_docs[0]]

    for doc in travel_docs[1:]:
        prev_date = current_trip[-1]["date"]
        if (doc["date"] - prev_date).days <= proximity_days:
            current_trip.append(doc)
        else:
            trips.append(current_trip)
            current_trip = [doc]
    trips.append(current_trip)

    # Build trip summaries
    results = []
    for trip_docs in trips:
        dates = [d["date"] for d in trip_docs]
        addresses = [d["address"] for d in trip_docs if d["address"]]
        total_cost = sum(d["amount"] for d in trip_docs)

        # Location hint: most common address or first available
        location_hint = None
        if addresses:
            # Use the longest address as hint (usually most descriptive)
            location_hint = max(addresses, key=len)

        results.append({
            "start_date": min(dates).isoformat(),
            "end_date": max(dates).isoformat(),
            "total_cost": round(total_cost, 2),
            "document_count": len(trip_docs),
            "location_hint": location_hint,
            "documents": [
                {
                    "doc_id": d["doc_id"],
                    "merchant": d["merchant"],
                    "date": d["date"].isoformat(),
                    "amount": d["amount"],
                }
                for d in trip_docs
            ],
        })

    return results
