"""Field extraction and storage for structured document data."""
import json as _json
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

    metadata = fields.get("metadata")
    metadata_json = _json.dumps(metadata) if metadata else '{}'

    query = """
        INSERT INTO extractions (doc_id, doc_type, merchant, date, total_amount, address, metadata)
        VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
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
        metadata_json,
    )

    return result


async def get_extractions_by_doc(db, doc_id: int) -> Dict[str, Any] | None:
    """Get extractions for a document."""
    query = """
        SELECT id, doc_type, merchant, date, total_amount, address, metadata
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
        "metadata": _json.loads(row["metadata"]) if row["metadata"] else {},
    }



# Document types that represent income, not expenses.
# Excluded from spending aggregation queries.
_INCOME_DOC_TYPES = ('payslip',)


async def get_total_spending(db, user_id: str, start_date: str = None, end_date: str = None, doc_type: str = None) -> dict:
    """Get total spending in date range, optionally filtered by document type."""
    sql = """
        SELECT COALESCE(SUM(e.total_amount), 0) as total, COUNT(*) as count
        FROM extractions e
        JOIN documents d ON e.doc_id = d.id
        WHERE d.user_id = $1
          AND e.total_amount IS NOT NULL AND e.total_amount > 0
          AND ($2::date IS NULL OR COALESCE(e.date, d.created_at::date) >= $2::date)
          AND ($3::date IS NULL OR COALESCE(e.date, d.created_at::date) <= $3::date)
          AND ($4::text IS NULL OR e.doc_type = $4::text)
          AND e.doc_type != ALL($5::text[])
    """
    row = await db.fetchrow(sql, user_id, parse_date(start_date), parse_date(end_date), doc_type, list(_INCOME_DOC_TYPES))
    return {"total": float(row["total"]), "receipt_count": row["count"]}


async def get_spending_by_merchant(db, user_id: str, start_date: str = None, end_date: str = None, limit: int = 10, doc_type: str = None) -> list:
    """Get spending grouped by merchant, optionally filtered by document type."""
    sql = """
        SELECT e.merchant, COALESCE(SUM(e.total_amount), 0) as total, COUNT(*) as count,
               array_agg(DISTINCT d.id) as doc_ids
        FROM extractions e
        JOIN documents d ON e.doc_id = d.id
        WHERE d.user_id = $1 AND e.merchant IS NOT NULL
          AND e.total_amount IS NOT NULL AND e.total_amount > 0
          AND ($2::date IS NULL OR COALESCE(e.date, d.created_at::date) >= $2::date)
          AND ($3::date IS NULL OR COALESCE(e.date, d.created_at::date) <= $3::date)
          AND ($5::text IS NULL OR e.doc_type = $5::text)
          AND e.doc_type != ALL($6::text[])
        GROUP BY e.merchant
        ORDER BY total DESC
        LIMIT $4
    """
    rows = await db.fetch(sql, user_id, parse_date(start_date), parse_date(end_date), limit, doc_type, list(_INCOME_DOC_TYPES))
    return [{"merchant": r["merchant"], "total": float(r["total"]), "count": r["count"],
             "doc_ids": [str(did) for did in r["doc_ids"]]} for r in rows]


async def get_receipts_by_merchant(db, user_id: str, merchant: str, limit: int = 20, doc_type: str = None) -> list:
    """Get documents from a specific merchant, optionally filtered by document type."""
    sql = """
        SELECT e.id, e.merchant, e.date, e.total_amount, d.id as doc_id
        FROM extractions e
        JOIN documents d ON e.doc_id = d.id
        WHERE d.user_id = $1 AND e.merchant ILIKE $2
          AND ($4::text IS NULL OR e.doc_type = $4::text)
          AND e.doc_type != ALL($5::text[])
        ORDER BY e.date DESC NULLS LAST
        LIMIT $3
    """
    rows = await db.fetch(sql, user_id, f"%{merchant}%", limit, doc_type, list(_INCOME_DOC_TYPES))
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


async def get_receipts_by_date_range(db, user_id: str, start_date: str, end_date: str, limit: int = 50, doc_type: str = None) -> list:
    """Get documents in a date range, optionally filtered by document type."""
    sql = """
        SELECT e.id, e.merchant, e.date, e.total_amount, d.id as doc_id
        FROM extractions e
        JOIN documents d ON e.doc_id = d.id
        WHERE d.user_id = $1 AND COALESCE(e.date, d.created_at::date) BETWEEN $2::date AND $3::date
          AND ($5::text IS NULL OR e.doc_type = $5::text)
          AND e.doc_type != ALL($6::text[])
        ORDER BY COALESCE(e.date, d.created_at::date) DESC
        LIMIT $4
    """
    rows = await db.fetch(sql, user_id, parse_date(start_date), parse_date(end_date), limit, doc_type, list(_INCOME_DOC_TYPES))
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


async def get_all_receipt_texts(db, user_id: str, limit: int = 50, doc_type: str = None) -> list:
    """Get all extracted documents with their full OCR text for broad reasoning queries."""
    sql = """
        SELECT e.merchant, e.date, e.total_amount, d.id as doc_id, d.doc_text
        FROM extractions e
        JOIN documents d ON e.doc_id = d.id
        WHERE d.user_id = $1
          AND d.doc_text NOT IN ('[No text detected]', '[Processing failed]')
          AND ($3::text IS NULL OR e.doc_type = $3::text)
          AND e.doc_type != ALL($4::text[])
        ORDER BY e.date DESC NULLS LAST
        LIMIT $2
    """
    rows = await db.fetch(sql, user_id, limit, doc_type, list(_INCOME_DOC_TYPES))
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


async def get_document_overview(db, user_id: str, limit: int = 50) -> list:
    """Get overview of all documents with optional extraction metadata."""
    sql = """
        SELECT d.id as doc_id, d.doc_type, d.created_at,
               e.merchant, e.date, e.total_amount
        FROM documents d
        LEFT JOIN extractions e ON e.doc_id = d.id
        WHERE d.user_id = $1
        ORDER BY d.created_at DESC
        LIMIT $2
    """
    rows = await db.fetch(sql, user_id, limit)
    return [
        {
            "doc_id": r["doc_id"],
            "doc_type": r["doc_type"],
            "uploaded": r["created_at"].isoformat() if r["created_at"] else None,
            "merchant": r["merchant"],
            "date": r["date"].isoformat() if r["date"] else None,
            "total": float(r["total_amount"]) if r["total_amount"] else None,
        }
        for r in rows
    ]


async def get_all_document_texts(db, user_id: str, limit: int = 30) -> list:
    """Get all documents with OCR text, including non-receipt docs like flyers, notes, screenshots."""
    sql = """
        SELECT d.id as doc_id, d.doc_type, d.doc_text, d.created_at,
               e.merchant, e.date, e.total_amount
        FROM documents d
        LEFT JOIN extractions e ON e.doc_id = d.id
        WHERE d.user_id = $1
          AND d.doc_text NOT IN ('[No text detected]', '[Processing failed]')
        ORDER BY d.created_at DESC
        LIMIT $2
    """
    rows = await db.fetch(sql, user_id, limit)
    return [
        {
            "doc_id": r["doc_id"],
            "doc_type": r["doc_type"],
            "uploaded": r["created_at"].isoformat() if r["created_at"] else None,
            "merchant": r["merchant"],
            "date": r["date"].isoformat() if r["date"] else None,
            "total": float(r["total_amount"]) if r["total_amount"] else None,
            "text": r["doc_text"],
        }
        for r in rows
    ]


async def detect_recurring_costs(db, user_id: str) -> List[dict]:
    """Detect recurring charges by analyzing purchase intervals per merchant
    and including documents explicitly classified as subscriptions or rental agreements."""
    sql = """
        SELECT e.merchant, e.date, e.total_amount, e.doc_type
        FROM extractions e
        JOIN documents d ON e.doc_id = d.id
        WHERE d.user_id = $1 AND e.merchant IS NOT NULL
          AND e.doc_type NOT IN ('payslip')
        ORDER BY e.merchant, e.date
    """
    rows = await db.fetch(sql, user_id)

    # Group by merchant, tracking doc_type
    by_merchant: Dict[str, list] = defaultdict(list)
    merchant_has_subscription: Dict[str, bool] = defaultdict(bool)
    for r in rows:
        by_merchant[r["merchant"]].append({
            "date": r["date"],
            "amount": float(r["total_amount"]) if r["total_amount"] else 0,
        })
        if r["doc_type"] in ("subscription", "rental_agreement"):
            merchant_has_subscription[r["merchant"]] = True

    results = []
    for merchant, transactions in by_merchant.items():
        amounts = [t["amount"] for t in transactions if t["amount"]]
        avg_amount = sum(amounts) / len(amounts) if amounts else 0
        dates_with_values = [t["date"] for t in transactions if t["date"]]

        # Documents classified as subscriptions are inherently recurring
        if merchant_has_subscription[merchant]:
            last_date = max(dates_with_values) if dates_with_values else None
            results.append({
                "merchant": merchant,
                "is_recurring": True,
                "interval_days": 30,
                "monthly_estimate": round(avg_amount, 2),
                "annual_estimate": round(avg_amount * 12, 2),
                "next_renewal_date": (last_date + timedelta(days=30)).isoformat() if last_date else None,
                "last_date": last_date.isoformat() if last_date else None,
                "transaction_count": len(transactions),
            })
            continue

        # For non-subscription merchants, detect recurring patterns from intervals
        if len(transactions) < 3 or not dates_with_values or len(dates_with_values) < 2:
            continue

        dates = sorted(dates_with_values)
        intervals = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]

        if not intervals:
            continue

        avg_interval = sum(intervals) / len(intervals)

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
          AND e.doc_type != ALL($2::text[])
        ORDER BY e.date
    """
    rows = await db.fetch(sql, user_id, list(_INCOME_DOC_TYPES))

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


# --- Income / Payslip / Rental Agreement Functions ---

def _num(v):
    """Safely convert a value to float, returning 0 on failure."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0


async def get_earnings_summary(db, user_id: str, start_date: str = None, end_date: str = None, limit: int = 50) -> list:
    """Get earnings summary from payslips over time."""
    sql = """
        SELECT e.merchant, e.date, e.total_amount,
               e.metadata->'earnings' as earnings,
               d.id as doc_id
        FROM extractions e
        JOIN documents d ON e.doc_id = d.id
        WHERE d.user_id = $1 AND e.doc_type = 'payslip'
          AND ($2::date IS NULL OR COALESCE(e.date, d.created_at::date) >= $2::date)
          AND ($3::date IS NULL OR COALESCE(e.date, d.created_at::date) <= $3::date)
        ORDER BY e.date DESC NULLS LAST
        LIMIT $4
    """
    rows = await db.fetch(sql, user_id, parse_date(start_date), parse_date(end_date), limit)
    results = []
    for r in rows:
        earnings = r["earnings"] or {}
        if isinstance(earnings, str):
            try: earnings = _json.loads(earnings)
            except (ValueError, TypeError): earnings = {}
        if not isinstance(earnings, dict): earnings = {}
        gross = sum(_num(v) for k, v in earnings.items() if k != "other" and v) + \
                sum(_num(item.get("amount")) for item in earnings.get("other", []) if item.get("amount"))
        results.append({
            "employer": r["merchant"],
            "date": r["date"].isoformat() if r["date"] else None,
            "net_pay": float(r["total_amount"]) if r["total_amount"] else None,
            "gross_pay": round(gross, 2) if gross else None,
            "doc_id": r["doc_id"],
        })
    return results


async def get_deductions_breakdown(db, user_id: str, start_date: str = None, end_date: str = None) -> dict:
    """Aggregate payslip deductions across a date range."""
    sql = """
        SELECT e.metadata->'deductions' as deductions
        FROM extractions e
        JOIN documents d ON e.doc_id = d.id
        WHERE d.user_id = $1 AND e.doc_type = 'payslip'
          AND ($2::date IS NULL OR COALESCE(e.date, d.created_at::date) >= $2::date)
          AND ($3::date IS NULL OR COALESCE(e.date, d.created_at::date) <= $3::date)
    """
    rows = await db.fetch(sql, user_id, parse_date(start_date), parse_date(end_date))

    totals = defaultdict(float)
    other_totals = defaultdict(float)
    for r in rows:
        deductions = r["deductions"] or {}
        if isinstance(deductions, str):
            try: deductions = _json.loads(deductions)
            except (ValueError, TypeError): deductions = {}
        if not isinstance(deductions, dict): deductions = {}
        for key, val in deductions.items():
            if key == "other":
                for item in (val or []):
                    if item.get("amount"):
                        other_totals[item["label"]] += _num(item["amount"])
            elif val:
                totals[key] += _num(val)

    return {
        "canonical": {k: round(v, 2) for k, v in totals.items()},
        "other": {k: round(v, 2) for k, v in other_totals.items()},
        "total_deductions": round(sum(totals.values()) + sum(other_totals.values()), 2),
    }


async def get_income_vs_spending(db, user_id: str, start_date: str = None, end_date: str = None) -> dict:
    """Compare monthly income (from payslips) vs spending (from other docs)."""
    income_sql = """
        SELECT TO_CHAR(COALESCE(e.date, d.created_at::date), 'YYYY-MM') as month,
               SUM(e.total_amount) as total
        FROM extractions e JOIN documents d ON e.doc_id = d.id
        WHERE d.user_id = $1 AND e.doc_type = 'payslip' AND e.total_amount IS NOT NULL
          AND ($2::date IS NULL OR COALESCE(e.date, d.created_at::date) >= $2::date)
          AND ($3::date IS NULL OR COALESCE(e.date, d.created_at::date) <= $3::date)
        GROUP BY month ORDER BY month
    """
    spending_sql = """
        SELECT TO_CHAR(COALESCE(e.date, d.created_at::date), 'YYYY-MM') as month,
               SUM(e.total_amount) as total
        FROM extractions e JOIN documents d ON e.doc_id = d.id
        WHERE d.user_id = $1 AND e.doc_type != ALL($4::text[])
          AND e.total_amount IS NOT NULL
          AND ($2::date IS NULL OR COALESCE(e.date, d.created_at::date) >= $2::date)
          AND ($3::date IS NULL OR COALESCE(e.date, d.created_at::date) <= $3::date)
        GROUP BY month ORDER BY month
    """
    sd, ed = parse_date(start_date), parse_date(end_date)
    income_rows = await db.fetch(income_sql, user_id, sd, ed)
    spending_rows = await db.fetch(spending_sql, user_id, sd, ed, list(_INCOME_DOC_TYPES))

    income_by_month = {r["month"]: float(r["total"]) for r in income_rows}
    spending_by_month = {r["month"]: float(r["total"]) for r in spending_rows}

    all_months = sorted(set(income_by_month) | set(spending_by_month))
    return {
        "months": [
            {
                "month": m,
                "income": round(income_by_month.get(m, 0), 2),
                "spending": round(spending_by_month.get(m, 0), 2),
                "net": round(income_by_month.get(m, 0) - spending_by_month.get(m, 0), 2),
            }
            for m in all_months
        ],
        "total_income": round(sum(income_by_month.values()), 2),
        "total_spending": round(sum(spending_by_month.values()), 2),
        "total_net": round(sum(income_by_month.values()) - sum(spending_by_month.values()), 2),
    }


async def get_lease_details(db, user_id: str) -> list:
    """Get rental agreement details for the user."""
    sql = """
        SELECT e.merchant, e.date, e.total_amount, e.address,
               e.metadata->>'tenant' as tenant,
               (e.metadata->>'security_deposit')::numeric as security_deposit,
               e.metadata->>'lease_end' as lease_end,
               (e.metadata->>'term_months')::int as term_months,
               e.metadata->'utilities_included' as utilities_included,
               (e.metadata->>'pet_deposit')::numeric as pet_deposit,
               (e.metadata->>'late_fee')::numeric as late_fee,
               d.id as doc_id
        FROM extractions e
        JOIN documents d ON e.doc_id = d.id
        WHERE d.user_id = $1 AND e.doc_type = 'rental_agreement'
        ORDER BY e.date DESC NULLS LAST
    """
    rows = await db.fetch(sql, user_id)
    results = []
    for r in rows:
        results.append({
            "landlord": r["merchant"],
            "lease_start": r["date"].isoformat() if r["date"] else None,
            "monthly_rent": float(r["total_amount"]) if r["total_amount"] else None,
            "property_address": r["address"],
            "tenant": r["tenant"],
            "security_deposit": float(r["security_deposit"]) if r["security_deposit"] else None,
            "lease_end": r["lease_end"],
            "term_months": r["term_months"],
            "utilities_included": r["utilities_included"],
            "pet_deposit": float(r["pet_deposit"]) if r["pet_deposit"] else None,
            "late_fee": float(r["late_fee"]) if r["late_fee"] else None,
            "doc_id": r["doc_id"],
        })
    return results


async def get_recurring_income(db, user_id: str) -> list:
    """Detect recurring income from payslip patterns."""
    sql = """
        SELECT e.merchant, e.date, e.total_amount
        FROM extractions e
        JOIN documents d ON e.doc_id = d.id
        WHERE d.user_id = $1 AND e.doc_type = 'payslip' AND e.merchant IS NOT NULL
        ORDER BY e.merchant, e.date
    """
    rows = await db.fetch(sql, user_id)

    by_employer: Dict[str, list] = defaultdict(list)
    for r in rows:
        by_employer[r["merchant"]].append({
            "date": r["date"],
            "amount": float(r["total_amount"]) if r["total_amount"] else 0,
        })

    results = []
    for employer, paychecks in by_employer.items():
        amounts = [p["amount"] for p in paychecks if p["amount"]]
        avg_amount = sum(amounts) / len(amounts) if amounts else 0
        dates = sorted([p["date"] for p in paychecks if p["date"]])

        if len(dates) >= 2:
            intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            avg_interval = sum(intervals) / len(intervals)
        else:
            avg_interval = 30  # Default assumption for single payslip

        # Determine pay frequency
        if 12 <= avg_interval <= 16:
            frequency = "biweekly"
            monthly_estimate = avg_amount * 26 / 12
        elif 28 <= avg_interval <= 31:
            frequency = "monthly"
            monthly_estimate = avg_amount
        elif 6 <= avg_interval <= 8:
            frequency = "weekly"
            monthly_estimate = avg_amount * 52 / 12
        else:
            frequency = "other"
            monthly_estimate = avg_amount * (365 / avg_interval) / 12 if avg_interval > 0 else avg_amount

        last_date = dates[-1] if dates else None
        results.append({
            "employer": employer,
            "frequency": frequency,
            "avg_net_pay": round(avg_amount, 2),
            "monthly_estimate": round(monthly_estimate, 2),
            "annual_estimate": round(monthly_estimate * 12, 2),
            "last_pay_date": last_date.isoformat() if last_date else None,
            "paycheck_count": len(paychecks),
        })

    return results
