"""Azure GPT-4.1 VLM client for OCR refinement and field extraction."""
import os
import base64
from datetime import date
from typing import Dict, Any

from openai import AsyncAzureOpenAI


def get_vlm_client() -> AsyncAzureOpenAI:
    """Get Azure OpenAI client."""
    return AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2024-02-15-preview",
    )


def encode_image(image_bytes: bytes) -> str:
    """Encode image bytes to base64."""
    return base64.b64encode(image_bytes).decode("utf-8")


async def refine_ocr_with_vlm(image_bytes: bytes, ocr_text: str) -> str:
    """Use VLM to refine low-confidence OCR output."""
    client = get_vlm_client()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

    base64_image = encode_image(image_bytes)

    response = await client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "system",
                "content": "You are an OCR refinement assistant. Given an image and noisy OCR output, "
                           "produce clean, accurate text. Preserve the original structure and content. "
                           "Only fix obvious OCR errors, do not add or remove information. "
                           "If the image contains no meaningful readable text (the OCR output is just "
                           "visual noise misinterpreted as characters), respond with exactly: [No text detected]"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Here is the noisy OCR output:\n\n{ocr_text}\n\nPlease provide the corrected text:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        max_tokens=2000,
        temperature=0.1,
    )

    return response.choices[0].message.content


async def extract_receipt_fields(image_bytes: bytes) -> Dict[str, Any] | None:
    """Extract structured fields from a receipt image."""
    client = get_vlm_client()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

    base64_image = encode_image(image_bytes)

    response = await client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "system",
                "content": """You are a receipt parsing assistant. Extract the following fields from the receipt image:
- merchant: Store/business name
- date: Transaction date (YYYY-MM-DD format)
- total_amount: Total amount paid (numeric, no currency symbol)
- address: Store address if visible

Respond in JSON format only. If a field is not found, use null.
Example: {"merchant": "Target", "date": "2024-01-15", "total_amount": 45.99, "address": "123 Main St"}"""
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract the receipt fields from this image:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        max_tokens=500,
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    import json
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return None


EXTRACTION_PROMPTS = {
    "receipt": """You are a receipt parsing assistant. Extract the following fields from the receipt image:
- merchant: Store/business name
- date: Transaction date (YYYY-MM-DD format)
- total_amount: Total amount paid (numeric, no currency symbol)
- address: Store address if visible

Respond in JSON format only. If a field is not found, use null.
Example: {"merchant": "Target", "date": "2024-01-15", "total_amount": 45.99, "address": "123 Main St"}""",

    "subscription": """You are a document parsing assistant. Extract the following fields from this subscription document:
- merchant: Service or company name
- date: Billing or document date (YYYY-MM-DD format)
- total_amount: Amount charged (numeric, no currency symbol)
- address: Company address if visible

Respond in JSON format only. If a field is not found, use null.
Example: {"merchant": "Netflix", "date": "2024-03-01", "total_amount": 15.99, "address": null}""",

    "invoice": """You are a document parsing assistant. Extract the following fields from this invoice:
- merchant: Vendor or company name
- date: Invoice date (YYYY-MM-DD format)
- total_amount: Total amount due (numeric, no currency symbol)
- address: Vendor address if visible

Respond in JSON format only. If a field is not found, use null.
Example: {"merchant": "Acme Corp", "date": "2024-02-20", "total_amount": 1250.00, "address": "456 Commerce Blvd"}""",

    "payslip": """You are a payslip parsing assistant. Extract the following fields from this payslip image:
- merchant: Employer/company name
- date: Pay date (YYYY-MM-DD format)
- total_amount: Net pay / take-home pay (numeric, no currency symbol)
- address: Employer address if visible
- metadata: Object with earnings breakdown, deductions breakdown, and pay period details (see below)

For metadata.earnings, map each earnings line to the closest key: regular, overtime, bonus, commission, tips, pto_payout, reimbursement. Use null for absent items. If a line doesn't match, add to "other" as {"label": "...", "amount": numeric}.

For metadata.deductions, map each deduction line to the closest key: federal_tax, state_tax, local_tax, social_security, medicare, retirement_401k, roth_401k, health_insurance, dental_insurance, vision_insurance, hsa, fsa, life_insurance, disability_insurance, union_dues, garnishments. Use null for absent items. If a line doesn't match, add to "other" as {"label": "...", "amount": numeric}.

Also extract in metadata: pay_period_start (YYYY-MM-DD or null), pay_period_end (YYYY-MM-DD or null), ytd_gross (numeric or null), ytd_net (numeric or null).

Respond in JSON format only. If a field is not found, use null.
Example: {"merchant": "Acme Corp", "date": "2026-03-15", "total_amount": 3890.50, "address": "123 Business Blvd", "metadata": {"earnings": {"regular": 4000.00, "overtime": 600.00, "bonus": null, "commission": null, "tips": null, "pto_payout": null, "reimbursement": null, "other": []}, "deductions": {"federal_tax": 780.00, "state_tax": 312.00, "local_tax": null, "social_security": 322.00, "medicare": 75.30, "retirement_401k": 500.00, "roth_401k": null, "health_insurance": 150.00, "dental_insurance": 25.00, "vision_insurance": 10.00, "hsa": null, "fsa": null, "life_insurance": null, "disability_insurance": null, "union_dues": null, "garnishments": null, "other": []}, "pay_period_start": "2026-03-01", "pay_period_end": "2026-03-15", "ytd_gross": 23000.00, "ytd_net": 17200.00}}""",

    "rental_agreement": """You are a rental agreement parsing assistant. Extract the following fields from this lease/rental agreement image:
- merchant: Landlord or property management company name
- date: Lease start date (YYYY-MM-DD format)
- total_amount: Monthly rent amount (numeric, no currency symbol)
- address: Property/rental address
- metadata: Object with tenant, security_deposit (numeric or null), lease_end (YYYY-MM-DD or null), term_months (integer or null), utilities_included (list of strings or null), pet_deposit (numeric or null), late_fee (numeric or null)

Respond in JSON format only. If a field is not found, use null.
Example: {"merchant": "Skyline Property Management", "date": "2026-01-01", "total_amount": 2400.00, "address": "456 Oak Ave Apt 3B", "metadata": {"tenant": "Jane Doe", "security_deposit": 2400.00, "lease_end": "2027-01-01", "term_months": 12, "utilities_included": ["water", "trash"], "pet_deposit": null, "late_fee": 50.00}}""",
}

EXTRACTION_TEXT_PROMPTS = {
    "receipt": """Extract receipt information from this text.

Respond in JSON:
{
  "merchant": "Store/business name or null",
  "total_amount": numeric amount or null,
  "date": "YYYY-MM-DD or null"
}

Only extract what's clearly present. Use null for missing fields.""",

    "subscription": """Extract subscription information from this text.

Respond in JSON:
{
  "merchant": "Service/company name or null",
  "total_amount": numeric amount or null,
  "date": "YYYY-MM-DD or null"
}

Only extract what's clearly present. Use null for missing fields.""",

    "invoice": """Extract invoice information from this text.

Respond in JSON:
{
  "merchant": "Vendor/company name or null",
  "total_amount": numeric amount due or null,
  "date": "YYYY-MM-DD or null"
}

Only extract what's clearly present. Use null for missing fields.""",

    "payslip": """Extract payslip information from this text.

Respond in JSON:
{
  "merchant": "Employer name or null",
  "total_amount": net pay as numeric or null,
  "date": "YYYY-MM-DD or null",
  "metadata": {
    "earnings": {"regular": amount, "overtime": amount, "bonus": amount, "commission": amount, "tips": amount, "pto_payout": amount, "reimbursement": amount, "other": [{"label": "...", "amount": amount}]},
    "deductions": {"federal_tax": amount, "state_tax": amount, "local_tax": amount, "social_security": amount, "medicare": amount, "retirement_401k": amount, "roth_401k": amount, "health_insurance": amount, "dental_insurance": amount, "vision_insurance": amount, "hsa": amount, "fsa": amount, "life_insurance": amount, "disability_insurance": amount, "union_dues": amount, "garnishments": amount, "other": [{"label": "...", "amount": amount}]},
    "pay_period_start": "YYYY-MM-DD or null",
    "pay_period_end": "YYYY-MM-DD or null",
    "ytd_gross": numeric or null,
    "ytd_net": numeric or null
  }
}

Map each earnings/deduction line to the closest canonical key. If no key fits, add to "other". Use null for absent fields.""",

    "rental_agreement": """Extract rental agreement information from this text.

Respond in JSON:
{
  "merchant": "Landlord or property management name or null",
  "total_amount": monthly rent as numeric or null,
  "date": "YYYY-MM-DD lease start or null",
  "metadata": {"tenant": "name or null", "security_deposit": numeric or null, "lease_end": "YYYY-MM-DD or null", "term_months": integer or null}
}

Only extract what's clearly present. Use null for missing fields.""",
}


async def extract_document_fields(image_bytes: bytes, doc_type: str) -> Dict[str, Any] | None:
    """Extract structured fields from a document image based on its type."""
    prompt = EXTRACTION_PROMPTS.get(doc_type)
    if not prompt:
        return None

    client = get_vlm_client()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

    base64_image = encode_image(image_bytes)

    token_limits = {"payslip": 800}
    max_tokens = token_limits.get(doc_type, 500)

    response = await client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Extract the fields from this {doc_type} image:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        max_tokens=max_tokens,
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    import json
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return None


async def extract_document_from_text(text: str, doc_type: str) -> Dict[str, Any] | None:
    """Extract fields from user-provided text based on document type.

    Used when user manually inputs text for a document that OCR couldn't read.
    """
    prompt = EXTRACTION_TEXT_PROMPTS.get(doc_type)
    if not prompt:
        return None

    client = get_vlm_client()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

    response = await client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ],
        max_tokens=200,
        temperature=0,
        response_format={"type": "json_object"},
    )

    import json
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return None


async def extract_event_from_text(doc_text: str, doc_date=None) -> Dict[str, Any] | None:
    """Extract event date from document text using text-only LLM (cheap/fast).

    Used by the radar crawler to identify upcoming dates without vision calls.
    """
    client = get_vlm_client()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

    response = await client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "system",
                "content": f"""Extract upcoming/future dates from this document text.
Look for: due dates, renewal dates, expiration dates, deadlines, appointment dates.

This document is dated {doc_date.isoformat() if doc_date else 'unknown'}. Today's date is {date.today().isoformat()}.
For relative deadlines like "within 14 days" or "in 30 days", calculate from the DOCUMENT date, not today.
For absolute dates, use them directly.

Respond in JSON:
{{
  "event_date": "YYYY-MM-DD or null",
  "event_description": "Brief description or null",
  "entity": "Business/person name or null"
}}

If multiple future dates exist, return the SOONEST one.
Only extract FUTURE dates (relative to today). Ignore dates that have already passed."""
            },
            {"role": "user", "content": doc_text}
        ],
        max_tokens=200,
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    import json
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return None


async def extract_receipt_from_text(text: str) -> Dict[str, Any] | None:
    """Extract receipt fields from user-provided text (no image needed).

    Used when user manually inputs text for a document that OCR couldn't read.
    """
    client = get_vlm_client()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

    response = await client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "system",
                "content": """Extract receipt information from this text.

Respond in JSON:
{
  "merchant": "Store/business name or null",
  "total_amount": numeric amount or null,
  "date": "YYYY-MM-DD or null"
}

Only extract what's clearly present. Use null for missing fields."""
            },
            {"role": "user", "content": text}
        ],
        max_tokens=200,
        temperature=0,
        response_format={"type": "json_object"},
    )

    import json
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return None


async def ask_vlm(image_bytes: bytes | None, question: str, context: str = "") -> str:
    """Ask a question to the VLM with optional image."""
    client = get_vlm_client()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant analyzing documents and answering questions. "
                       "Base your answers only on the provided context and images."
        }
    ]

    user_content = []
    if context:
        user_content.append({"type": "text", "text": f"Context:\n{context}\n\nQuestion: {question}"})
    else:
        user_content.append({"type": "text", "text": question})

    if image_bytes:
        base64_image = encode_image(image_bytes)
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "high"
            }
        })

    messages.append({"role": "user", "content": user_content})

    response = await client.chat.completions.create(
        model=deployment,
        messages=messages,
        max_tokens=1000,
        temperature=0.3,
    )

    return response.choices[0].message.content
