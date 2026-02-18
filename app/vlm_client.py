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
                           "Only fix obvious OCR errors, do not add or remove information."
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


async def extract_event_from_text(doc_text: str) -> Dict[str, Any] | None:
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

Today's date is {date.today().isoformat()}. Use this to resolve relative dates like "tomorrow", "next week", "in 3 days".

Respond in JSON:
{{
  "event_date": "YYYY-MM-DD or null",
  "event_description": "Brief description or null",
  "entity": "Business/person name or null"
}}

If multiple future dates exist, return the SOONEST one.
Only extract FUTURE dates. Ignore past dates (receipts, purchase dates)."""
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
