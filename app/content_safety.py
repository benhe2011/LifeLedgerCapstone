"""Azure AI Content Safety integration.

Provides text moderation, image moderation, prompt shields,
groundedness detection, and task adherence checks via the
Azure Content Safety REST API.

Design: All checks are non-blocking on failure — if the safety
service is unreachable or returns an error, the request is allowed
through with a warning log.  This matches the existing fail-open
pattern used in vlm_client.py.
"""

import os
import base64
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import httpx

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────
CONTENT_SAFETY_ENDPOINT = os.getenv("CONTENT_SAFETY_ENDPOINT", "")
CONTENT_SAFETY_KEY = os.getenv("CONTENT_SAFETY_KEY", "")

# Severity thresholds (0, 2, 4, 6 scale)
TEXT_SEVERITY_THRESHOLD = int(os.getenv("TEXT_SEVERITY_THRESHOLD", "4"))   # MEDIUM
IMAGE_SEVERITY_THRESHOLD = int(os.getenv("IMAGE_SEVERITY_THRESHOLD", "2"))  # LOW

# API versions
_TEXT_IMAGE_API_VERSION = "2024-09-01"
_GROUNDEDNESS_API_VERSION = "2024-09-15-preview"
_TASK_ADHERENCE_API_VERSION = "2025-09-15-preview"

# Timeout for safety API calls (seconds)
_TIMEOUT = float(os.getenv("CONTENT_SAFETY_TIMEOUT", "10"))


# ── Enums & data classes ─────────────────────────────────────

class Category(str, Enum):
    """Content safety categories."""
    HATE = "Hate"
    SELF_HARM = "SelfHarm"
    SEXUAL = "Sexual"
    VIOLENCE = "Violence"


class Action(str, Enum):
    """Decision actions."""
    ACCEPT = "Accept"
    REJECT = "Reject"


@dataclass
class SafetyResult:
    """Result of a content safety check."""
    action: Action
    categories: Dict[str, int] = field(default_factory=dict)
    reason: Optional[str] = None

    @property
    def is_rejected(self) -> bool:
        return self.action == Action.REJECT


# ── Shared HTTP helper ────────────────────────────────────────

def _get_headers() -> Dict[str, str]:
    """Build request headers for Content Safety API."""
    return {
        "Ocp-Apim-Subscription-Key": CONTENT_SAFETY_KEY,
        "Content-Type": "application/json",
    }


async def _call_safety_api(
    path: str,
    body: Dict[str, Any],
    api_version: str,
) -> Optional[Dict[str, Any]]:
    """Make an async POST to the Content Safety API.

    Returns parsed JSON on success, None on any failure.
    """
    if not CONTENT_SAFETY_ENDPOINT or not CONTENT_SAFETY_KEY:
        logger.warning("Content Safety not configured (missing endpoint or key), skipping check")
        return None

    url = (
        f"{CONTENT_SAFETY_ENDPOINT.rstrip('/')}"
        f"/contentsafety/{path}?api-version={api_version}"
    )

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.post(url, json=body, headers=_get_headers())
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException:
        logger.warning("Content Safety API timeout for %s", path)
        return None
    except httpx.HTTPStatusError as e:
        logger.warning(
            "Content Safety API error for %s: %s %s",
            path, e.response.status_code, e.response.text[:200],
        )
        return None
    except Exception as e:
        logger.warning("Content Safety API unexpected error for %s: %s", path, e)
        return None


# ── Decision helper ───────────────────────────────────────────

def _make_decision(
    categories_analysis: List[Dict[str, Any]],
    threshold: int,
) -> SafetyResult:
    """Evaluate category severities against threshold.

    Returns Accept if all categories are below threshold,
    Reject if any category meets or exceeds threshold.
    """
    categories: Dict[str, int] = {}
    rejected: List[str] = []

    for item in categories_analysis:
        cat = item.get("category", "Unknown")
        severity = item.get("severity", 0)
        categories[cat] = severity
        if severity >= threshold:
            rejected.append(f"{cat}(severity={severity})")

    if rejected:
        reason = f"Content flagged: {', '.join(rejected)} (threshold={threshold})"
        logger.info(reason)
        return SafetyResult(action=Action.REJECT, categories=categories, reason=reason)

    return SafetyResult(action=Action.ACCEPT, categories=categories)


# ── Feature 1: Text Moderation ────────────────────────────────

async def check_text(text: str) -> Optional[SafetyResult]:
    """Analyze text for harmful content.

    Uses TEXT_SEVERITY_THRESHOLD (default 4 = MEDIUM).
    Returns None on API failure (fail-open).
    """
    if not text or not text.strip():
        return SafetyResult(action=Action.ACCEPT)

    # Azure limit: 10K characters per request
    truncated = text[:10000]

    body = {"text": truncated, "blocklistNames": []}
    result = await _call_safety_api("text:analyze", body, _TEXT_IMAGE_API_VERSION)

    if result is None:
        return None

    return _make_decision(result.get("categoriesAnalysis", []), TEXT_SEVERITY_THRESHOLD)


# ── Feature 2: Image Moderation ───────────────────────────────

async def check_image(image_bytes: bytes) -> Optional[SafetyResult]:
    """Analyze image for harmful content.

    Uses IMAGE_SEVERITY_THRESHOLD (default 2 = LOW).
    Returns None on API failure (fail-open).
    """
    if not image_bytes:
        return SafetyResult(action=Action.ACCEPT)

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    body = {"image": {"content": b64}}
    result = await _call_safety_api("image:analyze", body, _TEXT_IMAGE_API_VERSION)

    if result is None:
        return None

    return _make_decision(result.get("categoriesAnalysis", []), IMAGE_SEVERITY_THRESHOLD)


# ── Feature 3: Prompt Shields ─────────────────────────────────

async def check_prompt_shield(
    user_prompt: str,
    documents: Optional[List[str]] = None,
) -> Optional[bool]:
    """Check for prompt injection / jailbreak attacks.

    Returns:
        True  = attack detected (block the request)
        False = no attack (safe to proceed)
        None  = API failure (fail-open)
    """
    body = {
        "userPrompt": user_prompt,
        "documents": documents or [],
    }
    result = await _call_safety_api("text:shieldPrompt", body, _TEXT_IMAGE_API_VERSION)

    if result is None:
        return None

    user_attack = result.get("userPromptAnalysis", {}).get("attackDetected", False)
    if user_attack:
        logger.info("Prompt shield: attack detected in user prompt")
        return True

    for i, doc_analysis in enumerate(result.get("documentsAnalysis", [])):
        if doc_analysis.get("attackDetected", False):
            logger.info("Prompt shield: attack detected in document[%d]", i)
            return True

    return False


# ── Feature 4: Groundedness Detection ─────────────────────────

async def check_groundedness(
    query: str,
    answer_text: str,
    grounding_sources: List[str],
) -> Optional[Dict[str, Any]]:
    """Check if an answer is grounded in the provided sources.

    Returns dict with ungrounded_detected, ungrounded_percentage,
    ungrounded_details — or None on API failure (fail-open).
    """
    if not answer_text or not grounding_sources:
        return None

    body = {
        "domain": "Generic",
        "task": "QnA",
        "qna": {"query": query},
        "text": answer_text,
        "groundingSources": grounding_sources,
        "reasoning": False,
    }
    result = await _call_safety_api(
        "text:detectGroundedness", body, _GROUNDEDNESS_API_VERSION
    )

    if result is None:
        return None

    ungrounded = result.get("ungroundedDetected", False)
    percentage = result.get("ungroundedPercentage", 0.0)
    details = result.get("ungroundedDetails", [])

    if ungrounded:
        logger.info(
            "Groundedness check: %.1f%% ungrounded, %d ungrounded span(s)",
            percentage, len(details),
        )

    return {
        "ungrounded_detected": ungrounded,
        "ungrounded_percentage": percentage,
        "ungrounded_details": [d.get("text", "") for d in details],
    }


# ── Feature 5: Task Adherence ─────────────────────────────────

async def check_task_adherence(
    tools: List[Dict[str, Any]],
    messages: List[Any],
) -> Optional[bool]:
    """Check if agent tool calls adhere to the user's task.

    Handles both dict messages and OpenAI SDK ChatCompletionMessage objects.

    Returns:
        True  = task risk detected
        False = no risk detected
        None  = API failure (fail-open)
    """
    serialized_messages: List[Dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, dict):
            entry: Dict[str, Any] = {"role": msg.get("role", "user")}
            if msg.get("content"):
                entry["contents"] = msg["content"]
            if msg.get("role") == "system":
                entry["source"] = "Prompt"
            elif msg.get("role") == "tool":
                entry["source"] = "Prompt"
                entry["role"] = "Tool"
                if msg.get("tool_call_id"):
                    entry["toolCallId"] = msg["tool_call_id"]
            else:
                entry["source"] = "Prompt" if msg.get("role") == "user" else "Completion"
            serialized_messages.append(entry)
        else:
            # OpenAI SDK message object
            role = getattr(msg, "role", "assistant")
            entry = {
                "role": role.capitalize() if role != "assistant" else "Assistant",
                "source": "Completion" if role == "assistant" else "Prompt",
                "contents": getattr(msg, "content", "") or "",
            }
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                entry["toolCalls"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                        "id": tc.id,
                    }
                    for tc in tool_calls
                ]
            serialized_messages.append(entry)

    body = {
        "tools": tools,
        "messages": serialized_messages,
    }
    result = await _call_safety_api(
        "agent:analyzeTaskAdherence", body, _TASK_ADHERENCE_API_VERSION
    )

    if result is None:
        return None

    risk_detected = result.get("taskRiskDetected", False)
    if risk_detected:
        details = result.get("details", "no details")
        logger.warning("Task adherence risk detected: %s", details)

    return risk_detected


# ── Convenience wrappers ──────────────────────────────────────

async def moderate_user_text(text: str) -> Tuple[bool, str]:
    """Check user-submitted text (questions, notes, search queries).

    Returns (is_safe, message). On API failure, defaults to safe.
    """
    result = await check_text(text)
    if result is None:
        return True, ""
    if result.is_rejected:
        return False, "Your message was flagged for inappropriate content. Please rephrase."
    return True, ""


async def moderate_output_text(text: str) -> Tuple[bool, str]:
    """Check agent/LLM output text before returning to user.

    Returns (is_safe, message). On API failure, defaults to safe.
    """
    result = await check_text(text)
    if result is None:
        return True, ""
    if result.is_rejected:
        return False, "I'm unable to provide that response. Please try a different question."
    return True, ""


async def moderate_image(image_bytes: bytes) -> Tuple[bool, str]:
    """Check uploaded image before S3 storage.

    Returns (is_safe, message). On API failure, defaults to safe.
    """
    result = await check_image(image_bytes)
    if result is None:
        return True, ""
    if result.is_rejected:
        return False, "This image was flagged for inappropriate content and cannot be uploaded."
    return True, ""


async def shield_agent_prompt(question: str) -> Tuple[bool, str]:
    """Run prompt shield + text moderation on user question before agent.

    Returns (is_safe, message). On API failure, defaults to safe.
    """
    # 1. Text moderation
    text_safe, text_msg = await moderate_user_text(question)
    if not text_safe:
        return False, text_msg

    # 2. Prompt shield
    attack = await check_prompt_shield(question)
    if attack is True:
        return False, "Your question was flagged as a potential prompt injection. Please rephrase."

    return True, ""
