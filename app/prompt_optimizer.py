"""Constraint mining from regeneration feedback.

Batch job that:
1. Finds sessions where users regenerated (2+ attempts)
2. Builds rejected/accepted pairs with tool traces
3. Feeds pairs to LLM to extract behavioral constraints
4. Stores constraints for injection into the agent system prompt
"""
import json
import logging
import os
from typing import Dict, Any, List

from openai import AsyncAzureOpenAI

from app.db import (
    get_unmined_sessions,
    get_session_attempts,
    save_prompt_constraints,
)

logger = logging.getLogger(__name__)

# Tool descriptions for context in the mining prompt
TOOL_SUMMARY = """
- search_documents: Semantic similarity search over OCR text
- get_total_spending: Sum spending in optional date range
- get_spending_by_merchant: Group spending by store/business
- get_receipts_by_merchant: Get all receipts from specific merchant
- get_receipts_by_date_range: Get receipts in date window
""".strip()

MINING_PROMPT = """You are analyzing user feedback on an AI agent that answers questions about receipts and documents.

The agent has these tools:
{tool_summary}

Below are sessions where a user rejected the agent's response and regenerated until satisfied.
For each session you can see:
- The user's query
- Each attempt with its tool calls and the answer produced
- The LAST attempt is the one the user accepted (implicitly, by not regenerating again)
- All earlier attempts were rejected

Analyze the patterns across ALL sessions and extract general rules for improving the agent.
Focus on recurring issues, not one-off flukes.

Each rule should be one of two types:
- "tool_selection": guidance on which tool to use for which type of query
- "answer_generation": guidance on how to format, phrase, or structure answers

IMPORTANT — Do NOT extract rules that:
- Change the overall response structure or format (e.g. "always use bullet points", "keep answers under N sentences")
- Contradict the agent's core numbered instructions (tool usage, citation tags, date handling)
- Affect how the agent tags or cites sources
Focus only on domain-specific improvements: which tool to pick for a query type, what data to include in answers, how to phrase domain-specific information.

Output a JSON array of rules. Each rule should be actionable and concise.
Example format:
[
  {{"type": "tool_selection", "rule": "When the user asks about spending at a specific store, use get_spending_by_merchant instead of search_documents"}},
  {{"type": "answer_generation", "rule": "Always include the dollar amount when answering spending questions"}}
]

If no clear patterns emerge, return an empty array: []

Sessions:
{sessions_json}"""


async def build_preference_pairs(conn, sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build rejected/accepted pairs from unmined sessions."""
    pairs = []
    for session in sessions:
        attempts = await get_session_attempts(conn, session["id"])
        if len(attempts) < 2:
            continue

        rejected = []
        for attempt in attempts[:-1]:
            rejected.append({
                "tool_calls": attempt["tool_trace"],
                "answer": attempt["answer"],
            })

        accepted = attempts[-1]

        pairs.append({
            "session_id": session["id"],
            "query": session["query_text"],
            "rejected_attempts": rejected,
            "accepted_attempt": {
                "tool_calls": accepted["tool_trace"],
                "answer": accepted["answer"],
            },
        })
    return pairs


async def mine_constraints(pairs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Feed preference pairs to LLM and extract constraints."""
    if not pairs:
        return []

    client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2024-02-15-preview",
    )
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

    # Build session descriptions for the prompt
    sessions_for_prompt = []
    for pair in pairs:
        session_desc = {
            "query": pair["query"],
            "attempts": [],
        }
        for i, rej in enumerate(pair["rejected_attempts"]):
            session_desc["attempts"].append({
                "attempt": i + 1,
                "status": "REJECTED",
                "tool_calls": rej["tool_calls"],
                "answer": rej["answer"][:300],  # Truncate long answers
            })
        acc = pair["accepted_attempt"]
        session_desc["attempts"].append({
            "attempt": len(pair["rejected_attempts"]) + 1,
            "status": "ACCEPTED",
            "tool_calls": acc["tool_calls"],
            "answer": acc["answer"][:300],
        })
        sessions_for_prompt.append(session_desc)

    prompt = MINING_PROMPT.format(
        tool_summary=TOOL_SUMMARY,
        sessions_json=json.dumps(sessions_for_prompt, indent=2),
    )

    response = await client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content or "[]"
    try:
        parsed = json.loads(raw)
        # Handle both {"rules": [...]} and bare [...]
        if isinstance(parsed, dict):
            rules = parsed.get("rules", [])
        elif isinstance(parsed, list):
            rules = parsed
        else:
            rules = []
    except json.JSONDecodeError:
        logger.error("Failed to parse mining LLM output: %s", raw[:200])
        return []

    # Validate structure
    valid = []
    for r in rules:
        if isinstance(r, dict) and "type" in r and "rule" in r:
            if r["type"] in ("tool_selection", "answer_generation"):
                valid.append({"type": r["type"], "rule": r["rule"]})
    return valid


async def run_mining_job(conn) -> Dict[str, Any]:
    """Run the full constraint mining pipeline.

    Returns stats: {sessions_processed, constraints_mined}
    """
    sessions = await get_unmined_sessions(conn)
    if not sessions:
        logger.info("No unmined sessions with regenerations found")
        return {"sessions_processed": 0, "constraints_mined": 0}

    logger.info("Found %d unmined sessions to process", len(sessions))

    pairs = await build_preference_pairs(conn, sessions)
    if not pairs:
        return {"sessions_processed": 0, "constraints_mined": 0}

    constraints = await mine_constraints(pairs)
    session_ids = [p["session_id"] for p in pairs]

    saved = await save_prompt_constraints(conn, constraints, session_ids)

    logger.info(
        "Mining complete: %d sessions → %d constraints",
        len(pairs), saved,
    )
    return {"sessions_processed": len(pairs), "constraints_mined": saved}
