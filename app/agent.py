"""VLM-driven agent with tool calling for document analysis."""
import json
import logging
import os
from typing import Dict, Any, List
from datetime import date

from openai import AsyncAzureOpenAI

from app.db import search_documents
from app.extraction import (
    get_total_spending,
    get_spending_by_merchant,
    get_receipts_by_merchant,
    get_receipts_by_date_range,
)
from app.content_safety import (
    shield_agent_prompt,
    moderate_output_text,
    check_groundedness,
    check_task_adherence,
)

logger = logging.getLogger(__name__)

# Agent configuration (env-configurable)
AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.3"))
AGENT_MAX_RETRIES = int(os.getenv("AGENT_MAX_RETRIES", "0"))


def repair_json(s: str) -> str:
    """Attempt to fix common JSON issues from truncated LLM output."""
    s = s.strip()
    # Fix unterminated strings by closing them
    if s.count('"') % 2 == 1:
        s += '"'
    # Close unclosed braces/brackets
    open_braces = s.count('{') - s.count('}')
    open_brackets = s.count('[') - s.count(']')
    s += ']' * open_brackets + '}' * open_braces
    return s


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search documents by text/semantic similarity. Use for general questions about document content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query text"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_total_spending",
            "description": "Get total spending amount. Use for 'how much did I spend' questions. Omit dates to query ALL TIME.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "Optional start date YYYY-MM-DD. Omit for all time."},
                    "end_date": {"type": "string", "description": "Optional end date YYYY-MM-DD. Omit for all time."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_spending_by_merchant",
            "description": "Get spending broken down by merchant/store. Use for 'where do I spend the most' questions. Omit dates to query ALL TIME.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "Optional start date YYYY-MM-DD. Omit for all time."},
                    "end_date": {"type": "string", "description": "Optional end date YYYY-MM-DD. Omit for all time."},
                    "limit": {"type": "integer", "description": "Max merchants to return", "default": 10}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_receipts_by_merchant",
            "description": "Get all receipts from a specific merchant. Use for 'show me Target receipts' type questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "merchant": {"type": "string", "description": "Merchant name (partial match)"},
                    "limit": {"type": "integer", "description": "Max receipts to return", "default": 20}
                },
                "required": ["merchant"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_receipts_by_date_range",
            "description": "Get all receipts in a date range. Use for 'receipts from last month' type questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "Start date YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "End date YYYY-MM-DD"},
                    "limit": {"type": "integer", "description": "Max receipts to return", "default": 50}
                },
                "required": ["start_date", "end_date"]
            }
        }
    }
]


SYSTEM_PROMPT = """You are a helpful assistant that analyzes the user's screenshots, images, documents and receipts.
All uploaded content is stored as searchable documents.
Your primary goal is to extract information from these files and answer any questions the user has about that information.
You have tools to search documents, query spending data, and retrieve receipt details.

When answering questions:
1. Use the appropriate tool(s) to get data
2. Analyze the results and provide a clear, helpful answer
3. Include specific numbers, dates, and merchant names when relevant
4. If asked about spending, always include the total amount
5. For "when" questions: first check if a date was extracted from the document content. If no date exists, fall back to `created_at` (the upload timestamp) as an approximate date for when the event occurred

Today's date is {today}. Use this to interpret relative dates like "last month" or "this year"."""


async def execute_tool(db, user_id: str, tool_call) -> str:
    """Execute a tool call and return JSON result."""
    name = tool_call.function.name
    raw_args = tool_call.function.arguments

    # Handle malformed JSON from LLM - try repair if needed
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        try:
            args = json.loads(repair_json(raw_args))
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid tool arguments: {e}"})

    if name == "search_documents":
        result = await search_documents(db, user_id, args["query"], limit=5)
    elif name == "get_total_spending":
        result = await get_total_spending(db, user_id, args.get("start_date"), args.get("end_date"))
    elif name == "get_spending_by_merchant":
        result = await get_spending_by_merchant(db, user_id, args.get("start_date"), args.get("end_date"), args.get("limit", 10))
    elif name == "get_receipts_by_merchant":
        result = await get_receipts_by_merchant(db, user_id, args["merchant"], args.get("limit", 20))
    elif name == "get_receipts_by_date_range":
        result = await get_receipts_by_date_range(db, user_id, args["start_date"], args["end_date"], args.get("limit", 50))
    else:
        result = {"error": f"Unknown tool: {name}"}

    return json.dumps(result)


async def ask_agent(db, user_id: str, question: str) -> Dict[str, Any]:
    """VLM-driven agent with tool calling. Supports configurable retries."""
    # ── Content Safety: check user question ──
    is_safe, safety_msg = await shield_agent_prompt(question)
    if not is_safe:
        return {"answer": safety_msg, "sources": []}

    for attempt in range(AGENT_MAX_RETRIES + 1):
        try:
            return await _ask_agent_impl(db, user_id, question)
        except Exception as e:
            if attempt == AGENT_MAX_RETRIES:
                return {
                    "answer": "Sorry, I had trouble processing that. Please try again.",
                    "sources": []
                }
            continue


async def _ask_agent_impl(db, user_id: str, question: str) -> Dict[str, Any]:
    """Internal implementation of ask_agent."""
    client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2024-02-15-preview",
    )
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(today=date.today().isoformat())},
        {"role": "user", "content": question}
    ]

    sources = []

    # Loop: VLM calls tools until it produces final answer
    for _ in range(5):  # Max 5 iterations to prevent infinite loops
        response = await client.chat.completions.create(
            model=deployment,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=AGENT_TEMPERATURE,
        )

        msg = response.choices[0].message
        messages.append(msg)  # Add assistant message to history

        if msg.tool_calls:
            # ── Content Safety: task adherence check (advisory) ──
            task_risk = await check_task_adherence(TOOLS, messages)
            if task_risk is True:
                logger.warning(
                    "Task adherence risk for user=%s, tool_calls=%s",
                    user_id,
                    [tc.function.name for tc in msg.tool_calls],
                )

            # Execute each tool call
            for tool_call in msg.tool_calls:
                result = await execute_tool(db, user_id, tool_call)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
                # Track doc_ids as sources
                try:
                    data = json.loads(result)
                    if isinstance(data, list):
                        for item in data:
                            if "doc_id" in item:
                                sources.append(item["doc_id"])
                            elif "id" in item:
                                sources.append(item["id"])
                except:
                    pass
        else:
            # No more tool calls - this is the final answer
            final_answer = msg.content or "I couldn't find relevant information."

            # ── Content Safety: moderate output text ──
            output_safe, output_msg = await moderate_output_text(final_answer)
            if not output_safe:
                return {"answer": output_msg, "sources": list(set(sources))}

            # ── Content Safety: groundedness check ──
            grounding_sources = []
            for m in messages:
                if isinstance(m, dict) and m.get("role") == "tool":
                    content = m.get("content", "")
                    if content:
                        grounding_sources.append(content)

            if grounding_sources:
                ground_result = await check_groundedness(
                    question, final_answer, grounding_sources
                )
                if ground_result and ground_result.get("ungrounded_detected"):
                    pct = ground_result.get("ungrounded_percentage", 0)
                    if pct > 50:
                        final_answer += (
                            "\n\n*Note: Some parts of this response may not be "
                            "fully supported by your documents.*"
                        )
                        logger.info(
                            "Groundedness warning appended: %.1f%% ungrounded", pct
                        )

            return {
                "answer": final_answer,
                "sources": list(set(sources))
            }

    # Max iterations reached
    return {
        "answer": "I wasn't able to complete the analysis. Please try a simpler question.",
        "sources": list(set(sources))
    }
