"""VLM-driven agent with tool calling for document analysis."""
import json
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
            "description": "Get total spending amount. Use for 'how much did I spend' questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "Start date YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "End date YYYY-MM-DD"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_spending_by_merchant",
            "description": "Get spending broken down by merchant/store. Use for 'where do I spend the most' questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "Start date YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "End date YYYY-MM-DD"},
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


SYSTEM_PROMPT = """You are a helpful assistant that analyzes the user's documents and receipts.
You have access to tools to search documents, query spending data, and retrieve receipt details.

When answering questions:
1. Use the appropriate tool(s) to get data
2. Analyze the results and provide a clear, helpful answer
3. Include specific numbers, dates, and merchant names when relevant
4. If asked about spending, always include the total amount

Today's date is {today}. Use this to interpret relative dates like "last month" or "this year"."""


async def execute_tool(db, user_id: str, tool_call) -> str:
    """Execute a tool call and return JSON result."""
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

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
    """VLM-driven agent with tool calling."""
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
            tool_choice="auto"
        )

        msg = response.choices[0].message
        messages.append(msg)  # Add assistant message to history

        if msg.tool_calls:
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
            return {
                "answer": msg.content or "I couldn't find relevant information.",
                "sources": list(set(sources))  # Dedupe
            }

    # Max iterations reached
    return {
        "answer": "I wasn't able to complete the analysis. Please try a simpler question.",
        "sources": list(set(sources))
    }
