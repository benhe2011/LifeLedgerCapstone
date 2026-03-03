"""VLM-driven ReAct agent with tool calling for document analysis."""
import json
import logging
import os
import re
from typing import Dict, Any, List, Annotated, TypedDict, Union
from datetime import date
from operator import add
from openai import AsyncAzureOpenAI
from langgraph.graph import StateGraph, END
from app.db import search_documents, get_active_constraints
from app.extraction import (
    get_total_spending,
    get_spending_by_merchant,
    get_receipts_by_merchant,
    get_receipts_by_date_range,
    get_all_receipt_texts,
    detect_recurring_costs,
    detect_trips,
)
from app.content_safety import (
    shield_agent_prompt,
    moderate_output_text,
    check_groundedness,
    check_task_adherence,
)

logger = logging.getLogger(__name__)

# --- Configuration ---
AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.3"))
AGENT_MAX_RETRIES = int(os.getenv("AGENT_MAX_RETRIES", "0"))
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")

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


def extract_cited_sources(answer: str) -> tuple[str, list[str]]:
    """Extract cited doc_ids from answer and return (clean_answer, cited_ids)."""
    cited_ids: list[str] = []
    def _collect(m):
        cited_ids.extend(s.strip() for s in m.group(1).split(',') if s.strip())
        return ''
    clean_answer = re.sub(r'<!--cited:([\w,\-]+)-->', _collect, answer).rstrip()
    return clean_answer, cited_ids


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
    },
    {
        "type": "function",
        "function": {
            "name": "get_recurring_costs",
            "description": "Detect recurring subscriptions and charges. Returns merchants with recurring billing patterns (monthly or annual), estimated costs, and next expected charge date. Use for 'what are my subscriptions' or 'recurring costs' questions.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_trips",
            "description": "Detect travel trips by clustering travel-related documents (hotels, flights, bookings) by date proximity. Returns trip summaries with dates, locations, costs, and linked documents. Use for 'show me my trips' or 'travel expenses' questions.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_receipt_texts",
            "description": "Get all receipt documents with their full item-level text. Use for broad questions that need reasoning across ALL purchases (e.g., 'what unhealthy items did I buy', 'show me all electronics purchases', 'what did I buy most frequently'). Returns merchant, date, total, and full OCR text containing line items.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max receipts to return", "default": 50}
                }
            }
        }
    }
]


SYSTEM_PROMPT_BASE = """You are a helpful assistant that analyzes the user's screenshots, images, documents and receipts.
All uploaded content is stored as searchable documents.
Your primary goal is to extract information from these files and answer any questions the user has about that information.
You have tools to search documents, query spending data, and retrieve receipt details.

When answering questions:
1. Use the appropriate tool(s) to get data
2. Analyze the results and provide a clear, helpful answer
3. Include specific numbers, dates, and merchant names when relevant
4. If asked about spending, always include the total amount
5. For "when" questions: first check if a date was extracted from the document content. If no date exists, fall back to `created_at` (the upload timestamp) as an approximate date for when the event occurred
6. CRITICAL — Source citation: You MUST end every answer with a citation tag listing the doc_ids of documents you used. Format: <!--cited:doc_id1,doc_id2,doc_id3-->
   Example: If tool results included documents with doc_id "abc-123" and "def-456" and you referenced both, end your answer with: <!--cited:abc-123,def-456-->
   Only cite documents whose data you directly used. This tag is required even if you only used one document.

Today's date is {today}. Use this to interpret relative dates like "last month" or "this year"."""


async def build_system_prompt(db) -> str:
    """Build system prompt with base instructions + mined constraints."""
    prompt = SYSTEM_PROMPT_BASE.format(today=date.today().isoformat())
    
    constraints = await get_active_constraints(db)
    if not constraints:
        return prompt
    
    tool_rules = [c["rule"] for c in constraints if c["type"] == "tool_selection"]
    answer_rules = [c["rule"] for c in constraints if c["type"] == "answer_generation"]
    
    if tool_rules:
        prompt += "\n\nTool selection guidance (learned from user feedback):\n"
        for rule in tool_rules:
            prompt += f"- {rule}\n"

    if answer_rules:
        prompt += "\n\nAnswer formatting guidance (learned from user feedback):\n"
        for rule in answer_rules:
            prompt += f"- {rule}\n"

    # Anchor: reinforce critical fixed rules after mined constraints
    prompt += "\nReminder: Always end your answer with the <!--cited:doc_id1,doc_id2--> citation tag as described above."

    return prompt


async def execute_tool(db, user_id: str, tool_call: Any) -> str:
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
    elif name == "get_recurring_costs":
        result = await detect_recurring_costs(db, user_id)
    elif name == "get_trips":
        result = await detect_trips(db, user_id)
    elif name == "get_all_receipt_texts":
        result = await get_all_receipt_texts(db, user_id, args.get("limit", 50))
    else:
        result = {"error": f"Unknown tool: {name}"}

    return json.dumps(result)

# --- LangGraph Implementation ---

class AgentState(TypedDict):
    messages: Annotated[List[Dict[str, Any]], add]
    db: Any
    user_id: str
    sources: Annotated[List[str], add]
    tool_trace: Annotated[List[Dict[str, Any]], add]
    final_result: Dict[str, Any]

async def call_model(state: AgentState):
    client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2024-02-15-preview"
    )
    sys_prompt = await build_system_prompt(state["db"])
    msgs = [{"role": "system", "content": sys_prompt}] + state["messages"]
    
    response = await client.chat.completions.create(
        model=DEPLOYMENT,
        messages=msgs,
        tools=TOOLS,
        tool_choice="auto",
        temperature=AGENT_TEMPERATURE
    )
    return {"messages": [response.choices[0].message]}

async def call_tools(state: AgentState):
    last_msg = state["messages"][-1]
    new_messages = []
    new_sources = []
    new_traces = []
    
    task_risk = await check_task_adherence(TOOLS, state["messages"])
    if task_risk is True:
        logger.warning(
            "Task adherence risk for user=%s, tool_calls=%s",
            state["user_id"],
            [tc.function.name for tc in last_msg.tool_calls],
            )
    
    for tool_call in last_msg.tool_calls:
        result_str = await execute_tool(state["db"], state["user_id"], tool_call)
        new_messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result_str})
        
        try:
            args = json.loads(tool_call.function.arguments)
        except:
            args = tool_call.function.arguments

        new_traces.append({
            "tool": tool_call.function.name,
            "args": args,
            "result_summary": result_str[:500],
            "task_adherence_risk": bool(task_risk)
        })

        try:
            data = json.loads(result_str)
            if isinstance(data, list):
                for item in data:
                    if "doc_id" in item: new_sources.append(str(item["doc_id"]))
                    elif "id" in item: new_sources.append(str(item["id"]))
        except: pass

    return {"messages": new_messages, "sources": new_sources, "tool_trace": new_traces}

def router(state: AgentState):
    last_msg = state["messages"][-1]
    # If the LLM generated tool calls, go to tools
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    # If it's a Tool message or an Assistant message with content, the loop continues 
    # until the Assistant provides a final text response without tool calls.
    return END

# --- Main Interface ---

async def ask_agent(db, user_id: str, question: str, history: list[dict] | None = None) -> Dict[str, Any]:
    """VLM-driven agent with LangGraph ReAct flow and UI-compatible serialization."""
    
    # 1. Content Safety: check user question (Prompt Shield)
    gate_result = await shield_agent_prompt(question)
    if not gate_result.is_safe:
        return {
            "answer": gate_result.message,
            "documents": [],
            "query": question,
            "safety": {
                "strategy": gate_result.strategy.value if gate_result.strategy else None,
                "message": gate_result.message,
                "detail": gate_result.detail,
            },
            "groundedness": None
        }

    # 2. Build the LangGraph Workflow
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", router)
    workflow.add_edge("tools", "agent")
    graph = workflow.compile()

    for attempt in range(AGENT_MAX_RETRIES + 1):
        try:
            # Initial state — prepend conversation history if provided
            messages = []
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": question})

            inputs = {
                "messages": messages,
                "db": db,
                "user_id": user_id,
                "sources": [],
                "tool_trace": []
            }
            
            output = await graph.ainvoke(inputs)

            # 3. Extract final answer safely (handles Objects vs Dicts)
            final_answer = "I couldn't find a specific answer in your documents."
            for m in reversed(output.get("messages", [])):
                is_dict = isinstance(m, dict)
                role = m.get("role") if is_dict else getattr(m, "role", "")
                content = m.get("content") if is_dict else getattr(m, "content", "")
                
                if role.lower() == "assistant" and content:
                    has_tools = (m.get("tool_calls") if is_dict else getattr(m, "tool_calls", None))
                    if not has_tools:
                        final_answer = content
                        break

            # 4. Content Safety: moderate final output
            out_gate = await moderate_output_text(final_answer)
            if not out_gate.is_safe:
                return {
                    "answer": out_gate.message,
                    "documents": list(set(str(s) for s in output.get("sources", []))),
                    "query": question,
                    "safety": {
                        "strategy": out_gate.strategy.value if out_gate.strategy else None,
                        "message": out_gate.message,
                        "detail": out_gate.detail
                    },
                    "groundedness": None
                }

            # 5. Groundedness logic
            grounding_sources = []
            for m in output.get("messages", []):
                is_dict = isinstance(m, dict)
                role = (m.get("role") if is_dict else getattr(m, "role", "")).lower()
                content = m.get("content") if is_dict else getattr(m, "content", "")
                if role == "tool" and content:
                    grounding_sources.append(content)

            groundedness_info = None
            if grounding_sources:
                ground_result = await check_groundedness(question, final_answer, grounding_sources)
                if ground_result and ground_result.get("ungrounded_detected"):
                    pct = ground_result.get("ungrounded_percentage", 0)
                    if pct > 50:
                        groundedness_info = {
                            "ungrounded_pct": pct,
                            "message": "Some parts of this response may not be fully supported by your documents.",
                        }

            # 6. Final Serialization aligned with "OLD" working UI schema
            clean_answer, cited_sources = extract_cited_sources(str(final_answer))
            clean_result = {
                "answer": clean_answer,
                "documents": list(set(str(s) for s in output.get("sources", []))),
                "cited_sources": cited_sources,
                "query": question,
                "session_id": None,  # Populated by the calling route handler
                "safety": None,      # UI expects null when everything is safe
                "groundedness": groundedness_info
            }

            # Optionally include tool_trace if needed for internal logging/debugging
            # clean_result["tool_trace"] = output.get("tool_trace", [])

            return clean_result

        except Exception as e:
            logger.error(f"Agent attempt {attempt} failed: {str(e)}", exc_info=True)
            if attempt == AGENT_MAX_RETRIES:
                return {
                    "answer": "I encountered an error while processing your request.",
                    "documents": [],
                    "query": question,
                    "safety": None,
                    "error_detail": str(e)
                }
            continue
