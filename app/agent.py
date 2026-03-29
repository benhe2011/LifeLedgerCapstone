"""VLM-driven ReAct agent with tool calling for document analysis."""
import json
import logging
import os
import re
from typing import Dict, Any, List, Annotated, TypedDict
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
    get_document_overview,
    get_all_document_texts,
    detect_recurring_costs,
    detect_trips,
    get_earnings_summary,
    get_deductions_breakdown,
    get_income_vs_spending,
    get_lease_details,
    get_recurring_income,
)
from app.content_safety import (
    shield_agent_prompt,
    moderate_output_text,
    check_groundedness,
    check_task_adherence,
)
from ddgs import DDGS

logger = logging.getLogger(__name__)

# --- Configuration ---
AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.3"))
AGENT_MAX_RETRIES = int(os.getenv("AGENT_MAX_RETRIES", "0"))
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
# Check safety break to prevent infinite loops (e.g., max n reroutes)
REROUTES = 2

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
    cited_ids: list[str] = []
    def _collect(m):
        # This now grabs doc_ids AND URLs separated by commas
        cited_ids.extend(s.strip() for s in m.group(1).split(',') if s.strip())
        return ''
    # Matches <!--cited:abc-123,https://example.com-->
    clean_answer = re.sub(r'<!--cited:(.*?)-->', _collect, answer).rstrip()
    return clean_answer, cited_ids



CHARTABLE_TOOLS = {
    "get_spending_by_merchant": "spending_by_merchant",
    "get_receipts_by_date_range": "spending_over_time",
    "get_receipts_by_merchant": "receipt_table",
}


def extract_chart_data(tool_results: list) -> list | None:
    """Inspect tool results and build chart_data entries for frontend rendering."""
    charts = []
    for tr in tool_results:
        chart_type = CHARTABLE_TOOLS.get(tr["tool"])
        data = tr.get("data")
        if not chart_type or not data or not isinstance(data, list) or len(data) == 0:
            continue

        if chart_type == "spending_by_merchant":
            charts.append({
                "type": "spending_by_merchant",
                "title": "Spending by Merchant",
                "data": data,
            })

        elif chart_type == "spending_over_time":
            from collections import defaultdict
            monthly: dict[str, float] = defaultdict(float)
            for r in data:
                if r.get("date") and r.get("total"):
                    monthly[r["date"][:7]] += r["total"]
            if monthly:
                charts.append({
                    "type": "spending_over_time",
                    "title": "Spending Over Time",
                    "data": [{"month": m, "total": round(t, 2)} for m, t in sorted(monthly.items())],
                })
            charts.append({
                "type": "receipt_table",
                "title": "Receipts",
                "data": data,
            })

        elif chart_type == "receipt_table":
            merchant = tr.get("args", {}).get("merchant", "")
            charts.append({
                "type": "receipt_table",
                "title": f"Receipts from {merchant}" if merchant else "Receipts",
                "data": data,
            })

    return charts if charts else None


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
            "description": "Get ONLY a single total spending number. Use ONLY when the user wants just a grand total. For breakdowns, trends, or 'show my spending' questions, prefer get_spending_by_merchant or get_receipts_by_date_range instead. Omit dates to query ALL TIME.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "Optional start date YYYY-MM-DD. Omit for all time."},
                    "end_date": {"type": "string", "description": "Optional end date YYYY-MM-DD. Omit for all time."},
                    "doc_type": {"type": "string", "enum": ["receipt", "subscription", "invoice", "payslip", "rental_agreement"], "description": "Optional filter by document type. Omit to include all types."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_spending_by_merchant",
            "description": "Get spending broken down by merchant/store. Use for 'where do I spend the most', 'show my spending', or any spending overview questions. Omit dates to query ALL TIME.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "Optional start date YYYY-MM-DD. Omit for all time."},
                    "end_date": {"type": "string", "description": "Optional end date YYYY-MM-DD. Omit for all time."},
                    "limit": {"type": "integer", "description": "Max merchants to return", "default": 10},
                    "doc_type": {"type": "string", "enum": ["receipt", "subscription", "invoice", "payslip", "rental_agreement"], "description": "Optional filter by document type. Omit to include all types."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_receipts_by_merchant",
            "description": "Get all documents from a specific merchant or service. Use for 'show me Target receipts' or 'show me Netflix charges' type questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "merchant": {"type": "string", "description": "Merchant or service name (partial match)"},
                    "limit": {"type": "integer", "description": "Max documents to return", "default": 20},
                    "doc_type": {"type": "string", "enum": ["receipt", "subscription", "invoice", "payslip", "rental_agreement"], "description": "Optional filter by document type. Omit to include all types."}
                },
                "required": ["merchant"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_receipts_by_date_range",
            "description": "Get all documents in a date range with dates and amounts. Use for spending trends, timelines, 'show my spending over time', or 'documents from last month' type questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "Start date YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "End date YYYY-MM-DD"},
                    "limit": {"type": "integer", "description": "Max documents to return", "default": 50},
                    "doc_type": {"type": "string", "enum": ["receipt", "subscription", "invoice", "payslip", "rental_agreement"], "description": "Optional filter by document type. Omit to include all types."}
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
            "description": "Get all extracted documents with their full item-level text. Use for broad questions that need reasoning across ALL purchases or documents (e.g., 'what unhealthy items did I buy', 'show me all electronics purchases', 'what subscriptions do I have'). Returns merchant, date, total, and full OCR text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max documents to return", "default": 50},
                    "doc_type": {"type": "string", "enum": ["receipt", "subscription", "invoice", "payslip", "rental_agreement"], "description": "Optional filter by document type. Omit to include all types."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_overview",
            "description": "Get a lightweight overview of all uploaded documents with their types, dates, and amounts. Use for 'what documents do I have', 'what types of documents', 'how many receipts', or any question about the user's document collection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max documents to return", "default": 50}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_document_texts",
            "description": "Get all documents with their full OCR text, including non-receipt items like flyers, screenshots, notes, emails, and posters. Use when the user asks about content in non-receipt documents or needs to reason across all uploaded content regardless of type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max documents to return", "default": 30}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_earnings_summary",
            "description": "Get earnings/income from payslips over time. Returns employer, pay date, net pay, and gross pay for each payslip. Use for 'how much did I earn', 'show my income', 'payslip history' questions. Omit dates to query ALL TIME.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "Optional start date YYYY-MM-DD. Omit for all time."},
                    "end_date": {"type": "string", "description": "Optional end date YYYY-MM-DD. Omit for all time."},
                    "limit": {"type": "integer", "description": "Max payslips to return", "default": 50}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_deductions_breakdown",
            "description": "Get aggregated payslip deductions breakdown (federal tax, state tax, social security, medicare, 401k, health insurance, etc.). Use for 'how much tax did I pay', 'show my deductions', 'total withholdings' questions. Omit dates to query ALL TIME.",
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
            "name": "get_income_vs_spending",
            "description": "Compare monthly income (from payslips) against spending (from receipts, subscriptions, invoices). Returns income, spending, and net savings per month. Use for 'am I saving money', 'income vs expenses', 'net savings', 'financial overview' questions. Omit dates to query ALL TIME.",
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
            "name": "get_lease_details",
            "description": "Get rental agreement details: landlord, rent amount, lease dates, security deposit, property address. Use for 'what's my rent', 'when does my lease end', 'show my rental agreement', 'security deposit' questions.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_recurring_income",
            "description": "Detect recurring income patterns from payslips. Returns employer, pay frequency (weekly/biweekly/monthly), average net pay, and estimated monthly/annual income. Use for 'how often do I get paid', 'what's my salary', 'recurring income' questions.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "LAST RESORT ONLY — almost never needed. Search the web for external information. "
            "RULES: (1) You MUST call at least one local document tool FIRST and review its results. "
            "(2) Only call web_search if the user explicitly requests a web search, OR if local tools "
            "returned ZERO relevant results and the question cannot be answered from the user's uploads. "
            "(3) Questions about the user's own documents, spending, receipts, or any topic they may "
            "have uploaded images about should NEVER trigger web_search — the answer is in their uploads. "
            "(4) Do NOT call this tool speculatively, as a first step, or to 'supplement' a local answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query (e.g., 'cheaper healthy alternatives to Target brand eggs')"}
            },
            "required": ["query"]
        }
    }
    }
]

TOOL_BUCKETS = {
    "spending": ["get_total_spending", "get_spending_by_merchant", "get_receipts_by_merchant",
                 "get_receipts_by_date_range", "get_recurring_costs", "web_search"],
    "search": ["search_documents", "get_document_overview", "get_all_document_texts",
               "get_all_receipt_texts", "get_lease_details", "web_search"],
    "income": ["get_earnings_summary", "get_deductions_breakdown", "get_income_vs_spending",
               "get_recurring_income", "web_search"],
    "travel": ["get_trips", "web_search"],
}

REROUTE_TOOL = {
    "type": "function",
    "function": {
        "name": "request_category_change",
        "description": "Switch to a different tool category when you need tools not available in your current category.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_category": {
                    "type": "string",
                    "enum": list(TOOL_BUCKETS.keys()),
                    "description": "The category to switch to"
                },
                "reason": {"type": "string", "description": "Why you need this category"}
            },
            "required": ["target_category"]
        }
    }
}

SYSTEM_PROMPT_BASE = """You are a helpful assistant that analyzes the user's screenshots, images, documents and receipts.
All uploaded content is stored as searchable documents.
Your primary goal is to extract information from these files and answer any questions the user has about that information.
You have tools to search documents, query spending data, and retrieve document details.
Documents are classified as: receipts, subscriptions, invoices, payslips, and rental agreements.
Many tools accept an optional doc_type filter. Use it when the user asks about a specific type.

CRITICAL — Income vs. Spending:
- Payslips represent INCOME (money the user earns), not spending. NEVER include payslip amounts in spending totals or spending breakdowns.
- When the user asks about spending, costs, or expenses: use spending tools (get_total_spending, get_spending_by_merchant, etc.). These automatically exclude payslips.
- When the user asks about income, earnings, salary, or pay: use income tools (get_earnings_summary, get_deductions_breakdown, get_recurring_income).
- When the user asks about savings, net income, or financial overview: use get_income_vs_spending to compare both.
- Payslip deductions (taxes, insurance, 401k) are NOT separate expenses — they are part of the payslip. Use get_deductions_breakdown to analyze them, not spending tools.

CRITICAL — Rental Agreements:
- Rental agreements are reference documents representing an ongoing obligation, not a one-time transaction.
- The rent amount is a recurring monthly cost. Use get_recurring_costs to surface it (it appears automatically alongside subscriptions).
- For lease details (end date, deposit, terms): use get_lease_details.
- Do NOT count the rental agreement total_amount as a single spending event.

CRITICAL — Tax ambiguity:
- "How much tax did I pay" likely refers to payslip deductions (federal/state income tax withholdings). Use get_deductions_breakdown.
- "How much sales tax" refers to receipt-level tax. Use spending/search tools to find receipts with tax line items.
- If ambiguous, prefer payslip deductions and mention that sales tax from receipts is a separate query.

When answering questions:
1. TOOL PRIORITY — The vast majority of questions can be fully answered with local document tools alone.
   a. ALWAYS start with local document tools to look up information from the user's own records.
   b. Only call web_search if ALL of these are true: (i) you already called a local tool, (ii) it returned
      no useful results, AND (iii) the question cannot be answered from the user's uploads.
   c. NEVER call web_search as a first step, before attempting local tools, or to supplement an already
      sufficient local answer. If local tools answered the question, do NOT also web search.
   d. Exception: if the user explicitly asks to search the web, you may call web_search directly.
2. Use multiple tools if needed — if your first retrieval returns no results or partial results,
   try a broader tool before concluding. Do not give up after a single failed attempt.
3. Analyze the results and provide a clear, helpful answer
4. Include specific numbers, dates, and merchant names when relevant
5. If asked about spending, always include the total amount
6. For "when" questions: first check if a date was extracted from the document content. If no date exists, fall back to `created_at` (the upload timestamp) as an approximate date for when the event occurred
7. CRITICAL — Source citation: You MUST end every answer with a citation tag listing all doc_ids OR web URLs you used to generate the response.
   Format: <!--cited:source1,source2-->
   - For internal documents: Use the 'doc_id' provided in the tool results.
   - For web search: Use the full URL (href) provided in the search results.
   Example: If you used a Target receipt (doc_id "abc-123") and a health article (URL "https://healthline.com"), end your answer with: <!--cited:abc-123,https://healthline.com-->
   Only cite sources whose data you directly referenced. This tag is required even if you only used one source.
   Do NOT add a human-readable "Source:", "Sources:", or "Reference:" section — the citation tag above is the only source attribution needed.
8. CRITICAL: Do NOT invent items. Only list products explicitly found in the provided tool results. If the tool result does not contain a line-item breakdown, state that you can only see the total amount.
9. CRITICAL: You must NEVER answer from memory. Always use the appropriate tool(s) to fetch data first.
10. CRITICAL — Final answer quality: Your final response MUST report what the tools actually returned.
    NEVER write future-tense phrases like "searching now", "let me check", "I will look", or
    "searching for X now…" in your final answer — tools have already been called by the time
    you write your response. Only state "I couldn't find any information about [topic]" after
    making a genuine multi-tool effort (trying at least 2 tools). Do not give up after a single
    empty result — try a broader retrieval approach first.
11. CRITICAL — Autonomous action: You are an autonomous agent. NEVER tell the user to call
    a tool, switch a category, perform a search, or take any technical action themselves.
    If a tool is needed, call it. If a category switch is needed, call request_category_change.
    Instructing the user to do what you can do yourself is always wrong.
12. WEB SOURCES: When using 'web_search' results, provide the answer in your own words,
   but ALWAYS include the source title as a clickable Markdown link.
   Example: 'According to [Healthline](https://healthline.com), lentils are a cheaper protein.'

You currently have access to {category}-category tools only. If you need tools from a different category ({available_categories}), call request_category_change with the target category.

Today's date is {today}. Use this to interpret relative dates like "last month" or "this year"."""


async def build_system_prompt(db, category: str) -> str:
    """Build system prompt with base instructions + mined constraints."""
    available_categories = ", ".join(TOOL_BUCKETS.keys())
    prompt = SYSTEM_PROMPT_BASE.format(
        today=date.today().isoformat(),
        category=category,
        available_categories=available_categories,
    )
    
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
    
    logger.info(f"AGENT_ACTION: Executing {name} with args {args}")

    if name == "search_documents":
        result = await search_documents(db, user_id, args["query"], limit=5)
    elif name == "get_total_spending":
        result = await get_total_spending(db, user_id, args.get("start_date"), args.get("end_date"), doc_type=args.get("doc_type"))
    elif name == "get_spending_by_merchant":
        result = await get_spending_by_merchant(db, user_id, args.get("start_date"), args.get("end_date"), args.get("limit", 10), doc_type=args.get("doc_type"))
    elif name == "get_receipts_by_merchant":
        result = await get_receipts_by_merchant(db, user_id, args["merchant"], args.get("limit", 20), doc_type=args.get("doc_type"))
    elif name == "get_receipts_by_date_range":
        result = await get_receipts_by_date_range(db, user_id, args["start_date"], args["end_date"], args.get("limit", 50), doc_type=args.get("doc_type"))
    elif name == "get_recurring_costs":
        result = await detect_recurring_costs(db, user_id)
    elif name == "get_trips":
        result = await detect_trips(db, user_id)
    elif name == "get_all_receipt_texts":
        result = await get_all_receipt_texts(db, user_id, args.get("limit", 50), doc_type=args.get("doc_type"))
    elif name == "get_document_overview":
        result = await get_document_overview(db, user_id, args.get("limit", 50))
    elif name == "get_all_document_texts":
        result = await get_all_document_texts(db, user_id, args.get("limit", 30))
    elif name == "get_earnings_summary":
        result = await get_earnings_summary(db, user_id, args.get("start_date"), args.get("end_date"), args.get("limit", 50))
    elif name == "get_deductions_breakdown":
        result = await get_deductions_breakdown(db, user_id, args.get("start_date"), args.get("end_date"))
    elif name == "get_income_vs_spending":
        result = await get_income_vs_spending(db, user_id, args.get("start_date"), args.get("end_date"))
    elif name == "get_lease_details":
        result = await get_lease_details(db, user_id)
    elif name == "get_recurring_income":
        result = await get_recurring_income(db, user_id)
    elif name == "web_search":
        query = args.get("query")
        try:
            with DDGS() as ddgs:
                # text() returns title, href (url), and body (snippet)
                results = [r for r in ddgs.text(query, max_results=5)]
                result = results if results else {"message": "No web results found."}
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            result = {"error": "Web search is currently unavailable."}
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
    tool_results: Annotated[List[Dict[str, Any]], add]
    selected_category: str     # Tracking the active bucket
    reroute_count: int         # Safety to prevent infinite re-routing

async def route_query(state: AgentState):
    client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2024-02-15-preview"
    )
    
    # Router Tool forces a choice including 'none'
    router_tool = {
        "type": "function",
        "function": {
            "name": "select_category",
            "description": "Select the tool category. Use 'none' for greetings or general chat.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": list(TOOL_BUCKETS.keys()) + ["none"]}
                },
                "required": ["category"]
            }
        }
    }
    # NEW: Get the current active category from state
    current_cat = state.get("selected_category", "none")
    
    router_prompt = f"""You are a Tool Router. Analyze the user's request and the ENTIRE conversation history to pick the best category.
    CURRENT ACTIVE CATEGORY: '{current_cat}'
    CATEGORIES:
    - 'spending': Use for totals, aggregate spending, trends, or merchant-level summaries (how much, where, when).
    - 'search': Use for specific items, products, line-item details, "what" was purchased, lease/rental details.
    - 'income': Use for earnings, salary, payslips, deductions, tax withholdings, income vs spending, net savings, financial overview.
    - 'travel': Use for trips, vacations, hotel bookings, or flight clusters.
    - 'none': Use for greetings, general chat, or questions requiring no data.

    ROUTING RULES (CRITICAL):
    1. If the user asks for "items," "products," or "line details" and you are currently in 'spending', you MUST switch to 'search'.
    2. If the user asks about a "trip" or "vacation" and you are in 'spending' or 'search', you MUST switch to 'travel'.
    3. If the user asks about income, earnings, salary, pay, deductions, tax withholdings, or savings: use 'income'.
    4. If the user asks about rent, lease, landlord, security deposit: use 'search' (lease details are in search bucket).
    5. Today's Date: {date.today().isoformat()}
    6. Only use 'none' for pure greetings or off-topic chat with no connection to the user's data.
       When in doubt, pick the most relevant data category — any category with tools is better
       than 'none' if the question could relate to the user's uploaded documents."""

    response = await client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "system", "content": router_prompt}] + state["messages"],
        tools=[router_tool],
        tool_choice={"type": "function", "function": {"name": "select_category"}}
    )
    
    raw_args = response.choices[0].message.tool_calls[0].function.arguments
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        args = json.loads(repair_json(raw_args))
    return {
        "selected_category": args["category"], 
        "reroute_count": state.get("reroute_count", 0)
    }

async def call_model(state: AgentState):
    client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2024-02-15-preview"
    )
    cat = state.get("selected_category", "none")
    # 1. Filter tools by category
    current_tools = [t for t in TOOLS if t["function"]["name"] in TOOL_BUCKETS.get(cat, [])]
    
    # 2. Add Re-route tool (unless it's a general chat or we've looped too much)
    if cat != "none" and state.get("reroute_count", 0) < REROUTES:
        current_tools.append(REROUTE_TOOL)
    
    sys_prompt = await build_system_prompt(state["db"], cat)
    response = await client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "system", "content": sys_prompt}] + state["messages"],
        tools=current_tools if current_tools else None,
        tool_choice="auto" if current_tools else None
    )
    return {"messages": [response.choices[0].message]}

async def call_tools(state: AgentState):
    """Executes tool calls, handles Re-route requests, and updates state."""
    last_msg = state["messages"][-1]
    new_messages = []
    new_sources = []
    new_traces = []
    new_tool_results = []
    
    # Track if any tool call in this batch is a Re-route request
    is_rerouting = False

    # Perform task adherence check (Safety)
    task_risk = await check_task_adherence(TOOLS, state["messages"])
    if task_risk is True:
        logger.warning("Task adherence risk for user=%s", state["user_id"])

    for tool_call in last_msg.tool_calls:
        name = tool_call.function.name
        raw_args = tool_call.function.arguments

        # --- 1. Handle Re-route (Category Change) ---
        if name == "request_category_change":
            is_rerouting = True
            try:
                reroute_args = json.loads(raw_args)
                target_cat = reroute_args.get("target_category", "search")
            except (json.JSONDecodeError, KeyError):
                reroute_args = {"target_category": "search"}
                target_cat = "search"
            result_str = json.dumps({"status": "switching_category", "target": target_cat, "message": f"Switching to {target_cat} tools."})
            args = reroute_args
        
        # --- 2. Handle Standard Tools ---
        else:
            result_str = await execute_tool(state["db"], state["user_id"], tool_call)
            try:
                args = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                args = raw_args

        # Append to tool messages for LLM context
        new_messages.append({
            "role": "tool", 
            "tool_call_id": tool_call.id, 
            "content": result_str
        })

        # --- 3. Build Trace & Metadata ---
        new_traces.append({
            "tool": name,
            "args": args,
            "result_summary": result_str[:500],
            "task_adherence_risk": bool(task_risk)
        })

        # Extract IDs for UI Citations (only for data tools, skip for re-route and web_search)
        if name != "request_category_change" and name != 'web_search':
            try:
                data = json.loads(result_str)
                if isinstance(data, list):
                    for item in data:
                        if "doc_id" in item: new_sources.append(str(item["doc_id"]))
                        elif "id" in item: new_sources.append(str(item["id"]))
                
                new_tool_results.append({
                    "tool": name,
                    "args": args,
                    "data": data if isinstance(data, (list, dict)) else None,
                })
            except (json.JSONDecodeError, TypeError, KeyError):
                pass
        # Extract web URLs for web_search
        if name == "web_search":
            try:
                data = json.loads(result_str)
                if isinstance(data, list):
                    for item in data:
                        # DuckDuckGo uses 'href' for the URL
                        if "href" in item: 
                            new_sources.append(item["href"])
                
                new_tool_results.append({
                    "tool": name,
                    "args": args,
                    "data": data,
                })
            except (json.JSONDecodeError, TypeError, KeyError):
                pass
    # Update state: Increment reroute_count if switching categories
    current_reroutes = state.get("reroute_count", 0)
    result = {
        "messages": new_messages,
        "sources": new_sources,
        "tool_trace": new_traces,
        "tool_results": new_tool_results,
        "reroute_count": current_reroutes,
    }
    if is_rerouting:
        result["reroute_count"] = current_reroutes + 1
        result["selected_category"] = target_cat  # skip router LLM call
    return result


def router(state: AgentState):
    """
    Conditional logic to determine the next node in the graph.
    All tool calls (including reroutes) go through 'tools' node.
    """
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
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
    workflow.add_node("router", route_query)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools)
    workflow.set_entry_point("router")
    workflow.add_edge("router", "agent")
    workflow.add_conditional_edges(
        "agent",
        router,
        {
            "tools": "tools",
            END: END
        }
    )
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
                "tool_trace": [],
                "tool_results": [],
                "selected_category": "none",           # tool category
                "reroute_count": 0,                    # count of re-routes
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
            chart_data = extract_chart_data(output.get("tool_results", []))
            clean_result = {
                "answer": clean_answer,
                "documents": list(set(str(s) for s in output.get("sources", []))),
                "cited_sources": cited_sources,
                "query": question,
                "session_id": None,  # Populated by the calling route handler
                "safety": None,      # UI expects null when everything is safe
                "groundedness": groundedness_info,
                "chart_data": chart_data,
            }

            # Optionally include tool_trace if needed for internal logging/debugging
            # clean_result["tool_trace"] = output.get("tool_trace", [])
            for trace in output.get("tool_trace", []):
                logger.debug(f"Tool Called -> {trace['tool']}")
                logger.debug(f"Arguments   -> {trace['args']}")
                # Using %s or f-strings both work; f-string shown for consistency
                logger.debug(f"Result (first 200 chars) -> {trace['result_summary'][:200]}...")
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
