"""LangGraph agent with tools for document analysis."""
from typing import Dict, Any, List, TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor

from app.db import search_documents, get_document_by_id
from app.extraction import get_user_receipts
from app.vlm_client import ask_vlm


class AgentState(TypedDict):
    """State for the agent graph."""
    question: str
    user_id: str
    db: Any
    messages: Annotated[List[Dict], operator.add]
    answer: str
    sources: List[int]


# Tool definitions
async def search_docs_tool(state: AgentState) -> Dict[str, Any]:
    """Search user documents."""
    results = await search_documents(
        state["db"],
        state["user_id"],
        state["question"],
        limit=5,
    )
    return {
        "results": results,
        "doc_ids": [r["id"] for r in results],
    }


async def query_receipts_tool(state: AgentState) -> Dict[str, Any]:
    """Query receipt extractions."""
    receipts = await get_user_receipts(state["db"], state["user_id"])
    return {"receipts": receipts}


async def analyze_with_vlm(state: AgentState, context: str) -> str:
    """Use VLM to analyze and answer the question."""
    return await ask_vlm(None, state["question"], context)


# Graph nodes
async def search_node(state: AgentState) -> Dict[str, Any]:
    """Search for relevant documents."""
    results = await search_documents(
        state["db"],
        state["user_id"],
        state["question"],
        limit=5,
    )

    context_parts = []
    doc_ids = []

    for doc in results:
        if doc.get("doc_text"):
            context_parts.append(f"Document {doc['id']} ({doc['doc_type']}):\n{doc['doc_text'][:500]}")
            doc_ids.append(doc["id"])

    return {
        "messages": [{"role": "system", "content": f"Found {len(results)} relevant documents"}],
        "sources": doc_ids,
        "_context": "\n\n".join(context_parts),
    }


async def reason_node(state: AgentState) -> Dict[str, Any]:
    """Reason about the question using VLM."""
    context = state.get("_context", "")

    if not context:
        answer = "I couldn't find any relevant documents to answer your question."
    else:
        answer = await ask_vlm(None, state["question"], context)

    return {
        "answer": answer,
        "messages": [{"role": "assistant", "content": answer}],
    }


def should_continue(state: AgentState) -> str:
    """Determine if we should continue to reasoning."""
    if state.get("sources"):
        return "reason"
    return "end"


# Build the graph
def create_agent_graph():
    """Create the LangGraph agent."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("search", search_node)
    workflow.add_node("reason", reason_node)

    # Set entry point
    workflow.set_entry_point("search")

    # Add edges
    workflow.add_conditional_edges(
        "search",
        should_continue,
        {
            "reason": "reason",
            "end": END,
        },
    )
    workflow.add_edge("reason", END)

    return workflow.compile()


# Cached agent
_agent = None


def get_agent():
    """Get or create the agent."""
    global _agent
    if _agent is None:
        _agent = create_agent_graph()
    return _agent


async def ask_agent(db, user_id: str, question: str) -> Dict[str, Any]:
    """Ask the agent a question."""
    agent = get_agent()

    initial_state = {
        "question": question,
        "user_id": user_id,
        "db": db,
        "messages": [],
        "answer": "",
        "sources": [],
    }

    result = await agent.ainvoke(initial_state)

    return {
        "answer": result.get("answer", "Unable to process your question."),
        "sources": result.get("sources", []),
    }
