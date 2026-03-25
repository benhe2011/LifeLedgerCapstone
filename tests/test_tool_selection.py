"""
Golden dataset test for agent tool selection.

Tests that the agent picks the correct tool for each query
by hitting the /ask endpoint and inspecting tool_trace.

SETUP REQUIRED:
1. In app/agent.py line ~532, uncomment:
       clean_result["tool_trace"] = output.get("tool_trace", [])
2. In app/main.py, add tool_trace to AskResponse:
       tool_trace: Optional[List[dict]] = None
   And in the /ask endpoint, add:
       tool_trace=result.get("tool_trace")
3. Server must be running with DEV_MODE=true

Usage:
    python tests/test_tool_selection.py
    python tests/test_tool_selection.py --base-url http://localhost:8000
    python tests/test_tool_selection.py --verbose
"""

import argparse
import json
import sys
import requests

# --- Golden Dataset ---
# Each entry: (query, expected_tool)

GOLDEN_DATASET = [
    # === get_total_spending ===
    ("How much have I spent in total?", "get_total_spending"),
    ("What's my total spending?", "get_total_spending"),
    ("How much money have I spent all time?", "get_total_spending"),
    ("What's my grand total?", "get_total_spending"),
    ("How much did I spend in 2025?", "get_total_spending"),
    ("How much have I spent over the past 10 years?", "get_total_spending"),
    ("What's the total damage to my wallet?", "get_total_spending"),
    ("Give me one number for all my spending", "get_total_spending"),

    # === get_spending_by_merchant ===
    ("Where do I spend the most?", "get_spending_by_merchant"),
    ("Show my spending breakdown", "get_spending_by_merchant"),
    ("Which stores do I shop at the most?", "get_spending_by_merchant"),
    ("What's my spending by store?", "get_spending_by_merchant"),
    ("Top 5 places I spend money", "get_spending_by_merchant"),
    ("Rank my merchants by how much I spend", "get_spending_by_merchant"),
    ("What stores am I loyal to?", "get_spending_by_merchant"),

    # === get_receipts_by_merchant ===
    ("Show me my Target receipts", "get_receipts_by_merchant"),
    ("What have I bought at Walmart?", "get_receipts_by_merchant"),
    ("How many times did I go to Starbucks?", "get_receipts_by_merchant"),
    ("List all my Amazon purchases", "get_receipts_by_merchant"),
    ("Show me everything from Costco", "get_receipts_by_merchant"),
    ("How much have I spent at Costco the last year?", "get_receipts_by_merchant"),
    ("Pull up my Trader Joe's history", "get_receipts_by_merchant"),
    ("When was the last time I went to Home Depot?", "get_receipts_by_merchant"),

    # === get_receipts_by_date_range ===
    ("What did I buy last month?", "get_receipts_by_date_range"),
    ("Show my receipts from January", "get_receipts_by_date_range"),
    ("What's my spending over the past 3 months?", "get_receipts_by_date_range"),
    ("Show me purchases from March 2025 to June 2025", "get_receipts_by_date_range"),
    ("What are my recent receipts?", "get_receipts_by_date_range"),
    ("How much have I spent this year compared to last year?", "get_receipts_by_date_range"),
    ("Is my spending going up or down?", "get_receipts_by_date_range"),
    ("What did I buy last week?", "get_receipts_by_date_range"),
    ("Show me everything from the holidays", "get_receipts_by_date_range"),

    # === get_recurring_costs ===
    ("What are my subscriptions?", "get_recurring_costs"),
    ("Do I have any recurring charges?", "get_recurring_costs"),
    ("What bills do I pay monthly?", "get_recurring_costs"),
    ("Show me my recurring costs", "get_recurring_costs"),
    ("What subscriptions am I paying for?", "get_recurring_costs"),
    ("What am I wasting money on?", "get_recurring_costs"),
    ("Where is my money going every month?", "get_recurring_costs"),
    ("Am I paying for anything I forgot about?", "get_recurring_costs"),
    ("What can I cancel to save money?", "get_recurring_costs"),

    # === get_trips ===
    ("Show me my trips", "get_trips"),
    ("What travel expenses do I have?", "get_trips"),
    ("Have I taken any trips recently?", "get_trips"),
    ("Show my travel history", "get_trips"),
    ("What did my last vacation cost?", "get_trips"),
    ("How much did I spend on my trip to New York?", "get_trips"),
    ("List my flights and hotels", "get_trips"),

    # === get_all_receipt_texts ===
    ("What unhealthy items did I buy?", "get_all_receipt_texts"),
    ("Show me all electronics I've purchased", "get_all_receipt_texts"),
    ("What items do I buy most frequently?", "get_all_receipt_texts"),
    ("Did I buy milk recently?", "get_all_receipt_texts"),
    ("What groceries have I been buying?", "get_all_receipt_texts"),
    ("What's the most expensive single item I bought?", "get_all_receipt_texts"),
    ("Do any of my receipts mention a warranty?", "get_all_receipt_texts"),
    ("How many energy drinks have I purchased?", "get_all_receipt_texts"),
    ("What did I actually get at Costco last time?", "get_all_receipt_texts"),

    # === get_document_overview ===
    ("What documents do I have?", "get_document_overview"),
    ("How many receipts have I uploaded?", "get_document_overview"),
    ("What types of documents are in my account?", "get_document_overview"),
    ("Give me an overview of my uploads", "get_document_overview"),
    ("How many documents do I have?", "get_document_overview"),
    ("What's in my documents?", "get_document_overview"),
    ("What does my receipt from the photo store say?", "get_document_overview"),
    ("Summarize what I've uploaded", "get_document_overview"),
    ("Do I have more receipts or other documents?", "get_document_overview"),

    # === get_all_document_texts ===
    ("What does that flyer say?", "get_all_document_texts"),
    ("Show me my notes", "get_all_document_texts"),
    ("What's in my uploaded screenshots?", "get_all_document_texts"),
    ("Do any of my documents mention a warranty?", "get_all_document_texts"),
    ("Search all my documents for a phone number", "get_all_document_texts"),
    ("What non-receipt documents do I have?", "get_all_document_texts"),
    ("Read me the text from that poster I uploaded", "get_all_document_texts"),
    ("What did that email I uploaded say?", "get_all_document_texts"),

    # === get_earnings_summary ===
    ("how much did I earn last month", "get_earnings_summary"),
    ("show my income over time", "get_earnings_summary"),
    ("what was my last paycheck", "get_earnings_summary"),

    # === get_deductions_breakdown ===
    ("how much tax did I pay this year", "get_deductions_breakdown"),
    ("show my deductions breakdown", "get_deductions_breakdown"),
    ("total withholdings", "get_deductions_breakdown"),
    ("how much went to 401k", "get_deductions_breakdown"),

    # === get_income_vs_spending ===
    ("am I saving money", "get_income_vs_spending"),
    ("income vs expenses", "get_income_vs_spending"),
    ("what's my net savings", "get_income_vs_spending"),
    ("financial overview", "get_income_vs_spending"),

    # === get_recurring_income ===
    ("how often do I get paid", "get_recurring_income"),
    ("what's my salary", "get_recurring_income"),

    # === get_lease_details ===
    ("what's my rent", "get_lease_details"),
    ("when does my lease end", "get_lease_details"),
    ("how much is my security deposit", "get_lease_details"),
    ("show my rental agreement", "get_lease_details"),

    # === search_documents ===
    ("Find documents about insurance", "search_documents"),
    ("Do I have anything about a rental agreement?", "search_documents"),
    ("Search for anything mentioning refund", "search_documents"),
    ("Find my tax-related documents", "search_documents"),
    ("Do I have a receipt for that blue jacket?", "search_documents"),
    ("Which document talks about my lease?", "search_documents"),
    ("Find the receipt where I bought a birthday gift", "search_documents"),
    ("Do I have anything related to my car?", "search_documents"),
]


def ask(base_url: str, token: str, question: str) -> dict:
    """Hit the /ask endpoint and return the response."""
    resp = requests.post(
        f"{base_url}/ask",
        json={"question": question},
        headers={"Authorization": f"Bearer {token}"},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def get_first_tool(response: dict) -> str | None:
    """Extract the first tool name from the tool_trace."""
    trace = response.get("tool_trace") or []
    if not trace:
        return None
    return trace[0].get("tool")


def run(base_url: str, token: str, verbose: bool = False):
    """Run all golden dataset queries and report results."""
    passed = 0
    failed = 0
    errors = 0
    results = []

    total = len(GOLDEN_DATASET)
    for i, (query, expected_tool) in enumerate(GOLDEN_DATASET, 1):
        print(f"[{i}/{total}] {query}")
        try:
            response = ask(base_url, token, query)
            actual_tool = get_first_tool(response)

            if actual_tool == expected_tool:
                status = "PASS"
                passed += 1
            else:
                status = "FAIL"
                failed += 1

            results.append({
                "query": query,
                "expected": expected_tool,
                "actual": actual_tool,
                "status": status,
            })

            icon = "✓" if status == "PASS" else "✗"
            print(f"  {icon} expected={expected_tool}, got={actual_tool}")

            if verbose and status == "FAIL":
                print(f"    answer preview: {response.get('answer', '')[:120]}...")

        except Exception as e:
            errors += 1
            results.append({
                "query": query,
                "expected": expected_tool,
                "actual": None,
                "status": "ERROR",
            })
            print(f"  ERROR: {e}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed, {errors} errors / {total} total")
    print(f"Accuracy: {passed / total * 100:.1f}%")

    # Breakdown by tool
    print("\nPer-tool accuracy:")
    tools = sorted(set(r["expected"] for r in results))
    for tool in tools:
        tool_results = [r for r in results if r["expected"] == tool]
        tool_pass = sum(1 for r in tool_results if r["status"] == "PASS")
        print(f"  {tool}: {tool_pass}/{len(tool_results)}")

    # Show failures
    failures = [r for r in results if r["status"] != "PASS"]
    if failures:
        print("\nFailed queries:")
        for r in failures:
            print(f"  \"{r['query']}\"")
            print(f"    expected: {r['expected']}, got: {r['actual']}")

    # Save full results to JSON
    with open("tests/tool_selection_results.json", "w") as f:
        json.dump({
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "accuracy": round(passed / total * 100, 1),
            },
            "results": results,
        }, f, indent=2)
    print("\nFull results saved to tests/tool_selection_results.json")

    return failed == 0 and errors == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test agent tool selection against golden dataset")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--token", default="dev_testuser", help="Auth token (default: dev_testuser for DEV_MODE)")
    parser.add_argument("--verbose", action="store_true", help="Show answer previews for failures")
    args = parser.parse_args()

    success = run(args.base_url, args.token, args.verbose)
    sys.exit(0 if success else 1)
