"""
Evaluation runner for the Contoso Product Expert Agent.

Sends test queries from eval_dataset.json to the live agent, collects
responses, runs all six evaluators, and prints a scored report.

Usage:
    python eval_runner.py                  # run all tests
    python eval_runner.py --category accuracy   # run one category
    python eval_runner.py --id accuracy_tents   # run one test case
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

from evaluators import (
    AccuracyEvaluator,
    GroundingEvaluator,
    CitationEvaluator,
    ContextAwarenessEvaluator,
    MCPApprovalEvaluator,
    ErrorHandlingEvaluator,
)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
load_dotenv()
PROJECT_ENDPOINT = os.getenv("PROJECT_ENDPOINT")
AGENT_NAME = os.getenv("AGENT_NAME")

DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
RESULTS_DIR = Path(__file__).parent / "eval_results"

# ──────────────────────────────────────────────
# Instantiate evaluators
# ──────────────────────────────────────────────
accuracy_eval = AccuracyEvaluator()
grounding_eval = GroundingEvaluator()
citation_eval = CitationEvaluator()
context_eval = ContextAwarenessEvaluator()
mcp_eval = MCPApprovalEvaluator()
error_eval = ErrorHandlingEvaluator()

# ──────────────────────────────────────────────
# Agent interaction helpers
# ──────────────────────────────────────────────

def connect_to_agent():
    """Return (openai_client, agent) using the Foundry SDK."""
    credential = DefaultAzureCredential(
        exclude_environment_credential=True,
        exclude_managed_identity_credential=True,
    )
    project_client = AIProjectClient(
        credential=credential,
        endpoint=PROJECT_ENDPOINT,
    )
    openai_client = project_client.get_openai_client()
    agent = project_client.agents.get(agent_name=AGENT_NAME)
    return openai_client, agent


def create_conversation(openai_client):
    """Create and return a fresh conversation."""
    return openai_client.conversations.create(items=[])


def send_message(openai_client, agent, conversation, user_message, auto_approve: bool = True):
    """
    Send *user_message* in *conversation* and return a dict with:
      response_text, mcp_requested, mcp_approved, error
    """
    result = {
        "response_text": None,
        "mcp_requested": False,
        "mcp_approved": None,
        "error": None,
        "crashed": False,
    }

    try:
        # Add user message
        openai_client.conversations.items.create(
            conversation_id=conversation.id,
            items=[{"type": "message", "role": "user", "content": user_message}],
        )

        # Get response
        response = openai_client.responses.create(
            conversation=conversation.id,
            extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
            input="",
        )

        # Check for MCP approval request
        approval_request = None
        if hasattr(response, "output") and response.output:
            for item in response.output:
                if hasattr(item, "type") and item.type == "mcp_approval_request":
                    approval_request = item
                    break

        if approval_request:
            result["mcp_requested"] = True
            result["mcp_approved"] = auto_approve

            approval_response = {
                "type": "mcp_approval_response",
                "approval_request_id": approval_request.id,
                "approve": auto_approve,
            }

            openai_client.conversations.items.create(
                conversation_id=conversation.id,
                items=[approval_response],
            )

            response = openai_client.responses.create(
                conversation=conversation.id,
                extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
                input="",
            )

        if response and hasattr(response, "output_text") and response.output_text:
            result["response_text"] = response.output_text
        else:
            result["response_text"] = None

    except Exception as exc:
        result["error"] = str(exc)
        result["crashed"] = False  # caught, not crashed

    return result


# ──────────────────────────────────────────────
# Evaluation logic
# ──────────────────────────────────────────────

def evaluate_test_case(test_case: dict, agent_result: dict,
                       prior_response: str | None = None) -> dict:
    """Run all applicable evaluators against one test-case result."""
    response = agent_result["response_text"] or ""
    category = test_case.get("category", "")
    scores: dict = {}

    # Always run error handling
    scores["error_handling"] = error_eval(
        response=agent_result["response_text"],
        error=agent_result["error"],
        crashed=agent_result["crashed"],
    )

    # Accuracy
    if category == "accuracy" or test_case.get("expected_keywords"):
        scores["accuracy"] = accuracy_eval(
            response=response,
            expected_keywords=test_case.get("expected_keywords", []),
            expected_facts=test_case.get("expected_facts"),
        )

    # Grounding
    if category == "grounding" or test_case.get("should_decline") is not None:
        scores["grounding"] = grounding_eval(
            response=response,
            should_decline=test_case.get("should_decline", False),
            grounding_signals=test_case.get("grounding_signals"),
        )

    # Citations
    if category == "citations" or test_case.get("check_citations"):
        scores["citations"] = citation_eval(
            response=response,
            check_citations=test_case.get("check_citations", True),
        )

    # Context awareness
    if category == "context_awareness" or test_case.get("requires_prior_context"):
        scores["context_awareness"] = context_eval(
            response=response,
            requires_prior_context=test_case.get("requires_prior_context", False),
            context_signals=test_case.get("context_signals"),
            prior_response=prior_response,
        )

    # MCP approval flow
    if category == "mcp_approval" or test_case.get("expect_mcp_request"):
        scores["mcp_approval"] = mcp_eval(
            response=response,
            mcp_requested=agent_result["mcp_requested"],
            mcp_approved=agent_result["mcp_approved"],
            expect_mcp_request=test_case.get("expect_mcp_request", False),
            auto_approve_mcp=test_case.get("auto_approve_mcp", True),
            error=agent_result["error"],
        )

    return scores


# ──────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────

SCORE_KEYS = {
    "accuracy": "accuracy_score",
    "grounding": "grounding_score",
    "citations": "citation_score",
    "context_awareness": "context_score",
    "mcp_approval": "mcp_flow_score",
    "error_handling": "error_handling_score",
}


def print_report(all_results: list[dict]):
    """Pretty-print a summary table plus per-test details."""
    width = 90
    print("\n" + "=" * width)
    print("  AGENT EVALUATION REPORT")
    print(f"  {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S UTC}")
    print("=" * width)

    # Aggregate scores per dimension
    dimension_scores: dict[str, list[float]] = {k: [] for k in SCORE_KEYS}

    for r in all_results:
        test_id = r["test_id"]
        category = r["category"]
        scores = r["scores"]
        response_snippet = (r["response_text"] or "")[:120].replace("\n", " ")

        print(f"\n── Test: {test_id}  [{category}] ──")
        print(f"   Query:    {r['query'][:80]}")
        print(f"   Response: {response_snippet}{'…' if len(r.get('response_text') or '') > 120 else ''}")

        if r.get("error"):
            print(f"   Error:    {r['error'][:100]}")

        for dim, score_data in scores.items():
            key = SCORE_KEYS.get(dim)
            if key and key in score_data:
                val = score_data[key]
                dimension_scores[dim].append(val)
                bar = "█" * int(val * 10) + "░" * (10 - int(val * 10))
                details = score_data.get("details", "")
                print(f"   {dim:<22} {bar} {val:.2f}  {details}")

    # Summary
    print("\n" + "=" * width)
    print("  SUMMARY BY DIMENSION")
    print("=" * width)
    overall_scores = []
    for dim, vals in dimension_scores.items():
        if vals:
            avg = sum(vals) / len(vals)
            overall_scores.append(avg)
            bar = "█" * int(avg * 10) + "░" * (10 - int(avg * 10))
            status = "PASS" if avg >= 0.7 else "WARN" if avg >= 0.4 else "FAIL"
            print(f"   {dim:<22} {bar} {avg:.2f}  ({len(vals)} tests)  [{status}]")
        else:
            print(f"   {dim:<22} {'░' * 10}  N/A   (0 tests)")

    if overall_scores:
        total = sum(overall_scores) / len(overall_scores)
        print(f"\n   {'OVERALL':<22} {'█' * int(total * 10) + '░' * (10 - int(total * 10))} {total:.2f}")
        print(f"\n   Result: {'✓ PASS' if total >= 0.7 else '⚠ NEEDS IMPROVEMENT' if total >= 0.4 else '✗ FAIL'}")

    print("=" * width + "\n")


def save_results(all_results: list[dict]):
    """Persist raw results as JSON for later analysis."""
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Raw results saved to {out_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate the Contoso Product Expert Agent")
    parser.add_argument("--category", help="Run only tests in this category")
    parser.add_argument("--id", help="Run only the test with this id")
    args = parser.parse_args()

    # Load dataset
    with open(DATASET_PATH, encoding="utf-8") as f:
        test_cases: list[dict] = json.load(f)

    if args.id:
        test_cases = [tc for tc in test_cases if tc["id"] == args.id]
    elif args.category:
        test_cases = [tc for tc in test_cases if tc["category"] == args.category]

    if not test_cases:
        print("No matching test cases found.")
        sys.exit(1)

    print(f"Loaded {len(test_cases)} test case(s).\n")

    # Connect
    print("Connecting to agent...")
    openai_client, agent = connect_to_agent()
    print(f"Connected to: {agent.name}\n")

    all_results: list[dict] = []
    prior_response: str | None = None  # for context-awareness chain

    # We need separate conversations for independent tests, but a shared
    # conversation for context-awareness chains.
    conversation = create_conversation(openai_client)
    current_chain_conversation = None  # lazily created for context tests

    for i, tc in enumerate(test_cases, 1):
        test_id = tc["id"]
        category = tc.get("category", "unknown")
        query = tc["query"]
        auto_approve = tc.get("auto_approve_mcp", True)

        print(f"[{i}/{len(test_cases)}] Running: {test_id} ({category})")

        # Context-awareness tests share a conversation
        if category == "context_awareness":
            if current_chain_conversation is None or tc.get("is_setup"):
                current_chain_conversation = create_conversation(openai_client)
                prior_response = None
            conv = current_chain_conversation
        else:
            # Each independent test gets a fresh conversation
            conv = create_conversation(openai_client)
            prior_response = None

        # Handle empty-input edge case
        if tc.get("expect_graceful_handling") and not query:
            # Don't send empty string to the API — test that our app would skip it
            agent_result = {
                "response_text": "(skipped – empty input handled by client)",
                "mcp_requested": False,
                "mcp_approved": None,
                "error": None,
                "crashed": False,
            }
        else:
            agent_result = send_message(openai_client, agent, conv, query,
                                        auto_approve=auto_approve)

        # Small delay to avoid throttling
        time.sleep(1)

        # Evaluate
        scores = evaluate_test_case(tc, agent_result, prior_response=prior_response)

        # Track prior response for context-awareness chains
        if category == "context_awareness":
            prior_response = agent_result["response_text"]

        all_results.append({
            "test_id": test_id,
            "category": category,
            "query": query,
            "response_text": agent_result["response_text"],
            "mcp_requested": agent_result["mcp_requested"],
            "mcp_approved": agent_result["mcp_approved"],
            "error": agent_result["error"],
            "scores": scores,
        })

    # Report
    print_report(all_results)
    save_results(all_results)


if __name__ == "__main__":
    main()
