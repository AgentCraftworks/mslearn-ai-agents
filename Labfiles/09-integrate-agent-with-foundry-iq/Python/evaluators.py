"""
Custom evaluators for the Contoso Product Expert Agent.

Covers six evaluation dimensions:
  1. MCP Approval Flow
  2. Accuracy
  3. Citations
  4. Context Awareness
  5. Grounding
  6. Error Handling
"""

import json
import re


class AccuracyEvaluator:
    """
    Checks that the agent response contains expected keywords and factual
    statements drawn directly from the knowledge base documents.
    """

    def __call__(self, *, response: str, expected_keywords: list[str],
                 expected_facts: list[str] | None = None, **kwargs) -> dict:
        if not response:
            return {"accuracy_score": 0.0, "keyword_hits": [], "keyword_misses": expected_keywords,
                    "fact_results": {}, "details": "Empty response"}

        response_lower = response.lower()

        # --- keyword check ---
        hits = [kw for kw in expected_keywords if kw.lower() in response_lower]
        misses = [kw for kw in expected_keywords if kw.lower() not in response_lower]
        keyword_score = len(hits) / len(expected_keywords) if expected_keywords else 1.0

        # --- fact check (fuzzy substring) ---
        fact_results: dict[str, bool] = {}
        if expected_facts:
            for fact in expected_facts:
                # Check if key numeric / named entities from the fact appear
                tokens = re.findall(r'\$[\d,.]+|\b[A-Z][\w-]+\b|\d{2,}', fact)
                matched = sum(1 for t in tokens if t.lower() in response_lower)
                fact_results[fact] = matched >= max(1, len(tokens) // 2)
            fact_score = sum(fact_results.values()) / len(fact_results)
        else:
            fact_score = 1.0

        combined = round((keyword_score * 0.5 + fact_score * 0.5), 2)

        return {
            "accuracy_score": combined,
            "keyword_hits": hits,
            "keyword_misses": misses,
            "fact_results": fact_results,
        }


class GroundingEvaluator:
    """
    Verifies the agent stays grounded:
      - For on-topic queries, responses should NOT contain hedging/refusal phrases.
      - For off-topic or unknown-product queries, the agent SHOULD refuse or
        indicate the info is not in the knowledge base.
    """

    REFUSAL_SIGNALS = [
        "i don't have", "no information", "not found", "outside",
        "knowledge base", "not available", "cannot find", "don't find",
        "i'm not able", "no relevant", "not related",
    ]

    def __call__(self, *, response: str, should_decline: bool = False,
                 grounding_signals: list[str] | None = None, **kwargs) -> dict:
        if not response:
            return {"grounding_score": 0.0, "grounded": False,
                    "details": "Empty response"}

        response_lower = response.lower()
        signals = [s.lower() for s in (grounding_signals or self.REFUSAL_SIGNALS)]
        found_signals = [s for s in signals if s in response_lower]

        if should_decline:
            # Agent SHOULD indicate it can't answer
            score = 1.0 if found_signals else 0.0
            return {
                "grounding_score": score,
                "grounded": bool(found_signals),
                "found_signals": found_signals,
                "details": ("Agent correctly declined off-topic query"
                            if found_signals else "Agent answered an off-topic query it should have declined"),
            }
        else:
            # Agent should NOT be refusing on-topic queries
            score = 0.0 if found_signals else 1.0
            return {
                "grounding_score": score,
                "grounded": not bool(found_signals),
                "found_signals": found_signals,
                "details": ("Agent provided a grounded on-topic answer"
                            if not found_signals else "Agent incorrectly refused an on-topic query"),
            }


class CitationEvaluator:
    """
    Checks whether the response includes source references, document names,
    SKUs, or structured citation markers.
    """

    CITATION_PATTERNS = [
        r"SKU[:\s]*[\w-]+",          # SKU references
        r"\bsource[s]?\b",           # "source" / "sources"
        r"\bcited?\b",               # "cite" / "cited"
        r"\breference[s]?\b",        # "reference(s)"
        r"\bdocument\b",             # "document"
        r"contoso[\w\s-]*\.(pdf|md|txt)",  # file names
        r"\[.*?\]\(.*?\)",           # markdown links
    ]

    def __call__(self, *, response: str, check_citations: bool = True,
                 **kwargs) -> dict:
        if not response:
            return {"citation_score": 0.0, "citations_found": [],
                    "details": "Empty response"}

        if not check_citations:
            return {"citation_score": 1.0, "citations_found": [],
                    "details": "Citation check skipped"}

        found: list[str] = []
        for pattern in self.CITATION_PATTERNS:
            matches = re.findall(pattern, response, re.IGNORECASE)
            found.extend(matches)

        score = min(1.0, len(found) / 1)  # at least 1 citation → full score
        return {
            "citation_score": score,
            "citations_found": found[:10],
            "details": (f"Found {len(found)} citation indicator(s)"
                        if found else "No citation indicators found"),
        }


class ContextAwarenessEvaluator:
    """
    For follow-up queries that depend on prior conversation context, checks
    that the response references elements from the earlier exchange.
    """

    def __call__(self, *, response: str, requires_prior_context: bool = False,
                 context_signals: list[str] | None = None,
                 prior_response: str | None = None, **kwargs) -> dict:
        if not requires_prior_context:
            return {"context_score": 1.0, "details": "No prior context required"}

        if not response:
            return {"context_score": 0.0, "details": "Empty response"}

        response_lower = response.lower()
        signals = [s.lower() for s in (context_signals or [])]
        found = [s for s in signals if s in response_lower]
        score = len(found) / len(signals) if signals else 0.0

        # Bonus: check overlap with prior response content
        overlap_score = 0.0
        if prior_response:
            prior_tokens = set(re.findall(r'\b\w{4,}\b', prior_response.lower()))
            resp_tokens = set(re.findall(r'\b\w{4,}\b', response_lower))
            common = prior_tokens & resp_tokens
            overlap_score = min(1.0, len(common) / 5) if prior_tokens else 0.0

        combined = round(max(score, overlap_score), 2)
        return {
            "context_score": combined,
            "found_signals": found,
            "overlap_score": overlap_score,
            "details": (f"Context signals found: {found}" if found
                        else "No context signals found in response"),
        }


class MCPApprovalEvaluator:
    """
    Validates MCP approval flow behaviour:
      - When auto_approve_mcp=True, the agent should eventually return content.
      - When auto_approve_mcp=False (deny), the agent should handle the denial
        gracefully (no crash, degraded-but-polite response).
      - If expect_mcp_request=True, we check that an approval WAS requested.
    """

    def __call__(self, *, response: str, mcp_requested: bool = False,
                 mcp_approved: bool | None = None,
                 expect_mcp_request: bool = False, auto_approve_mcp: bool = True,
                 error: str | None = None, **kwargs) -> dict:

        results: dict = {
            "mcp_flow_score": 0.0,
            "mcp_requested": mcp_requested,
            "mcp_approved": mcp_approved,
            "details": "",
        }

        if error:
            results["details"] = f"Error during MCP flow: {error}"
            return results

        if expect_mcp_request and not mcp_requested:
            results["details"] = "Expected MCP approval request but none was triggered"
            return results

        if not auto_approve_mcp:
            # Denial path – agent should still respond without crashing
            if response and "error" not in response.lower():
                results["mcp_flow_score"] = 1.0
                results["details"] = "Agent handled MCP denial gracefully"
            else:
                results["mcp_flow_score"] = 0.5
                results["details"] = "Agent responded after denial but may have errors"
            return results

        # Approval path – should produce substantive content
        if response and len(response.strip()) > 20:
            results["mcp_flow_score"] = 1.0
            results["details"] = "MCP approval flow completed successfully"
        elif response:
            results["mcp_flow_score"] = 0.5
            results["details"] = "Response received but suspiciously short"
        else:
            results["details"] = "No response after MCP approval"

        return results


class ErrorHandlingEvaluator:
    """
    Checks that the application handles edge cases gracefully:
      - Empty inputs
      - Malformed requests
      - Connection or API errors
    The application should not crash and should return a meaningful message.
    """

    def __call__(self, *, response: str | None, error: str | None = None,
                 crashed: bool = False, **kwargs) -> dict:
        if crashed:
            return {"error_handling_score": 0.0,
                    "details": f"Application crashed: {error}"}

        if error and response is None:
            # Error was raised but caught gracefully
            return {"error_handling_score": 0.5,
                    "details": f"Error caught gracefully: {error}"}

        if response is not None:
            return {"error_handling_score": 1.0,
                    "details": "No errors encountered"}

        return {"error_handling_score": 0.5,
                "details": "No response and no error — unclear outcome"}
