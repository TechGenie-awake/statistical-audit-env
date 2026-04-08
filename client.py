"""
StatAudit HTTP Client

A simple Python client for the StatAudit OpenEnv environment.

Usage:
    from client import StatAuditClient, Finding

    client = StatAuditClient("http://localhost:8000")

    obs = client.reset("ab_testing_easy")
    print(obs["report_text"])

    result = client.step({
        "action_type": "submit_audit",
        "findings": [
            {
                "error_id": "multiple_testing_violation",
                "severity": "critical",
                "location": "Results section",
                "description": "5 metrics tested without correction",
                "impact": "False positive rate inflated to ~23%",
                "correction": "Apply Bonferroni correction",
                "confidence": 0.9
            }
        ]
    })
    print(f"Reward: {result['reward']:.3f}")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx


class StatAuditClient:
    """HTTP client for the StatAudit OpenEnv environment."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 60):
        self._http = httpx.Client(base_url=base_url, timeout=timeout)

    # ------------------------------------------------------------------
    # Standard OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Start a new episode.

        Args:
            task_id: One of the 9 task IDs, or None for a random task.

        Returns:
            Observation dict containing report_text, report_metadata, etc.
        """
        payload = {"task_id": task_id} if task_id else {}
        resp = self._http.post("/reset", json=payload)
        resp.raise_for_status()
        return resp.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an action in the environment.

        Args:
            action: Dict with keys:
                - action_type: "submit_audit" | "request_clarification" | "mark_complete"
                - findings: list of finding dicts (for submit_audit)
                - clarification_request: str (for request_clarification)

        Returns:
            Observation dict with reward, done, finding_feedback, etc.
        """
        resp = self._http.post("/step", json=action)
        resp.raise_for_status()
        return resp.json()

    def state(self) -> Dict[str, Any]:
        """Return current episode state."""
        resp = self._http.get("/state")
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Extra endpoints
    # ------------------------------------------------------------------

    def tasks(self) -> Dict[str, Any]:
        """List all 9 tasks with their schemas."""
        resp = self._http.get("/tasks")
        resp.raise_for_status()
        return resp.json()

    def grader(self) -> Dict[str, Any]:
        """Get grader score for the current episode."""
        resp = self._http.post("/grader")
        resp.raise_for_status()
        return resp.json()

    def health(self) -> Dict[str, Any]:
        """Check server health."""
        resp = self._http.get("/health")
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        """Close the HTTP connection."""
        self._http.close()

    def __enter__(self) -> "StatAuditClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ------------------------------------------------------------------
# Convenience helpers
# ------------------------------------------------------------------

def make_finding(
    error_id: str,
    severity: str,
    location: str,
    description: str,
    impact: str,
    correction: str,
    confidence: float = 0.8,
) -> Dict[str, Any]:
    """Build a finding dict."""
    return {
        "error_id": error_id,
        "severity": severity,
        "location": location,
        "description": description,
        "impact": impact,
        "correction": correction,
        "confidence": confidence,
    }


def submit_audit(
    client: StatAuditClient,
    findings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Submit an audit and return the graded result."""
    return client.step({"action_type": "submit_audit", "findings": findings})


def request_clarification(
    client: StatAuditClient,
    request: str,
) -> Dict[str, Any]:
    """Request additional context (data summary or test details)."""
    return client.step({
        "action_type": "request_clarification",
        "clarification_request": request,
    })


if __name__ == "__main__":
    # Quick smoke test
    with StatAuditClient() as client:
        print("Health:", client.health())
        tasks = client.tasks()
        print(f"Tasks available: {tasks['total_tasks']}")

        obs = client.reset("ab_testing_easy")
        print(f"Task: ab_testing_easy")
        print(f"Report preview: {obs['report_text'][:120]}...")

        result = submit_audit(client, [
            make_finding(
                error_id="multiple_testing_violation",
                severity="critical",
                location="Results section",
                description="5 metrics were tested and only the best p-value was reported without multiple comparison correction.",
                impact="Family-wise false positive rate inflates from 5% to ~23%.",
                correction="Apply Bonferroni correction (alpha=0.05/5=0.01) or pre-specify one primary metric.",
                confidence=0.95,
            )
        ])
        print(f"Reward: {result['reward']:.3f}")
        print(f"Feedback: {result['finding_feedback'][:200]}")
