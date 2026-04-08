"""Tests for the FastAPI HTTP endpoints.

Note: The openenv-core HTTP /reset and /step endpoints are stateless (each
creates a new env instance), so multi-step interactions are tested via the
environment directly in test_environment.py.  These tests cover:
  - /health (openenv-core)
  - /reset response format (openenv-core, single-call)
  - /tasks, /baseline, /grader (custom endpoints)
"""

import pytest
from fastapi.testclient import TestClient

EASY_FINDINGS = [
    {
        "error_id": "multiple_testing_violation",
        "severity": "critical",
        "location": "Results section",
        "description": (
            "Five metrics were tested and only the best p-value reported without "
            "multiple comparison correction. Family-wise false positive rate inflates "
            "from 5% to ~23%."
        ),
        "impact": "The p=0.041 result is unreliable; false positive rate is ~23%, not 5%.",
        "correction": (
            "Apply Bonferroni correction (alpha = 0.05/5 = 0.01) or pre-specify "
            "a single primary metric."
        ),
        "confidence": 0.95,
    },
    {
        "error_id": "early_stopping_peeking",
        "severity": "major",
        "location": "Methodology section",
        "description": (
            "Test was stopped on Day 5 when significance was first observed. "
            "Early stopping without pre-specified stopping rules inflates false positive rate."
        ),
        "impact": "Optional stopping makes the significance level unreliable.",
        "correction": (
            "Pre-specify required sample size. Use sequential testing methods "
            "such as SPRT or O'Brien-Fleming boundaries."
        ),
        "confidence": 0.88,
    },
]


class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"


class TestTasks:
    def test_tasks_returns_all_tasks(self, client):
        resp = client.get("/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_tasks"] == 9
        assert len(data["tasks"]) == 9

    def test_tasks_includes_schemas(self, client):
        resp = client.get("/tasks")
        data = resp.json()
        assert "action_schema" in data
        assert "observation_schema" in data

    def test_task_has_required_fields(self, client):
        resp = client.get("/tasks")
        for task in resp.json()["tasks"]:
            assert "task_id" in task
            assert "difficulty" in task
            assert "domain" in task
            assert "num_errors" in task

    def test_task_difficulties(self, client):
        resp = client.get("/tasks")
        difficulties = {t["difficulty"] for t in resp.json()["tasks"]}
        assert difficulties == {"easy", "medium", "hard", "very_hard"}

    def test_task_domains(self, client):
        resp = client.get("/tasks")
        domains = {t["domain"] for t in resp.json()["tasks"]}
        assert domains == {"ab_testing", "regression", "causal_inference"}


class TestReset:
    def test_reset_with_valid_task(self, client):
        resp = client.post("/reset", json={"task_id": "ab_testing_easy"})
        assert resp.status_code == 200
        data = resp.json()
        assert "observation" in data
        obs = data["observation"]
        assert "report_text" in obs
        assert obs["step_count"] == 0
        assert data["done"] is False

    def test_reset_without_task_id(self, client):
        resp = client.post("/reset", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "observation" in data
        assert "report_text" in data["observation"]

    def test_reset_with_invalid_task_returns_error(self, client):
        resp = client.post("/reset", json={"task_id": "invalid_task_xyz"})
        assert resp.status_code == 500

    def test_reset_report_text_not_empty(self, client):
        resp = client.post("/reset", json={"task_id": "regression_easy"})
        obs = resp.json()["observation"]
        assert len(obs["report_text"]) > 100


class TestBaseline:
    def test_baseline_keyword_only(self, client):
        resp = client.get("/baseline", params={"agents": "keyword"})
        assert resp.status_code == 200
        data = resp.json()
        assert "keyword_scanner" in data["baselines"]
        assert data["tasks_evaluated"] == 9
        kb = data["baselines"]["keyword_scanner"]
        assert "scores_by_task" in kb
        assert "average_score" in kb
        assert 0.0 <= kb["average_score"] <= 1.0

    def test_baseline_all_tasks_scored(self, client):
        resp = client.get("/baseline", params={"agents": "keyword"})
        scores = resp.json()["baselines"]["keyword_scanner"]["scores_by_task"]
        expected_tasks = [
            "ab_testing_easy", "ab_testing_medium", "ab_testing_hard",
            "regression_easy", "regression_medium", "regression_hard",
            "causal_inference_medium", "causal_inference_hard", "causal_inference_very_hard",
        ]
        for tid in expected_tasks:
            assert tid in scores
            assert 0.0 <= scores[tid] <= 1.0
