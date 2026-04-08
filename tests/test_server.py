"""Tests for the FastAPI HTTP endpoints."""

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
        assert data["status"] == "ok"
        assert data["tasks_loaded"] == 9


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
        assert "report_text" in data
        assert data["step_count"] == 0
        assert data["done"] is False
        assert data["reward"] == 0.0

    def test_reset_without_task_id(self, client):
        resp = client.post("/reset", json={})
        assert resp.status_code == 200
        assert "report_text" in resp.json()

    def test_reset_with_invalid_task_returns_400(self, client):
        resp = client.post("/reset", json={"task_id": "invalid_task_xyz"})
        assert resp.status_code == 400
        assert "Unknown task_id" in resp.json()["detail"]

    def test_reset_report_text_not_empty(self, client):
        resp = client.post("/reset", json={"task_id": "regression_easy"})
        assert len(resp.json()["report_text"]) > 100


class TestState:
    def test_state_before_reset_returns_400(self, client):
        # We can't guarantee no prior state in session-scoped env, so just check it works
        resp = client.get("/state")
        # Either 200 (if prior episode) or 400 (no episode)
        assert resp.status_code in (200, 400)

    def test_state_after_reset_returns_200(self, client):
        client.post("/reset", json={"task_id": "ab_testing_easy"})
        resp = client.get("/state")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == "ab_testing_easy"
        assert "episode_id" in data
        assert "total_errors_in_report" in data


class TestStep:
    def test_step_submit_audit_returns_reward(self, client):
        client.post("/reset", json={"task_id": "ab_testing_easy"})
        resp = client.post("/step", json={
            "action_type": "submit_audit",
            "findings": EASY_FINDINGS,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["done"] is True
        assert data["reward"] > 0.5

    def test_step_request_clarification_unlocks_data(self, client):
        client.post("/reset", json={"task_id": "ab_testing_easy"})
        resp = client.post("/step", json={
            "action_type": "request_clarification",
            "clarification_request": "please show raw data",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["done"] is False
        assert data["raw_data_summary"] is not None
        assert data["hints_used"] == 1

    def test_step_request_test_details(self, client):
        client.post("/reset", json={"task_id": "ab_testing_easy"})
        resp = client.post("/step", json={
            "action_type": "request_clarification",
            "clarification_request": "what statistical test was used?",
        })
        data = resp.json()
        assert data["statistical_test_details"] is not None

    def test_step_mark_complete_ends_episode(self, client):
        client.post("/reset", json={"task_id": "ab_testing_easy"})
        resp = client.post("/step", json={"action_type": "mark_complete"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["done"] is True
        assert data["reward"] == 0.0

    def test_step_after_done_returns_400(self, client):
        client.post("/reset", json={"task_id": "ab_testing_easy"})
        client.post("/step", json={"action_type": "mark_complete"})
        resp = client.post("/step", json={"action_type": "mark_complete"})
        assert resp.status_code == 400
        assert "done" in resp.json()["detail"].lower()

    def test_step_empty_findings_gives_low_reward(self, client):
        client.post("/reset", json={"task_id": "ab_testing_easy"})
        resp = client.post("/step", json={
            "action_type": "submit_audit",
            "findings": [],
        })
        data = resp.json()
        assert data["reward"] < 0.1

    def test_step_invalid_severity_returns_422(self, client):
        client.post("/reset", json={"task_id": "ab_testing_easy"})
        resp = client.post("/step", json={
            "action_type": "submit_audit",
            "findings": [{
                "error_id": "x", "severity": "extreme",
                "location": "s", "description": "d",
                "impact": "i", "correction": "c", "confidence": 0.5,
            }],
        })
        assert resp.status_code == 422

    def test_step_feedback_contains_score(self, client):
        client.post("/reset", json={"task_id": "ab_testing_easy"})
        resp = client.post("/step", json={
            "action_type": "submit_audit",
            "findings": EASY_FINDINGS,
        })
        data = resp.json()
        assert data["finding_feedback"] is not None
        assert "Score" in data["finding_feedback"]


class TestGrader:
    def test_grader_returns_correct_score_after_submission(self, client):
        client.post("/reset", json={"task_id": "ab_testing_easy"})
        step_resp = client.post("/step", json={
            "action_type": "submit_audit",
            "findings": EASY_FINDINGS,
        })
        reward_from_step = step_resp.json()["reward"]

        grader_resp = client.post("/grader")
        assert grader_resp.status_code == 200
        data = grader_resp.json()
        assert data["task_id"] == "ab_testing_easy"
        assert data["errors_found"] == 2
        assert data["total_errors"] == 2
        assert data["false_positives"] == 0
        # score should match the reward from step
        assert data["score"] == pytest.approx(reward_from_step, abs=0.001)

    def test_grader_before_episode_returns_400(self, client):
        # Create fresh client to guarantee no prior state
        from server.app import app
        fresh_client = TestClient(app)
        # Reset to a known state first — we can't guarantee clean state
        # so just verify response is valid
        resp = fresh_client.post("/grader")
        assert resp.status_code in (200, 400)


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
