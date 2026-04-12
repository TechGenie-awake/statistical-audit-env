"""Tests for the StatAuditEnvironment."""

import pytest
from models import StatAuditAction, Finding
from server.environment import StatAuditEnvironment

EASY_TASK = "ab_testing_easy"
ALL_TASK_IDS = [
    "ab_testing_easy", "ab_testing_medium", "ab_testing_hard",
    "regression_easy", "regression_medium", "regression_hard",
    "causal_inference_medium", "causal_inference_hard", "causal_inference_very_hard",
]


class TestEnvironmentInit:
    def test_loads_all_tasks(self, env):
        assert len(env.tasks) == 9

    def test_all_expected_task_ids_present(self, env):
        for tid in ALL_TASK_IDS:
            assert tid in env.tasks, f"Missing task: {tid}"

    def test_initial_state_is_empty(self, env):
        # Before reset, state returns an empty default StatAuditState
        state = env.state
        assert state.episode_id == ""
        assert state.task_id == ""


class TestReset:
    def test_reset_with_valid_task_id(self, env):
        obs = env.reset(task_id=EASY_TASK)
        assert obs.step_count == 0
        assert obs.done is False
        assert obs.reward == 0.01
        assert obs.hints_used == 0
        assert len(obs.report_text) > 0

    def test_reset_sets_state(self, env):
        env.reset(task_id=EASY_TASK)
        assert env.state is not None
        assert env.state.task_id == EASY_TASK
        assert env.state.current_step == 0

    def test_reset_invalid_task_raises(self, env):
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset(task_id="nonexistent_task")

    def test_reset_random_when_no_task_id(self, env):
        obs = env.reset()
        assert obs.report_text  # some task was loaded

    def test_reset_creates_new_episode_id(self, env):
        env.reset(task_id=EASY_TASK)
        id1 = env.state.episode_id
        env.reset(task_id=EASY_TASK)
        id2 = env.state.episode_id
        assert id1 != id2

    def test_reset_after_completed_episode(self, env, easy_findings):
        env.reset(task_id=EASY_TASK)
        env.step(StatAuditAction(action_type="submit_audit", findings=easy_findings))
        # Should be able to reset and start fresh
        obs = env.reset(task_id=EASY_TASK)
        assert obs.done is False
        assert obs.step_count == 0

    def test_task_has_report_text(self, env):
        for tid in ALL_TASK_IDS:
            obs = env.reset(task_id=tid)
            assert len(obs.report_text) > 100, f"Short report for {tid}"

    def test_state_difficulty_domain_match(self, env):
        env.reset(task_id="ab_testing_easy")
        assert env.state.task_difficulty == "easy"
        assert env.state.task_domain == "ab_testing"

        env.reset(task_id="regression_hard")
        assert env.state.task_difficulty == "hard"
        assert env.state.task_domain == "regression"

        env.reset(task_id="causal_inference_very_hard")
        assert env.state.task_difficulty == "very_hard"
        assert env.state.task_domain == "causal_inference"


class TestStepBeforeReset:
    def test_step_before_reset_raises(self, env):
        with pytest.raises(RuntimeError, match="reset"):
            env.step(StatAuditAction(action_type="mark_complete"))


class TestStepAfterDone:
    def test_step_after_mark_complete_raises(self, env):
        env.reset(task_id=EASY_TASK)
        env.step(StatAuditAction(action_type="mark_complete"))
        with pytest.raises(RuntimeError, match="done"):
            env.step(StatAuditAction(action_type="mark_complete"))

    def test_step_after_submit_audit_raises(self, env, easy_findings):
        env.reset(task_id=EASY_TASK)
        env.step(StatAuditAction(action_type="submit_audit", findings=easy_findings))
        with pytest.raises(RuntimeError, match="done"):
            env.step(StatAuditAction(action_type="mark_complete"))

    def test_reset_clears_done_state(self, env):
        env.reset(task_id=EASY_TASK)
        env.step(StatAuditAction(action_type="mark_complete"))
        env.reset(task_id=EASY_TASK)
        # Should not raise
        obs = env.step(StatAuditAction(action_type="mark_complete"))
        assert obs.done is True


class TestRequestClarification:
    def test_data_clarification_returns_data_summary(self, env):
        env.reset(task_id=EASY_TASK)
        obs = env.step(StatAuditAction(
            action_type="request_clarification",
            clarification_request="please show data",
        ))
        assert obs.raw_data_summary is not None
        assert obs.statistical_test_details is None
        assert obs.done is False
        assert obs.hints_used == 1

    def test_test_clarification_returns_test_details(self, env):
        env.reset(task_id=EASY_TASK)
        obs = env.step(StatAuditAction(
            action_type="request_clarification",
            clarification_request="what is the statistical test used?",
        ))
        assert obs.statistical_test_details is not None
        assert obs.done is False

    def test_multiple_clarifications_increment_hints(self, env):
        env.reset(task_id=EASY_TASK)
        env.step(StatAuditAction(
            action_type="request_clarification",
            clarification_request="show data",
        ))
        obs = env.step(StatAuditAction(
            action_type="request_clarification",
            clarification_request="show test details",
        ))
        assert obs.hints_used == 2
        assert env.state.hints_used == 2


class TestMarkComplete:
    def test_mark_complete_ends_episode(self, env):
        env.reset(task_id=EASY_TASK)
        obs = env.step(StatAuditAction(action_type="mark_complete"))
        assert obs.done is True
        assert obs.reward > 0  # clamped to strict (0, 1)
        assert obs.reward < 0.05  # minimal, not a real score


class TestSubmitAudit:
    def test_submit_ends_episode(self, env, easy_findings):
        env.reset(task_id=EASY_TASK)
        obs = env.step(StatAuditAction(action_type="submit_audit", findings=easy_findings))
        assert obs.done is True

    def test_submit_with_correct_findings_gives_high_reward(self, env, easy_findings):
        env.reset(task_id=EASY_TASK)
        obs = env.step(StatAuditAction(action_type="submit_audit", findings=easy_findings))
        assert obs.reward > 0.7, f"Expected reward > 0.7, got {obs.reward}"

    def test_submit_empty_findings_gives_zero_reward(self, env):
        env.reset(task_id=EASY_TASK)
        obs = env.step(StatAuditAction(action_type="submit_audit", findings=[]))
        assert obs.reward < 0.1

    def test_submit_produces_feedback(self, env, easy_findings):
        env.reset(task_id=EASY_TASK)
        obs = env.step(StatAuditAction(action_type="submit_audit", findings=easy_findings))
        assert obs.finding_feedback is not None
        assert "Score" in obs.finding_feedback

    def test_false_positives_tracked(self, env):
        env.reset(task_id=EASY_TASK)
        findings = [
            Finding(
                error_id="nonexistent_error_xyz",
                severity="critical",
                location="somewhere",
                description="this error does not exist",
                impact="none",
                correction="none",
                confidence=0.5,
            )
        ]
        env.step(StatAuditAction(action_type="submit_audit", findings=findings))
        assert env.state.false_positives == 1
        assert env.state.errors_found == 0

    def test_correct_errors_found_tracked(self, env, easy_findings):
        env.reset(task_id=EASY_TASK)
        env.step(StatAuditAction(action_type="submit_audit", findings=easy_findings))
        assert env.state.errors_found == 2
        assert env.state.total_errors_in_report == 2

    def test_step_count_increments(self, env):
        env.reset(task_id=EASY_TASK)
        env.step(StatAuditAction(
            action_type="request_clarification",
            clarification_request="data",
        ))
        assert env.state.current_step == 1
        env.step(StatAuditAction(action_type="mark_complete"))
        assert env.state.current_step == 2


class TestMaxSteps:
    def test_episode_ends_at_max_steps(self, env):
        env.reset(task_id=EASY_TASK)
        obs = None
        for _ in range(10):
            obs = env.step(StatAuditAction(
                action_type="request_clarification",
                clarification_request="data",
            ))
        assert obs.done is True
        assert obs.step_count == 10
