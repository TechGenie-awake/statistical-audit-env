"""Tests for Pydantic data models."""

import pytest
from pydantic import ValidationError
from models import Finding, StatAuditAction, StatAuditObservation, StatAuditState


class TestFinding:
    def test_valid_finding(self):
        f = Finding(
            error_id="multiple_testing_violation",
            severity="critical",
            location="Results section",
            description="Multiple metrics tested without correction.",
            impact="False positive rate inflated.",
            correction="Apply Bonferroni correction.",
            confidence=0.9,
        )
        assert f.error_id == "multiple_testing_violation"
        assert f.severity == "critical"
        assert f.confidence == 0.9

    def test_invalid_severity(self):
        with pytest.raises(ValidationError):
            Finding(
                error_id="x",
                severity="extreme",  # invalid
                location="s",
                description="d",
                impact="i",
                correction="c",
                confidence=0.5,
            )

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            Finding(
                error_id="x", severity="minor", location="s",
                description="d", impact="i", correction="c",
                confidence=1.5,  # > 1.0
            )
        with pytest.raises(ValidationError):
            Finding(
                error_id="x", severity="minor", location="s",
                description="d", impact="i", correction="c",
                confidence=-0.1,  # < 0.0
            )

    def test_all_severity_levels(self):
        for sev in ("critical", "major", "minor"):
            f = Finding(
                error_id="x", severity=sev, location="s",
                description="d", impact="i", correction="c", confidence=0.5,
            )
            assert f.severity == sev


class TestStatAuditAction:
    def test_submit_audit_action(self):
        action = StatAuditAction(
            action_type="submit_audit",
            findings=[
                Finding(
                    error_id="e1", severity="major", location="s",
                    description="d", impact="i", correction="c", confidence=0.7,
                )
            ],
        )
        assert action.action_type == "submit_audit"
        assert len(action.findings) == 1

    def test_request_clarification_action(self):
        action = StatAuditAction(
            action_type="request_clarification",
            clarification_request="Please provide raw data.",
        )
        assert action.clarification_request == "Please provide raw data."
        assert action.findings is None

    def test_mark_complete_action(self):
        action = StatAuditAction(action_type="mark_complete")
        assert action.action_type == "mark_complete"
        assert action.findings is None

    def test_invalid_action_type(self):
        with pytest.raises(ValidationError):
            StatAuditAction(action_type="invalid_type")

    def test_submit_audit_with_empty_findings(self):
        action = StatAuditAction(action_type="submit_audit", findings=[])
        assert action.findings == []


class TestStatAuditObservation:
    def test_default_observation(self):
        obs = StatAuditObservation(
            report_text="Report content",
            report_metadata={"domain": "ab_testing"},
        )
        assert obs.step_count == 0
        assert obs.max_steps == 10
        assert obs.done is False
        assert obs.reward == 0.0
        assert obs.hints_used == 0
        assert obs.previous_findings == []
        assert obs.raw_data_summary is None
        assert obs.statistical_test_details is None

    def test_observation_with_all_fields(self):
        obs = StatAuditObservation(
            report_text="text",
            report_metadata={},
            raw_data_summary="data",
            statistical_test_details="details",
            step_count=3,
            max_steps=10,
            hints_used=1,
            done=True,
            reward=0.75,
        )
        assert obs.raw_data_summary == "data"
        assert obs.statistical_test_details == "details"
        assert obs.reward == 0.75
        assert obs.done is True
