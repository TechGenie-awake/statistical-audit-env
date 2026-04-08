"""
StatAudit OpenEnv - Data Models

Pydantic models for Actions, Observations, and States.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class Finding(BaseModel):
    """Individual error finding submitted by the agent."""

    error_id: str = Field(
        description="Error identifier (e.g., 'multiple_testing_violation', 'selection_bias')"
    )
    severity: Literal["critical", "major", "minor"] = Field(
        description="Severity classification of the error"
    )
    location: str = Field(
        description="Location in report (e.g., 'Section 2.3, Results' or 'Table 1, Methodology')"
    )
    description: str = Field(
        description="What is wrong — explain the methodological error clearly"
    )
    impact: str = Field(
        description="Why it matters — business or scientific consequences of the error"
    )
    correction: str = Field(
        description="How to fix it — specific remediation suggestion"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Agent confidence in this finding (0.0–1.0)",
    )


class StatAuditAction(BaseModel):
    """Action submitted by the agent to the environment."""

    action_type: Literal["submit_audit", "request_clarification", "mark_complete"] = Field(
        description=(
            "Type of action. "
            "'submit_audit': submit findings for grading; "
            "'request_clarification': request additional data/details; "
            "'mark_complete': signal completion without findings."
        )
    )
    findings: Optional[List[Finding]] = Field(
        default=None,
        description="List of findings for submit_audit action. Required when action_type='submit_audit'.",
    )
    clarification_request: Optional[str] = Field(
        default=None,
        description=(
            "Free-text clarification request for request_clarification action. "
            "Mention 'data' to receive raw data summary, 'test' to receive test details."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "action_type": "submit_audit",
                    "findings": [
                        {
                            "error_id": "multiple_testing_violation",
                            "severity": "critical",
                            "location": "Results section",
                            "description": "Five metrics were tested and only the best p-value reported without multiple comparison correction.",
                            "impact": "Inflates the false positive rate from 5% to ~23%, making the result unreliable.",
                            "correction": "Apply Bonferroni correction (α = 0.05 / 5 = 0.01) or pre-specify a single primary metric.",
                            "confidence": 0.95,
                        }
                    ],
                }
            ]
        }
    }


class StatAuditObservation(BaseModel):
    """What the agent observes at each step."""

    # Core content
    report_text: str = Field(
        description="The full statistical analysis report to audit"
    )
    report_metadata: Dict[str, Any] = Field(
        description="Contextual metadata about the report (sample_size, test_type, domain, etc.)"
    )

    # Progressive assistance (unlocks on clarification request)
    raw_data_summary: Optional[str] = Field(
        default=None,
        description="Summary of raw underlying data (available on explicit request)"
    )
    statistical_test_details: Optional[str] = Field(
        default=None,
        description="Detailed test parameters and configuration (available on explicit request)"
    )

    # Feedback on previous findings
    previous_findings: List[Finding] = Field(
        default_factory=list,
        description="Agent's findings submitted in previous steps of this episode"
    )
    finding_feedback: Optional[str] = Field(
        default=None,
        description="Qualitative feedback on the agent's submitted findings"
    )

    # Step tracking
    step_count: int = Field(default=0, description="Current step number in the episode")
    max_steps: int = Field(default=10, description="Maximum steps allowed per episode")
    hints_used: int = Field(default=0, description="Number of clarification hints requested")

    # Episode status
    done: bool = Field(default=False, description="Whether the episode has ended")
    reward: float = Field(default=0.0, description="Reward earned in this step")


class StatAuditState(BaseModel):
    """Full episode metadata (internal state)."""

    episode_id: str = Field(description="Unique episode identifier (UUID)")
    task_id: str = Field(description="Task identifier (e.g., 'ab_testing_easy')")
    task_difficulty: Literal["easy", "medium", "hard", "very_hard"] = Field(
        description="Difficulty level of the current task"
    )
    task_domain: Literal["ab_testing", "regression", "causal_inference"] = Field(
        description="Domain of the current task"
    )
    current_step: int = Field(description="Current step number")
    total_errors_in_report: int = Field(
        description="Ground truth count of planted errors in the report"
    )
    errors_found: int = Field(description="Number of errors correctly identified so far")
    false_positives: int = Field(description="Number of incorrect error claims")
    cumulative_reward: float = Field(default=0.0, description="Accumulated reward this episode")
    hints_used: int = Field(default=0, description="Number of hints/clarifications requested")
