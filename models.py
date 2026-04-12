"""
StatAudit OpenEnv — Data Models

Pydantic models extending openenv-core base classes for Actions, Observations, and States.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import Field
from openenv.core import Action, Observation, State


class Finding(Action):
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


class StatAuditAction(Action):
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
        description="List of findings for submit_audit action.",
    )
    clarification_request: Optional[str] = Field(
        default=None,
        description=(
            "Free-text clarification request. "
            "Mention 'data' for raw data summary, 'test' for test details."
        ),
    )


class StatAuditObservation(Observation):
    """What the agent observes at each step."""

    # Core content
    report_text: str = Field(
        default="",
        description="The full statistical analysis report to audit"
    )
    report_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Contextual metadata about the report"
    )

    # Progressive assistance
    raw_data_summary: Optional[str] = Field(
        default=None,
        description="Summary of raw underlying data (available on request)"
    )
    statistical_test_details: Optional[str] = Field(
        default=None,
        description="Detailed test parameters (available on request)"
    )

    # Feedback
    previous_findings: List[Finding] = Field(
        default_factory=list,
        description="Agent's findings submitted in previous steps"
    )
    finding_feedback: Optional[str] = Field(
        default=None,
        description="Qualitative feedback on submitted findings"
    )

    # Step tracking
    step_count: int = Field(default=0, description="Current step number")
    max_steps: int = Field(default=10, description="Maximum steps per episode")
    hints_used: int = Field(default=0, description="Clarifications requested")

    # Episode status (openenv standard fields)
    done: bool = Field(default=False, description="Whether the episode has ended")
    reward: float = Field(default=0.01, description="Reward earned this step")


class StatAuditState(State):
    """Full episode metadata."""

    episode_id: str = Field(default="", description="Unique episode identifier")
    task_id: str = Field(default="", description="Task identifier")
    task_difficulty: str = Field(default="easy", description="Difficulty level")
    task_domain: str = Field(default="ab_testing", description="Domain")
    current_step: int = Field(default=0, description="Current step number")
    total_errors_in_report: int = Field(default=0, description="Ground truth error count")
    errors_found: int = Field(default=0, description="Errors correctly identified")
    false_positives: int = Field(default=0, description="Incorrect error claims")
    cumulative_reward: float = Field(default=0.0, description="Accumulated reward")
    hints_used: int = Field(default=0, description="Hints requested")
