"""
StatAudit Environment — core RL environment logic.

Implements the standard OpenEnv interface:
  reset() → StatAuditObservation
  step(action) → StatAuditObservation
  state  → StatAuditState (property)
"""

from __future__ import annotations

import random
import uuid
from typing import Any, Dict, List, Optional

from models import StatAuditAction, StatAuditObservation, StatAuditState
from server.tasks import load_all_tasks
from server.graders.base_grader import grade_episode


MAX_STEPS = 10


class StatAuditEnvironment:
    """Statistical Analysis Auditing OpenEnv Environment."""

    def __init__(self) -> None:
        self._tasks: Dict[str, Any] = load_all_tasks()
        self._current_task: Optional[Dict[str, Any]] = None
        self._current_findings: List[Dict[str, Any]] = []
        self._step_count: int = 0
        self._hints_used: int = 0
        self._cumulative_reward: float = 0.0
        self._episode_id: Optional[str] = None
        self._state: Optional[StatAuditState] = None
        self._done: bool = False

    # ------------------------------------------------------------------
    # OpenEnv standard interface
    # ------------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None) -> StatAuditObservation:
        """Start a new episode. If task_id is None, a random task is chosen."""
        if task_id and task_id in self._tasks:
            self._current_task = self._tasks[task_id]
        elif task_id:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid tasks: {list(self._tasks.keys())}"
            )
        else:
            self._current_task = random.choice(list(self._tasks.values()))

        self._current_findings = []
        self._step_count = 0
        self._hints_used = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._episode_id = str(uuid.uuid4())

        self._state = StatAuditState(
            episode_id=self._episode_id,
            task_id=self._current_task["task_id"],
            task_difficulty=self._current_task["difficulty"],
            task_domain=self._current_task["domain"],
            current_step=0,
            total_errors_in_report=len(self._current_task["ground_truth_errors"]),
            errors_found=0,
            false_positives=0,
            cumulative_reward=0.0,
            hints_used=0,
        )

        return StatAuditObservation(
            report_text=self._current_task["report_text"],
            report_metadata=self._current_task["metadata"],
            step_count=0,
            max_steps=MAX_STEPS,
            hints_used=0,
            done=False,
            reward=0.0,
        )

    def step(self, action: StatAuditAction) -> StatAuditObservation:
        """Execute one step in the environment."""
        if self._current_task is None or self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step_count += 1
        self._state.current_step = self._step_count

        # Build base observation shared by all action types
        obs_kwargs: Dict[str, Any] = dict(
            report_text=self._current_task["report_text"],
            report_metadata=self._current_task["metadata"],
            previous_findings=[],
            step_count=self._step_count,
            max_steps=MAX_STEPS,
            hints_used=self._hints_used,
            done=False,
            reward=0.0,
        )

        if action.action_type == "submit_audit":
            # --- Grade submitted findings ---
            findings_dicts = [f.model_dump() for f in (action.findings or [])]
            reward, grading_details = grade_episode(
                findings_dicts,
                self._current_task["ground_truth_errors"],
            )

            self._current_findings = findings_dicts
            self._cumulative_reward += reward
            self._state.errors_found = grading_details["errors_found"]
            self._state.false_positives = grading_details["false_positives"]
            self._state.cumulative_reward = self._cumulative_reward

            obs_kwargs["previous_findings"] = action.findings or []
            obs_kwargs["finding_feedback"] = self._generate_feedback(grading_details)
            obs_kwargs["done"] = True
            obs_kwargs["reward"] = reward

        elif action.action_type == "request_clarification":
            # --- Provide additional information on request ---
            query = (action.clarification_request or "").lower()
            self._hints_used += 1
            self._state.hints_used = self._hints_used
            obs_kwargs["hints_used"] = self._hints_used

            if "data" in query:
                obs_kwargs["raw_data_summary"] = self._current_task.get("data_summary")
            if "test" in query or "statistical" in query or "method" in query:
                obs_kwargs["statistical_test_details"] = self._current_task.get("test_details")

            obs_kwargs["done"] = False
            obs_kwargs["reward"] = 0.0

        elif action.action_type == "mark_complete":
            obs_kwargs["done"] = True
            obs_kwargs["reward"] = 0.0

        # Force episode end when max steps reached
        if self._step_count >= MAX_STEPS:
            obs_kwargs["done"] = True

        if obs_kwargs["done"]:
            self._done = True

        return StatAuditObservation(**obs_kwargs)

    @property
    def state(self) -> Optional[StatAuditState]:
        """Return current episode state (None if no episode started)."""
        return self._state

    @property
    def tasks(self) -> Dict[str, Any]:
        """Return the task corpus."""
        return self._tasks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_feedback(self, grading_details: Dict[str, Any]) -> str:
        errors_found = grading_details["errors_found"]
        total_errors = grading_details["total_errors"]
        false_positives = grading_details["false_positives"]
        total_reward = grading_details["total_reward"]

        lines = [
            f"Audit complete. Score: {total_reward:.3f}/1.000",
            f"Errors identified: {errors_found}/{total_errors}",
        ]

        if errors_found == total_errors:
            lines.append("Excellent — you found all planted errors!")
        elif errors_found / max(total_errors, 1) >= 0.7:
            lines.append("Good — you caught most errors. Review missed items.")
        elif errors_found / max(total_errors, 1) >= 0.4:
            lines.append("Partial credit — significant errors remain undetected.")
        else:
            lines.append("Needs improvement — most errors were missed.")

        if false_positives > 0:
            lines.append(
                f"False positives: {false_positives} — "
                "flagged issues that are not methodological errors."
            )

        # Highlight best/worst findings
        scored = [d for d in grading_details["detailed_scores"] if d["score"] > 0]
        if scored:
            best = max(scored, key=lambda x: x["score"])
            lines.append(f"Best finding: '{best['error_id']}' (score {best['score']:.2f})")
            worst = min(scored, key=lambda x: x["score"])
            if worst["score"] < 0.5:
                lines.append(
                    f"Weakest finding: '{worst['error_id']}' (score {worst['score']:.2f}) — "
                    "deepen explanation and correction."
                )

        component = grading_details["component_rewards"]
        lines.append(
            f"\nComponent breakdown: "
            f"detection={component['error_detection']:.2f}, "
            f"severity={component['severity_accuracy']:.2f}, "
            f"explanation={component['explanation_quality']:.2f}, "
            f"correction={component['correction_validity']:.2f}, "
            f"efficiency={component['efficiency']:.2f}"
        )

        return "\n".join(lines)
