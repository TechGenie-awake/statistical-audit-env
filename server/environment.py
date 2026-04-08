"""
StatAudit Environment — extends openenv.core.Environment.

Implements the standard OpenEnv interface:
  reset(**kwargs) → StatAuditObservation
  step(action)    → StatAuditObservation
  state           → StatAuditState (property)
"""

from __future__ import annotations

import random
import uuid
from typing import Any, Dict, List, Optional

from openenv.core import Environment

from models import StatAuditAction, StatAuditObservation, StatAuditState, Finding
from server.tasks import load_all_tasks
from server.graders.base_grader import grade_episode


MAX_STEPS = 10


class StatAuditEnvironment(
    Environment[StatAuditAction, StatAuditObservation, StatAuditState]
):
    """Statistical Analysis Auditing OpenEnv Environment."""

    def __init__(self) -> None:
        super().__init__()
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

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> StatAuditObservation:
        """Start a new episode."""
        task_id = kwargs.get("task_id")

        if task_id and task_id in self._tasks:
            self._current_task = self._tasks[task_id]
        elif task_id:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid tasks: {list(self._tasks.keys())}"
            )
        else:
            if seed is not None:
                random.seed(seed)
            self._current_task = random.choice(list(self._tasks.values()))

        self._current_findings = []
        self._step_count = 0
        self._hints_used = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._episode_id = episode_id or str(uuid.uuid4())

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

    def step(
        self,
        action: StatAuditAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> StatAuditObservation:
        """Execute one step in the environment."""
        if self._current_task is None or self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step_count += 1
        self._state.current_step = self._step_count

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

        if self._step_count >= MAX_STEPS:
            obs_kwargs["done"] = True

        if obs_kwargs["done"]:
            self._done = True

        return StatAuditObservation(**obs_kwargs)

    @property
    def state(self) -> StatAuditState:
        """Return current episode state."""
        if self._state is None:
            return StatAuditState()
        return self._state

    @property
    def tasks(self) -> Dict[str, Any]:
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
            lines.append("Good — you caught most errors.")
        elif errors_found / max(total_errors, 1) >= 0.4:
            lines.append("Partial credit — significant errors remain.")
        else:
            lines.append("Needs improvement — most errors were missed.")

        if false_positives > 0:
            lines.append(f"False positives: {false_positives}")

        scored = [d for d in grading_details["detailed_scores"] if d["score"] > 0]
        if scored:
            best = max(scored, key=lambda x: x["score"])
            lines.append(f"Best finding: '{best['error_id']}' (score {best['score']:.2f})")
            worst = min(scored, key=lambda x: x["score"])
            if worst["score"] < 0.5:
                lines.append(f"Weakest: '{worst['error_id']}' (score {worst['score']:.2f})")

        component = grading_details["component_rewards"]
        lines.append(
            f"\nBreakdown: detection={component['error_detection']:.2f}, "
            f"severity={component['severity_accuracy']:.2f}, "
            f"explanation={component['explanation_quality']:.2f}, "
            f"correction={component['correction_validity']:.2f}, "
            f"efficiency={component['efficiency']:.2f}"
        )

        return "\n".join(lines)
