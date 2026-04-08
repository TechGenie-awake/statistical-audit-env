"""
StatAudit FastAPI Server

Implements the OpenEnv standard HTTP interface plus required extra endpoints:
  POST /reset       — start new episode
  POST /step        — execute action
  GET  /state       — current episode state
  GET  /tasks       — list all tasks and action schema
  GET  /baseline    — run baseline agents on all tasks
  POST /grader      — return grader score for current episode
"""

from __future__ import annotations

import os
import sys
import asyncio
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Make sure the repo root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import StatAuditAction, StatAuditObservation, StatAuditState
from server.environment import StatAuditEnvironment
from server.baselines.keyword_baseline import KeywordBaseline
from server.baselines.zero_shot_llm import ZeroShotLLM
from server.baselines.few_shot_cot import FewShotCoT

app = FastAPI(
    title="StatAudit — Statistical Analysis Auditing Environment",
    description=(
        "An OpenEnv environment for training AI agents to identify methodological "
        "errors in statistical reports. Supports A/B testing, regression, and "
        "causal inference domains across 9 tasks (easy → very hard)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global environment instance (stateful, per OpenEnv spec)
env = StatAuditEnvironment()


# ---------------------------------------------------------------------------
# Request / response helpers
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class GraderResponse(BaseModel):
    episode_id: Optional[str]
    task_id: Optional[str]
    errors_found: int
    total_errors: int
    false_positives: int
    score: float


# ---------------------------------------------------------------------------
# Standard OpenEnv endpoints
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=Dict[str, Any])
async def reset(body: ResetRequest = ResetRequest()):
    """
    Start a new episode.

    Pass `task_id` to select a specific task, or omit for a random task.
    Returns the initial observation containing the report to audit.
    """
    try:
        obs = env.reset(task_id=body.task_id)
        return obs.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step", response_model=Dict[str, Any])
async def step(action: StatAuditAction):
    """
    Execute one step in the current episode.

    - **submit_audit**: submit your findings for grading
    - **request_clarification**: request raw data or test details
    - **mark_complete**: end episode without submitting findings
    """
    try:
        obs = env.step(action)
        return obs.model_dump()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state", response_model=Dict[str, Any])
async def get_state():
    """Return the current episode state (metadata, progress, scores)."""
    if env.state is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    return env.state.model_dump()


# ---------------------------------------------------------------------------
# Required extra endpoints
# ---------------------------------------------------------------------------

@app.get("/tasks", response_model=Dict[str, Any])
async def list_tasks():
    """
    List all 9 tasks with metadata and return the action/observation schemas.
    """
    task_list = [
        {
            "task_id": task["task_id"],
            "title": task.get("title", task["task_id"]),
            "difficulty": task["difficulty"],
            "domain": task["domain"],
            "num_errors": len(task["ground_truth_errors"]),
            "description": task.get(
                "description",
                f"{task['difficulty'].title()} {task['domain']} analysis with "
                f"{len(task['ground_truth_errors'])} planted errors",
            ),
        }
        for task in sorted(
            env.tasks.values(),
            key=lambda t: (t["domain"], ["easy", "medium", "hard", "very_hard"].index(t["difficulty"])),
        )
    ]
    return {
        "tasks": task_list,
        "total_tasks": len(task_list),
        "action_schema": StatAuditAction.model_json_schema(),
        "observation_schema": StatAuditObservation.model_json_schema(),
    }


@app.get("/baseline", response_model=Dict[str, Any])
async def run_baseline(
    agents: Optional[str] = Query(
        default=None,
        description="Comma-separated list of agents to run: keyword,zero_shot,few_shot. Defaults to all.",
    )
):
    """
    Run baseline agents against all tasks and return their scores.

    Requires OPENAI_API_KEY environment variable for LLM-based baselines.
    If OPENAI_API_KEY is not set, only the keyword baseline will run.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    requested = set((agents or "keyword,zero_shot,few_shot").split(","))

    baselines: Dict[str, Any] = {}
    if "keyword" in requested:
        baselines["keyword_scanner"] = KeywordBaseline()
    if api_key:
        if "zero_shot" in requested:
            baselines["zero_shot_gpt35"] = ZeroShotLLM(api_key=api_key, model="gpt-3.5-turbo")
        if "few_shot" in requested:
            baselines["few_shot_gpt4"] = FewShotCoT(api_key=api_key, model="gpt-4o-mini")
    else:
        # Degrade gracefully when no API key
        if "zero_shot" in requested or "few_shot" in requested:
            pass  # Skip LLM baselines silently

    results: Dict[str, Any] = {}

    for baseline_name, baseline in baselines.items():
        scores_by_task: Dict[str, float] = {}

        for task_id, task in env.tasks.items():
            obs = env.reset(task_id=task_id)
            findings = baseline.audit_report(obs.report_text, obs.report_metadata)
            action = StatAuditAction(action_type="submit_audit", findings=findings)
            result_obs = env.step(action)
            scores_by_task[task_id] = round(result_obs.reward, 4)

        avg = round(sum(scores_by_task.values()) / len(scores_by_task), 4) if scores_by_task else 0.0
        results[baseline_name] = {
            "scores_by_task": scores_by_task,
            "average_score": avg,
        }

    return {
        "baselines": results,
        "tasks_evaluated": len(env.tasks),
        "note": (
            "LLM baselines skipped — set OPENAI_API_KEY to run zero_shot and few_shot."
            if not api_key and len(baselines) < len(requested)
            else "All requested baselines evaluated."
        ),
    }


@app.post("/grader", response_model=GraderResponse)
async def get_grader_score():
    """
    Return the grader score for the current (or most recently completed) episode.
    """
    if env.state is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")

    state = env.state
    total = state.total_errors_in_report
    score = round(state.cumulative_reward, 4)

    return GraderResponse(
        episode_id=state.episode_id,
        task_id=state.task_id,
        errors_found=state.errors_found,
        total_errors=total,
        false_positives=state.false_positives,
        score=score,
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "tasks_loaded": len(env.tasks)}


def main():
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
