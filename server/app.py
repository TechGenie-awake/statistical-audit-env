"""
StatAudit FastAPI Server

Uses openenv-core's create_fastapi_app for standard endpoints (/reset, /step,
/state, /ws, /health, /schema, /metadata) and adds custom endpoints on top:
  GET  /tasks       — list all tasks and action schema
  GET  /baseline    — run baseline agents on all tasks
  POST /grader      — return grader score for current episode
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

from fastapi import HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Make sure the repo root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core import create_fastapi_app

from models import StatAuditAction, StatAuditObservation
from server.environment import StatAuditEnvironment
from server.baselines.keyword_baseline import KeywordBaseline
from server.baselines.zero_shot_llm import ZeroShotLLM
from server.baselines.few_shot_cot import FewShotCoT


# ---------------------------------------------------------------------------
# Create the app via openenv-core (provides /ws, /reset, /step, /state, etc.)
# ---------------------------------------------------------------------------

app = create_fastapi_app(
    env=StatAuditEnvironment,
    action_cls=StatAuditAction,
    observation_cls=StatAuditObservation,
)

app.title = "StatAudit — Statistical Analysis Auditing Environment"
app.description = (
    "An OpenEnv environment for training AI agents to identify methodological "
    "errors in statistical reports. Supports A/B testing, regression, and "
    "causal inference domains across 9 tasks (easy → very hard)."
)
app.version = "1.0.0"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper: standalone environment instance for custom endpoints
# (The openenv-core app manages its own env instances per WebSocket session,
#  but our custom endpoints need a shared instance for task listing / baselines.)
# ---------------------------------------------------------------------------

_shared_env = StatAuditEnvironment()


# ---------------------------------------------------------------------------
# Request / response helpers
# ---------------------------------------------------------------------------

class GraderResponse(BaseModel):
    episode_id: Optional[str]
    task_id: Optional[str]
    errors_found: int
    total_errors: int
    false_positives: int
    score: float


# ---------------------------------------------------------------------------
# Custom endpoints
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
            _shared_env.tasks.values(),
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
        if "zero_shot" in requested or "few_shot" in requested:
            pass  # Skip LLM baselines silently

    results: Dict[str, Any] = {}

    for baseline_name, baseline in baselines.items():
        scores_by_task: Dict[str, float] = {}

        for task_id, task in _shared_env.tasks.items():
            obs = _shared_env.reset(task_id=task_id)
            findings = baseline.audit_report(obs.report_text, obs.report_metadata)
            action = StatAuditAction(action_type="submit_audit", findings=findings)
            result_obs = _shared_env.step(action)
            scores_by_task[task_id] = round(result_obs.reward, 4)

        avg = round(sum(scores_by_task.values()) / len(scores_by_task), 4) if scores_by_task else 0.0
        results[baseline_name] = {
            "scores_by_task": scores_by_task,
            "average_score": avg,
        }

    return {
        "baselines": results,
        "tasks_evaluated": len(_shared_env.tasks),
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
    state = _shared_env.state
    if not state.episode_id:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")

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
# Entry point
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
