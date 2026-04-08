"""
StatAudit — Baseline Inference Script
======================================
MANDATORY environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME  The name of the local Docker image (optional, for from_docker_image())

STDOUT FORMAT (strictly followed):
    [START] task=<task_id> env=stataudit model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI
from openenv.core import GenericEnvClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK = "stataudit"
SUCCESS_THRESHOLD = 0.1

TASK_ORDER = [
    "ab_testing_easy",
    "ab_testing_medium",
    "ab_testing_hard",
    "regression_easy",
    "regression_medium",
    "regression_hard",
    "causal_inference_medium",
    "causal_inference_hard",
    "causal_inference_very_hard",
]


# ---------------------------------------------------------------------------
# Logging helpers (strict format — do not modify)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = "true" if done else "false"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = "true" if success else "false"
    print(
        f"[END] success={success_val} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM audit agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert statistician auditing a statistical analysis report.
    Identify all methodological errors: multiple testing, selection bias,
    omitted variable bias, heteroskedasticity, endogeneity, reverse causality,
    Simpson's paradox, survivorship bias, spurious regression, parallel trends
    violations, IV exclusion restriction violations, SUTVA/spillover, and any
    other statistical flaws.

    For each error return a JSON object with these exact keys:
    - error_id: snake_case identifier (e.g. "multiple_testing_violation")
    - severity: one of "critical", "major", "minor"
    - location: section or table name in the report
    - description: what is wrong (be specific)
    - impact: business/scientific consequences
    - correction: how to fix it
    - confidence: float 0.0-1.0

    Return ONLY a valid JSON array of findings. No prose, no markdown fences.
""").strip()


def call_llm(report_text: str, llm: OpenAI) -> List[Dict[str, Any]]:
    """Call LLM and return parsed findings list."""
    try:
        response = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Audit this report:\n\n{report_text}"},
            ],
            temperature=0.2,
            max_tokens=2000,
        )
        content = response.choices[0].message.content or "[]"

        start = content.find("[")
        end = content.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        raw = json.loads(content[start:end])

        findings = []
        for item in raw:
            severity = item.get("severity", "minor")
            if severity not in ("critical", "major", "minor"):
                severity = "minor"
            confidence = float(item.get("confidence", 0.7))
            confidence = max(0.0, min(1.0, confidence))
            findings.append({
                "error_id": str(item.get("error_id", "unknown_error")).strip(),
                "severity": severity,
                "location": str(item.get("location", "Unknown")),
                "description": str(item.get("description", "")),
                "impact": str(item.get("impact", "")),
                "correction": str(item.get("correction", "")),
                "confidence": confidence,
            })
        return findings
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Single episode runner (async, uses openenv GenericEnvClient)
# ---------------------------------------------------------------------------

async def run_episode(
    task_id: str,
    client: GenericEnvClient,
    llm: OpenAI,
) -> Dict[str, Any]:
    """Run one full episode for a task."""
    rewards: List[float] = []
    step = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # --- Reset ---
        obs = await client.reset(options={"task_id": task_id})
        report_text = ""
        if hasattr(obs, "report_text"):
            report_text = obs.report_text
        elif isinstance(obs, dict):
            report_text = obs.get("report_text", "")
        else:
            report_text = str(obs)

        # --- Step 1: request clarification for extra context ---
        step += 1
        try:
            clarify_action = {
                "action_type": "request_clarification",
                "clarification_request": "Please provide the raw data summary and statistical test details.",
            }
            clarify_obs = await client.step(clarify_action)

            reward_c = 0.0
            done_c = False
            if hasattr(clarify_obs, "reward"):
                reward_c = float(clarify_obs.reward)
            elif isinstance(clarify_obs, dict):
                reward_c = float(clarify_obs.get("reward", 0.0))

            if hasattr(clarify_obs, "done"):
                done_c = bool(clarify_obs.done)
            elif isinstance(clarify_obs, dict):
                done_c = bool(clarify_obs.get("done", False))

            rewards.append(reward_c)
            log_step(step, "request_clarification", reward_c, done_c, None)

            # Append extra context
            extra = []
            raw_data = getattr(clarify_obs, "raw_data_summary", None) or (
                clarify_obs.get("raw_data_summary") if isinstance(clarify_obs, dict) else None
            )
            test_details = getattr(clarify_obs, "statistical_test_details", None) or (
                clarify_obs.get("statistical_test_details") if isinstance(clarify_obs, dict) else None
            )
            if raw_data:
                extra.append(f"DATA: {raw_data}")
            if test_details:
                extra.append(f"TESTS: {test_details}")
            if extra:
                report_text += "\n\n" + "\n".join(extra)

        except Exception as exc:
            rewards.append(0.0)
            log_step(step, "request_clarification", 0.0, False, str(exc))

        # --- Step 2: LLM audit and submit ---
        findings = call_llm(report_text, llm)

        step += 1
        try:
            submit_action = {
                "action_type": "submit_audit",
                "findings": findings,
            }
            result_obs = await client.step(submit_action)

            reward_s = 0.0
            done_s = True
            if hasattr(result_obs, "reward"):
                reward_s = float(result_obs.reward)
            elif isinstance(result_obs, dict):
                reward_s = float(result_obs.get("reward", 0.0))

            if hasattr(result_obs, "done"):
                done_s = bool(result_obs.done)
            elif isinstance(result_obs, dict):
                done_s = bool(result_obs.get("done", True))

            rewards.append(reward_s)
            score = reward_s
            success = score >= SUCCESS_THRESHOLD
            log_step(step, "submit_audit", reward_s, done_s, None)

        except Exception as exc:
            rewards.append(0.0)
            log_step(step, "submit_audit", 0.0, True, str(exc))

    except Exception as exc:
        if not rewards:
            rewards.append(0.0)
        log_step(step or 1, "error", 0.0, True, str(exc))

    log_end(success=success, steps=step, score=score, rewards=rewards)
    return {"task_id": task_id, "score": score, "steps": step, "success": success}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main() -> None:
    if not API_KEY:
        print("ERROR: HF_TOKEN or API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    llm = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    # Connect to the environment using the openenv client
    if LOCAL_IMAGE_NAME:
        # Validator provides LOCAL_IMAGE_NAME — spin up Docker container
        client = await GenericEnvClient.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        # Fallback: connect to HF Space or local server
        env_url = os.getenv(
            "STATAUDIT_BASE_URL",
            "https://GJaiswal2006-statistical-audit-env.hf.space",
        )
        client = GenericEnvClient(base_url=env_url)

    async with client:
        results = []
        for task_id in TASK_ORDER:
            try:
                summary = await run_episode(task_id, client, llm)
                results.append(summary)
            except Exception as exc:
                log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
                log_step(1, "error", 0.0, True, str(exc))
                log_end(success=False, steps=1, score=0.0, rewards=[0.0])
                results.append({"task_id": task_id, "score": 0.0, "steps": 1, "success": False})

    total = len(results)
    avg = sum(r["score"] for r in results) / total if total else 0.0
    print(f"\n# Summary: {total} tasks, average score={avg:.3f}", file=sys.stderr)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
