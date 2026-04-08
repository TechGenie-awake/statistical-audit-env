"""
StatAudit — Baseline Inference Script
======================================
MANDATORY environment variables:
    API_BASE_URL   The API endpoint for the LLM (e.g. https://router.huggingface.co/v1)
    MODEL_NAME     The model identifier (e.g. Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       Your Hugging Face / API key

Optional:
    STATAUDIT_BASE_URL   StatAudit environment server URL (default: http://localhost:8000)

STDOUT FORMAT (strictly followed):
    [START] task=<task_id> env=stataudit model=<model_name>
    [STEP]  step=<n> action=<action_type> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
STATAUDIT_BASE_URL = os.getenv("STATAUDIT_BASE_URL", "http://localhost:8000")

MAX_STEPS = 3          # clarification + submit (well within 20 min limit)
SUCCESS_THRESHOLD = 0.3

BENCHMARK = "stataudit"

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
# Logging helpers (strict format)
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

    Return ONLY a valid JSON array of findings. No prose, no markdown.
""").strip()


def call_llm(report_text: str, client: OpenAI) -> List[Dict[str, Any]]:
    """Call LLM and return parsed findings list."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Audit this report:\n\n{report_text}"},
            ],
            temperature=0.2,
            max_tokens=2000,
        )
        content = response.choices[0].message.content or "[]"

        # Extract JSON array robustly
        start = content.find("[")
        end = content.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        raw = json.loads(content[start:end])

        # Validate and normalise each finding
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

    except Exception as exc:
        return []


# ---------------------------------------------------------------------------
# Environment interaction helpers
# ---------------------------------------------------------------------------

def env_reset(http: httpx.Client, task_id: str) -> Dict[str, Any]:
    return http.post("/reset", json={"task_id": task_id}, timeout=30).json()


def env_step(http: httpx.Client, action: Dict[str, Any]) -> Dict[str, Any]:
    return http.post("/step", json=action, timeout=60).json()


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(task_id: str, http: httpx.Client, llm: OpenAI) -> Dict[str, Any]:
    """Run one full episode for a task. Returns episode summary."""
    rewards: List[float] = []
    step = 0
    score = 0.0
    success = False
    last_error: Optional[str] = None

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # --- Step 0: reset ---
        obs = env_reset(http, task_id)
        report_text = obs.get("report_text", "")

        # --- Step 1: optional clarification for extra context ---
        step += 1
        clarify_action = {
            "action_type": "request_clarification",
            "clarification_request": "Please provide the raw data summary and statistical test details.",
        }
        try:
            clarify_obs = env_step(http, clarify_action)
            reward_step1 = float(clarify_obs.get("reward", 0.0))
            done_step1 = bool(clarify_obs.get("done", False))
            rewards.append(reward_step1)
            log_step(step, "request_clarification", reward_step1, done_step1, None)

            # Append any extra context to report
            extra = []
            if clarify_obs.get("raw_data_summary"):
                extra.append(f"DATA: {clarify_obs['raw_data_summary']}")
            if clarify_obs.get("statistical_test_details"):
                extra.append(f"TESTS: {clarify_obs['statistical_test_details']}")
            if extra:
                report_text += "\n\n" + "\n".join(extra)
        except Exception as exc:
            last_error = str(exc)
            log_step(step, "request_clarification", 0.0, False, last_error)

        # --- Step 2: LLM audit ---
        findings = call_llm(report_text, llm)

        step += 1
        submit_action = {
            "action_type": "submit_audit",
            "findings": findings,
        }
        try:
            result_obs = env_step(http, submit_action)
            reward_step2 = float(result_obs.get("reward", 0.0))
            done_step2 = bool(result_obs.get("done", True))
            rewards.append(reward_step2)
            score = reward_step2
            success = score >= SUCCESS_THRESHOLD
            log_step(step, "submit_audit", reward_step2, done_step2, None)
        except Exception as exc:
            last_error = str(exc)
            log_step(step, "submit_audit", 0.0, True, last_error)

    except Exception as exc:
        last_error = str(exc)
        log_step(step or 1, "error", 0.0, True, last_error)

    log_end(success=success, steps=step, score=score, rewards=rewards)
    return {"task_id": task_id, "score": score, "steps": step, "success": success}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not API_KEY:
        print("ERROR: HF_TOKEN or API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    llm = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    http = httpx.Client(base_url=STATAUDIT_BASE_URL, timeout=120)

    # Verify server health
    try:
        health = http.get("/health", timeout=15).json()
    except Exception as exc:
        print(f"ERROR: Cannot reach StatAudit server at {STATAUDIT_BASE_URL}: {exc}", file=sys.stderr)
        sys.exit(1)

    # Get available tasks
    try:
        tasks_resp = http.get("/tasks", timeout=15).json()
        available = {t["task_id"] for t in tasks_resp.get("tasks", [])}
    except Exception:
        available = set(TASK_ORDER)

    results = []
    for task_id in TASK_ORDER:
        if task_id not in available:
            continue
        summary = run_episode(task_id, http, llm)
        results.append(summary)

    # Final summary to stderr (doesn't pollute [START]/[STEP]/[END] stdout)
    total = len(results)
    avg = sum(r["score"] for r in results) / total if total else 0.0
    print(f"\n# Summary: {total} tasks, average score={avg:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
