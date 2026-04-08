"""
StatAudit — Baseline Inference Script

Runs all three baseline agents against all 9 tasks and prints a score table.
Reads OPENAI_API_KEY from the environment.

Usage:
  python inference.py                          # runs all baselines
  python inference.py --agent keyword          # keyword scanner only
  python inference.py --agent zero_shot        # GPT-3.5 zero-shot only
  python inference.py --agent few_shot         # GPT-4 few-shot CoT only
  python inference.py --url http://host:8000   # against a remote server
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import httpx

BASE_URL_DEFAULT = "http://localhost:8000"

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

DIFFICULTY_EMOJI = {
    "easy": "🟢",
    "medium": "🟡",
    "hard": "🔴",
    "very_hard": "🟣",
}


def reset_episode(client: httpx.Client, task_id: str) -> Dict[str, Any]:
    resp = client.post("/reset", json={"task_id": task_id})
    resp.raise_for_status()
    return resp.json()


def submit_findings(client: httpx.Client, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    action = {"action_type": "submit_audit", "findings": findings}
    resp = client.post("/step", json=action)
    resp.raise_for_status()
    return resp.json()


def get_grader_score(client: httpx.Client) -> Dict[str, Any]:
    resp = client.post("/grader")
    resp.raise_for_status()
    return resp.json()


def run_keyword_baseline(client: httpx.Client, task_ids: List[str]) -> Dict[str, float]:
    """Run keyword baseline via the server's /baseline endpoint."""
    resp = client.get("/baseline", params={"agents": "keyword"}, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    scores = data["baselines"].get("keyword_scanner", {}).get("scores_by_task", {})
    return {tid: scores.get(tid, 0.0) for tid in task_ids}


def run_llm_baseline_local(
    client: httpx.Client,
    task_ids: List[str],
    agent_type: str,
    api_key: str,
) -> Dict[str, float]:
    """Run an LLM baseline locally (without going through the server endpoint)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from server.baselines.zero_shot_llm import ZeroShotLLM
    from server.baselines.few_shot_cot import FewShotCoT

    if agent_type == "zero_shot":
        agent = ZeroShotLLM(api_key=api_key, model="gpt-3.5-turbo")
    else:
        agent = FewShotCoT(api_key=api_key, model="gpt-4o-mini")

    scores: Dict[str, float] = {}
    for task_id in task_ids:
        obs = reset_episode(client, task_id)
        report_text = obs["report_text"]
        metadata = obs.get("report_metadata", {})

        print(f"    Auditing {task_id}...", end=" ", flush=True)
        findings = agent.audit_report(report_text, metadata)
        result = submit_findings(client, [f.model_dump() for f in findings])
        reward = result.get("reward", 0.0)
        scores[task_id] = round(reward, 4)
        print(f"score={reward:.3f}")
        time.sleep(0.5)  # Rate limit courtesy

    return scores


def print_results_table(results: Dict[str, Dict[str, float]], task_meta: Dict[str, Any]) -> None:
    tasks_info = {t["task_id"]: t for t in task_meta.get("tasks", [])}

    # Header
    agent_names = list(results.keys())
    col_w = 14
    header = f"{'Task':<35} {'Diff':<10}" + "".join(f"{n:>{col_w}}" for n in agent_names)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    totals = {name: 0.0 for name in agent_names}
    count = 0

    for task_id in TASK_ORDER:
        if task_id not in tasks_info:
            continue
        info = tasks_info[task_id]
        diff = info.get("difficulty", "")
        emoji = DIFFICULTY_EMOJI.get(diff, "")
        row = f"{task_id:<35} {emoji+' '+diff:<10}"
        for name in agent_names:
            score = results[name].get(task_id, 0.0)
            totals[name] += score
            row += f"{score:>{col_w}.3f}"
        print(row)
        count += 1

    print("-" * len(header))
    avg_row = f"{'AVERAGE':<35} {'':10}"
    for name in agent_names:
        avg = totals[name] / count if count else 0.0
        avg_row += f"{avg:>{col_w}.3f}"
    print(avg_row)
    print("=" * len(header))


def main() -> None:
    parser = argparse.ArgumentParser(description="StatAudit baseline inference script")
    parser.add_argument("--url", default=BASE_URL_DEFAULT, help="Server base URL")
    parser.add_argument(
        "--agent",
        choices=["keyword", "zero_shot", "few_shot", "all"],
        default="all",
        help="Which baseline agent to run",
    )
    parser.add_argument("--output", help="Save results to JSON file")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if args.agent in ("zero_shot", "few_shot", "all") and not api_key:
        print("Warning: OPENAI_API_KEY not set. LLM baselines will be skipped.")

    client = httpx.Client(base_url=args.url, timeout=300)

    # Verify server is healthy
    try:
        health = client.get("/health").json()
        print(f"Server healthy. Tasks loaded: {health['tasks_loaded']}")
    except Exception as e:
        print(f"ERROR: Cannot connect to server at {args.url}: {e}")
        sys.exit(1)

    # Get task list
    tasks_resp = client.get("/tasks").json()
    task_ids = [t["task_id"] for t in tasks_resp["tasks"]]

    results: Dict[str, Dict[str, float]] = {}
    agents_to_run = [args.agent] if args.agent != "all" else ["keyword", "zero_shot", "few_shot"]

    for agent_type in agents_to_run:
        print(f"\nRunning baseline: {agent_type}")
        print("-" * 40)

        if agent_type == "keyword":
            scores = run_keyword_baseline(client, task_ids)
            results["keyword_scanner"] = scores
        elif agent_type == "zero_shot" and api_key:
            scores = run_llm_baseline_local(client, task_ids, "zero_shot", api_key)
            results["zero_shot_gpt35"] = scores
        elif agent_type == "few_shot" and api_key:
            scores = run_llm_baseline_local(client, task_ids, "few_shot", api_key)
            results["few_shot_gpt4"] = scores
        else:
            print(f"  Skipped (no API key).")

    if not results:
        print("\nNo results to display.")
        sys.exit(0)

    print_results_table(results, tasks_resp)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
