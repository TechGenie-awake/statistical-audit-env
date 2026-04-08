"""Task corpus loader for StatAudit environment."""

import json
from pathlib import Path
from typing import Dict, Any

TASKS_DIR = Path(__file__).parent


def load_all_tasks() -> Dict[str, Any]:
    """Load all task JSON files from the task directories."""
    tasks = {}
    for task_file in TASKS_DIR.rglob("*.json"):
        with open(task_file) as f:
            task = json.load(f)
        tasks[task["task_id"]] = task
    return tasks
