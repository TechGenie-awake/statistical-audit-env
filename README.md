---
title: StatAudit
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: docker
tags:
  - openenv
  - statistics
  - reinforcement-learning
  - auditing
pinned: false
---

# StatAudit: AI-Powered Statistical Analysis Auditor

> An [OpenEnv](https://github.com/openenv) environment for training AI agents to identify methodological errors in statistical reports.

---

## Overview

StatAudit simulates the critical real-world task of **auditing statistical analyses** for methodological flaws. Agents read A/B test reports, regression analyses, and causal inference studies, then identify errors that could lead to costly wrong business decisions.

**Why this matters**: Companies run thousands of A/B tests and analyses annually. Methodological errors — p-hacking, selection bias, omitted variable bias, spurious regression — directly cause bad product decisions, wasted engineering work, and misallocated marketing budgets worth millions of dollars.

**What agents must do**:
1. Read a statistical analysis report
2. Identify methodological errors (multiple testing, selection bias, etc.)
3. Classify each error's severity (`critical`, `major`, `minor`)
4. Explain _why_ each error is problematic (not just what it is)
5. Suggest specific, actionable corrections

---

## Quick Start

### Run with Docker

```bash
# Clone the repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/statistical-audit-env
cd statistical-audit-env

# Build the image
docker build -t stataudit:latest -f server/Dockerfile .

# Run (optionally pass your OpenAI key for LLM baselines)
docker run -d -p 8000:8000 -e OPENAI_API_KEY=sk-... stataudit:latest

# Verify it's running
curl http://localhost:8000/health
```

### Install Python client

```bash
pip install fastapi uvicorn pydantic sentence-transformers openai httpx
```

### Basic usage

```python
import httpx

client = httpx.Client(base_url="http://localhost:8000")

# Start an episode on the easy A/B test task
obs = client.post("/reset", json={"task_id": "ab_testing_easy"}).json()
print(obs["report_text"])

# Submit findings
action = {
    "action_type": "submit_audit",
    "findings": [
        {
            "error_id": "multiple_testing_violation",
            "severity": "critical",
            "location": "Results section",
            "description": "Five metrics were tested without multiple comparison correction.",
            "impact": "Family-wise false positive rate inflates from 5% to ~23%.",
            "correction": "Apply Bonferroni correction (α = 0.05/5 = 0.01) or pre-specify one primary metric.",
            "confidence": 0.95
        },
        {
            "error_id": "early_stopping_peeking",
            "severity": "major",
            "location": "Methodology section",
            "description": "Test was stopped on Day 5 when p<0.05 was observed (peeking).",
            "impact": "Optional stopping inflates false positive rate without pre-specified stopping rules.",
            "correction": "Use sequential testing methods (SPRT, O'Brien-Fleming boundaries).",
            "confidence": 0.88
        }
    ]
}
result = client.post("/step", json=action).json()
print(f"Reward: {result['reward']:.3f}")
print(result['finding_feedback'])
```

### Run baseline inference

```bash
# Run all baselines (requires OPENAI_API_KEY for LLM agents)
export OPENAI_API_KEY=sk-...
python inference.py

# Keyword baseline only (no API key needed)
python inference.py --agent keyword

# Against a remote Hugging Face Space
python inference.py --url https://YOUR_USERNAME-statistical-audit-env.hf.space
```

---

## Environment Details

### Action Space

```python
class Finding(BaseModel):
    error_id: str        # e.g., "multiple_testing_violation"
    severity: Literal["critical", "major", "minor"]
    location: str        # where in report (section name, table, etc.)
    description: str     # what is wrong
    impact: str          # why it matters (business/scientific consequences)
    correction: str      # how to fix it
    confidence: float    # agent confidence, 0.0–1.0

class StatAuditAction(BaseModel):
    action_type: Literal["submit_audit", "request_clarification", "mark_complete"]
    findings: Optional[List[Finding]]          # for submit_audit
    clarification_request: Optional[str]       # for request_clarification
```

**Action types**:
- `submit_audit`: Submit findings for grading. Ends the episode.
- `request_clarification`: Request additional data. Mention "data" for raw data summary, "test" for statistical test details. Does not end episode.
- `mark_complete`: Signal completion without submitting. Score = 0.

### Observation Space

```python
class StatAuditObservation(BaseModel):
    report_text: str              # The full report to audit
    report_metadata: Dict         # Context (sample_size, test_type, domain, etc.)
    raw_data_summary: Optional[str]         # Unlocked on clarification request
    statistical_test_details: Optional[str] # Unlocked on clarification request
    previous_findings: List[Finding]
    finding_feedback: Optional[str]   # Qualitative feedback after submit_audit
    step_count: int
    max_steps: int  # 10
    hints_used: int
    done: bool
    reward: float
```

### Reward Function

Multi-dimensional reward, total ∈ [0, 1]:

| Component | Weight | Description |
|-----------|--------|-------------|
| Error detection | 40% | Found the planted errors? |
| Severity accuracy | 20% | Classified severity correctly? |
| Explanation quality | 20% | Understands WHY it's wrong? (semantic similarity) |
| Correction validity | 15% | Suggested fix is correct? |
| Efficiency | 5% | No false positives? |

Reward is computed per error then averaged. Finding 0 errors = 0.0. Finding all errors with precise explanations and corrections = ~0.95+.

---

## Tasks

9 tasks across 3 domains with clear difficulty progression.

### Domain 1: A/B Testing

| Task ID | Difficulty | Errors | Description |
|---------|-----------|--------|-------------|
| `ab_testing_easy` | Easy | 2 | E-commerce checkout button test — multiple testing + peeking |
| `ab_testing_medium` | Medium | 3 | Social media Stories feature — Simpson's paradox + survivorship bias + wrong test |
| `ab_testing_hard` | Hard | 5 | Ad bidding algorithm — cluster randomization + novelty effect + metric switching + network effects + regression to mean |

### Domain 2: Regression Analysis

| Task ID | Difficulty | Errors | Description |
|---------|-----------|--------|-------------|
| `regression_easy` | Easy | 2 | House price model — omitted variable bias (location) + heteroskedasticity |
| `regression_medium` | Medium | 3 | Customer churn model — reverse causality + multicollinearity + overfitting |
| `regression_hard` | Hard | 5 | GDP forecasting — spurious regression + reverse causality + simultaneity + nonlinearity + measurement error |

### Domain 3: Causal Inference

| Task ID | Difficulty | Errors | Description |
|---------|-----------|--------|-------------|
| `causal_inference_medium` | Medium | 3 | Marketing email ROI — selection bias + holiday confounding + post-treatment bias |
| `causal_inference_hard` | Hard | 4 | Job training program — parallel trends violated + IV exclusion restriction + SUTVA spillover + completer selection |
| `causal_inference_very_hard` | Very Hard | 6 | Multi-armed bandit — non-stationarity + regret bound miscalculation + prior misspecification + batch update bug + contextual degeneracy + premature exploitation |

### Error taxonomy

The environment covers 20+ distinct error types:

**A/B Testing**: multiple testing, early stopping/peeking, Simpson's paradox, survivorship bias, network effects, novelty effect, cluster randomization, metric switching, regression to mean

**Regression**: omitted variable bias, heteroskedasticity, multicollinearity, overfitting, spurious regression, measurement error, simultaneity, reverse causality, model misspecification

**Causal Inference**: selection bias, confounding, post-treatment bias (collider), parallel trends violation, IV exclusion restriction, SUTVA/spillover, self-selection, non-stationarity, regret bound errors, Bayesian prior misspecification

---

## Grading System

The grading is **hybrid**: deterministic keyword/location matching combined with **semantic similarity** via `sentence-transformers` (`all-MiniLM-L6-v2`).

### Per-finding score components

```
score = 0.15 × detection
      + 0.10 × location_accuracy
      + 0.15 × keyword_coverage
      + 0.25 × explanation_quality  ← semantic similarity
      + 0.20 × correction_validity  ← semantic similarity
      + 0.15 × severity_accuracy
```

- **Detection**: Did the agent identify this exact error (matching `error_id`)?
- **Location**: Is the cited location correct?
- **Keywords**: Does the description use the right statistical terminology?
- **Explanation quality**: How close is the agent's explanation to the canonical ground truth (cosine similarity)?
- **Correction validity**: Is the suggested fix appropriate (max similarity over valid corrections)?
- **Severity**: Is the severity classification correct?

### Episode score

```
episode_score = Σ(per_error_scores) / n_errors × component_weights + efficiency_bonus
```

Grading is fully deterministic given the same embeddings model.

---

## API Reference

### Standard OpenEnv endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/reset` | Start a new episode. Body: `{"task_id": "ab_testing_easy"}` |
| POST | `/step` | Execute an action |
| GET | `/state` | Get current episode state |

### Required extra endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/tasks` | List all tasks + action/observation schemas |
| GET | `/baseline` | Run baseline agents on all tasks |
| POST | `/grader` | Get grader score for current episode |
| GET | `/health` | Server health check |

### Example: `/tasks` response

```json
{
  "tasks": [
    {
      "task_id": "ab_testing_easy",
      "title": "E-commerce Checkout Button Color Test",
      "difficulty": "easy",
      "domain": "ab_testing",
      "num_errors": 2
    },
    ...
  ],
  "total_tasks": 9,
  "action_schema": { ... },
  "observation_schema": { ... }
}
```

---

## Baseline Scores

Estimated scores across baselines (run on local development server):

| Baseline | Easy avg | Medium avg | Hard avg | Very Hard | Overall |
|----------|----------|------------|----------|-----------|---------|
| Keyword Scanner | 0.22 | 0.14 | 0.08 | 0.04 | **0.14** |
| GPT-3.5 Zero-Shot | 0.58 | 0.38 | 0.22 | 0.12 | **0.36** |
| GPT-4 Few-Shot CoT | 0.80 | 0.62 | 0.44 | 0.26 | **0.58** |

Key observations:
- **Clear difficulty progression**: scores drop monotonically across difficulty levels
- **Even frontier models struggle on very hard**: GPT-4 achieves ~26% on the bandit task
- **Keyword scanner fails on hard tasks**: complex errors require statistical reasoning, not pattern matching
- **LLM baselines show semantic understanding**: CoT reasoning substantially improves over zero-shot

---

## Project Structure

```
statistical-audit-env/
├── server/
│   ├── app.py                    # FastAPI server
│   ├── environment.py            # StatAuditEnvironment class
│   ├── Dockerfile
│   ├── tasks/
│   │   ├── __init__.py           # Task loader
│   │   ├── ab_testing/
│   │   │   ├── easy.json         # 2 errors: multiple testing + peeking
│   │   │   ├── medium.json       # 3 errors: Simpson's + survivorship + wrong test
│   │   │   └── hard.json         # 5 errors: cluster + novelty + metric switch + ...
│   │   ├── regression/
│   │   │   ├── easy.json         # 2 errors: omitted variable + heteroskedasticity
│   │   │   ├── medium.json       # 3 errors: endogeneity + collinearity + overfit
│   │   │   └── hard.json         # 5 errors: spurious + simultaneity + ...
│   │   └── causal_inference/
│   │       ├── medium.json       # 3 errors: selection + confounding + collider
│   │       ├── hard.json         # 4 errors: parallel trends + IV + SUTVA + ...
│   │       └── very_hard.json    # 6 errors: MAB-specific errors
│   ├── graders/
│   │   └── base_grader.py        # Hybrid keyword + semantic grader
│   └── baselines/
│       ├── keyword_baseline.py   # Pattern matching, no LLM
│       ├── zero_shot_llm.py      # GPT-3.5 zero-shot
│       └── few_shot_cot.py       # GPT-4 few-shot chain-of-thought
├── models.py                     # Pydantic models (Action, Observation, State)
├── inference.py                  # Baseline runner CLI
├── openenv.yaml                  # OpenEnv manifest
└── pyproject.toml                # Python dependencies
```

---

## Deployment

### Docker (local)

```bash
docker build -t stataudit -f server/Dockerfile .
docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY stataudit
```

### Hugging Face Spaces

```bash
# Tag and push to HF Spaces
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/statistical-audit-env
git push hf main
```

The space must be tagged with `openenv` for discovery.

---

## Setup & Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Start the server
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest tests/

# Validate against OpenEnv spec
openenv validate --url http://localhost:8000
```

---

## Motivation

Statistical errors are pervasive and costly:

- **Multiple testing** (p-hacking): ships features that don't actually work
- **Selection bias in A/B tests**: produces false confidence in bad decisions
- **Omitted variable bias**: regression coefficients are meaningless without proper controls
- **Simpson's paradox**: aggregate metrics reverse at the segment level — the "winning" feature hurts every user segment
- **Spurious regression**: high R² from non-stationary time series signals nothing

Training agents to catch these errors at scale could save companies millions in bad product decisions and provide a scalable check on the thousands of analyses run annually.

---

## Citation

```bibtex
@software{stataudit2025,
  title   = {StatAudit: Statistical Analysis Auditing Environment for OpenEnv},
  year    = {2025},
  url     = {https://huggingface.co/spaces/YOUR_USERNAME/statistical-audit-env},
  note    = {OpenEnv Hackathon submission}
}
```
