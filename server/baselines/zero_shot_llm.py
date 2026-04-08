"""
Zero-Shot LLM Baseline Agent

Uses an OpenAI-compatible chat model with a generic prompt to identify
statistical errors. No examples provided — pure zero-shot.

Expected performance: ~35–55% average score across tasks.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from models import Finding

SYSTEM_PROMPT = """You are an expert statistician and data scientist specializing in experimental design, causal inference, and statistical methodology. Your job is to audit statistical analysis reports and identify methodological errors, biases, and flaws that could lead to incorrect conclusions or business decisions.

For each error you find, you must provide:
- error_id: a snake_case identifier (e.g., "multiple_testing_violation", "selection_bias")
- severity: one of "critical", "major", or "minor"
- location: where in the report the error occurs (section name, table, etc.)
- description: what the error is — explain the methodological problem clearly
- impact: why it matters — what wrong conclusions or business decisions could result
- correction: how to fix it — specific, actionable remediation
- confidence: your confidence in this finding as a float from 0.0 to 1.0

Common error types to check for:
- Multiple testing / p-hacking (testing many outcomes and reporting the best)
- Early stopping / peeking (stopping a test when p < 0.05 without pre-specified stopping rules)
- Selection bias (non-random treatment assignment)
- Survivorship / conditioning on post-treatment variables
- Simpson's paradox (aggregate reversal of segment-level effects)
- Omitted variable bias (important confounders excluded from model)
- Heteroskedasticity (non-constant residual variance)
- Endogeneity / reverse causality (predictor caused by outcome)
- Multicollinearity (highly correlated predictors)
- Overfitting / lack of train-test split
- Spurious regression (non-stationary time series)
- Parallel trends violation (in DiD designs)
- IV exclusion restriction violation
- SUTVA / spillover / interference
- Non-stationarity in bandit environments
- Incorrect regret bounds
- Prior misspecification

Return your findings as a JSON array. Only include genuine methodological errors, not stylistic issues."""

USER_PROMPT_TEMPLATE = """Please audit the following statistical analysis report and identify all methodological errors.

REPORT:
---
{report_text}
---

Return ONLY a JSON array of findings with this structure:
[
  {{
    "error_id": "snake_case_identifier",
    "severity": "critical|major|minor",
    "location": "section name or description",
    "description": "what is wrong",
    "impact": "why it matters",
    "correction": "how to fix it",
    "confidence": 0.0
  }}
]

Return only the JSON array, no other text."""


class ZeroShotLLM:
    """Zero-shot LLM auditor using OpenAI-compatible API."""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def audit_report(
        self,
        report_text: str,
        report_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Finding]:
        """Call the LLM to audit a report and return structured findings."""
        prompt = USER_PROMPT_TEMPLATE.format(report_text=report_text)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=3000,
            )
            content = response.choices[0].message.content or ""
            return self._parse_findings(content)
        except Exception as exc:
            print(f"[ZeroShotLLM] API error: {exc}")
            return []

    def _parse_findings(self, content: str) -> List[Finding]:
        """Parse JSON findings from LLM response."""
        # Extract JSON array from response
        match = re.search(r"\[[\s\S]*\]", content)
        if not match:
            return []

        try:
            raw = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

        findings: List[Finding] = []
        for item in raw:
            try:
                severity = item.get("severity", "minor")
                if severity not in ("critical", "major", "minor"):
                    severity = "minor"

                confidence = float(item.get("confidence", 0.7))
                confidence = max(0.0, min(1.0, confidence))

                findings.append(
                    Finding(
                        error_id=str(item.get("error_id", "unknown_error")).strip(),
                        severity=severity,
                        location=str(item.get("location", "Unknown")),
                        description=str(item.get("description", "")),
                        impact=str(item.get("impact", "")),
                        correction=str(item.get("correction", "")),
                        confidence=confidence,
                    )
                )
            except Exception:
                continue

        return findings
