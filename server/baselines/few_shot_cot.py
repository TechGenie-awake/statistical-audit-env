"""
Few-Shot Chain-of-Thought LLM Baseline Agent

Uses an OpenAI-compatible chat model with worked examples and chain-of-thought
reasoning to identify statistical errors. The most capable baseline.

Expected performance: ~55–75% average score across tasks.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from models import Finding

SYSTEM_PROMPT = """You are a world-class statistician and methodologist specializing in experimental design, causal inference, and data analysis review. You have reviewed thousands of statistical reports and have a keen eye for subtle methodological errors.

Your task is to systematically audit a statistical analysis report using step-by-step reasoning, then produce structured findings for every methodological error you identify.

You will think through the analysis carefully before finalizing your findings."""

FEW_SHOT_EXAMPLES = """
## WORKED EXAMPLE 1

REPORT EXCERPT:
"We tested 8 different UI layouts and found layout #3 had p = 0.042 for click-through rate."

STEP-BY-STEP REASONING:
1. Statistical methods used: t-test on CTR across 8 layouts
2. Assumptions: each test at alpha = 0.05
3. Problem identified: Testing 8 layouts = 8 statistical tests without correction
4. Consequence: Family-wise false positive rate ≈ 1 - 0.95^8 = 33.7%, not 5%
5. Conclusion: This is a multiple testing violation — p = 0.042 is not reliable

FINDING:
{
  "error_id": "multiple_testing_violation",
  "severity": "critical",
  "location": "Results section",
  "description": "Eight UI layouts were compared and only the best p-value (p=0.042) was reported, without applying multiple comparison correction.",
  "impact": "The family-wise false positive rate inflates to ~34% (not 5%), making the p=0.042 result unreliable and likely spurious.",
  "correction": "Apply Bonferroni correction (require p < 0.05/8 = 0.006), or pre-specify a single layout as primary before the test.",
  "confidence": 0.95
}

---

## WORKED EXAMPLE 2

REPORT EXCERPT:
"We sent promotional emails to our top 10% most loyal customers and found they converted at 3x the rate of non-email recipients."

STEP-BY-STEP REASONING:
1. Study design: Emails sent to top 10% loyal customers (non-random)
2. Control group: Non-loyal customers who didn't receive email
3. Problem: Loyal customers buy more even without the email — pre-existing difference
4. Consequence: The 3x conversion difference reflects customer loyalty, not email effect
5. Conclusion: Selection bias — non-comparable treatment and control groups

FINDING:
{
  "error_id": "selection_bias",
  "severity": "critical",
  "location": "Study design / email targeting",
  "description": "Emails were sent to the most loyal 10% of customers, while the control group consists of less loyal customers. These groups have fundamentally different baseline conversion rates.",
  "impact": "The observed 3x conversion lift reflects pre-existing loyalty differences, not the causal effect of the email campaign. The estimate is massively overstated.",
  "correction": "Randomize within the loyal segment: send emails to a random 50% of top customers and use the other 50% as the control group.",
  "confidence": 0.97
}

---

## WORKED EXAMPLE 3

REPORT EXCERPT:
"GDP, unemployment rate, and inflation all have strong upward trends from 1990-2023. Our regression shows R² = 0.89 and unemployment significantly predicts GDP."

STEP-BY-STEP REASONING:
1. Statistical methods: OLS regression on time series in levels
2. Data: All variables trend upward over 33 years
3. Problem: Non-stationary time series regressed on each other without stationarity check
4. Consequence: Spurious regression — high R² is artifact of shared trend, not real relationship
5. Additionally: Even if real, correlation ≠ causation; GDP may cause unemployment (Okun's Law)
6. Conclusions: Two errors — spurious regression AND potential reverse causality

"""

AUDIT_PROMPT_TEMPLATE = """Please audit the following statistical analysis report. Work through the analysis systematically before giving your final findings.

REPORT:
---
{report_text}
---

ADDITIONAL CONTEXT:
{metadata_text}

Use this step-by-step process:
1. Identify the statistical method(s) used
2. List the key assumptions each method makes
3. Check each assumption: is it satisfied or violated?
4. Identify any confounders, biases, or design flaws
5. Check for issues with: randomization, causality, multiple testing, model specification, data quality

After your reasoning, output your final findings as a JSON array:

FINDINGS:
[
  {{
    "error_id": "snake_case_identifier",
    "severity": "critical|major|minor",
    "location": "specific section or table name",
    "description": "precise explanation of the methodological error",
    "impact": "concrete business/scientific consequences of this error",
    "correction": "specific, actionable remediation steps",
    "confidence": 0.0
  }}
]

Be thorough but precise. Only flag genuine methodological errors, not stylistic issues.
Output the FINDINGS JSON array at the end of your response."""


class FewShotCoT:
    """Few-shot chain-of-thought LLM auditor."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def audit_report(
        self,
        report_text: str,
        report_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Finding]:
        """Call LLM with few-shot examples + CoT to audit a report."""
        metadata_text = ""
        if report_metadata:
            metadata_text = "\n".join(
                f"- {k}: {v}" for k, v in report_metadata.items()
            )

        prompt = AUDIT_PROMPT_TEMPLATE.format(
            report_text=report_text,
            metadata_text=metadata_text or "No additional metadata provided.",
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": FEW_SHOT_EXAMPLES},
                    {"role": "assistant", "content": "I've studied these examples. I'll apply the same systematic approach to the next report."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=4000,
            )
            content = response.choices[0].message.content or ""
            return self._parse_findings(content)
        except Exception as exc:
            print(f"[FewShotCoT] API error: {exc}")
            return []

    def _parse_findings(self, content: str) -> List[Finding]:
        """Extract JSON findings array from LLM response."""
        # Find the last JSON array in the response (after chain-of-thought)
        all_arrays = re.findall(r"\[[\s\S]*?\]", content)
        if not all_arrays:
            return []

        # Use the last/largest array (the final findings, not intermediate examples)
        raw_json = max(all_arrays, key=len)

        try:
            raw = json.loads(raw_json)
        except json.JSONDecodeError:
            # Try to fix common LLM JSON formatting issues
            cleaned = re.sub(r",\s*\]", "]", raw_json)  # trailing comma
            try:
                raw = json.loads(cleaned)
            except json.JSONDecodeError:
                return []

        findings: List[Finding] = []
        for item in raw:
            try:
                severity = item.get("severity", "minor")
                if severity not in ("critical", "major", "minor"):
                    severity = "minor"

                confidence = float(item.get("confidence", 0.75))
                confidence = max(0.0, min(1.0, confidence))

                error_id = str(item.get("error_id", "unknown_error")).strip()
                # Normalize error_id to snake_case
                error_id = re.sub(r"[^a-z0-9_]", "_", error_id.lower())
                error_id = re.sub(r"_+", "_", error_id).strip("_")

                findings.append(
                    Finding(
                        error_id=error_id,
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
