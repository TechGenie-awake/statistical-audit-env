"""
Keyword Baseline Agent

Simple pattern-matching scanner that flags errors based on known keyword
signatures in the report text. No LLM required.

Expected performance: ~15–25% average score across tasks.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from models import Finding


# error_id → list of (trigger_keywords, default_severity, description_template, correction_template)
ERROR_PATTERNS: Dict[str, Dict[str, Any]] = {
    "multiple_testing_violation": {
        "triggers": [
            "multiple test", "5 metrics", "five metrics", "several metrics",
            "multiple metrics", "many metrics", "different metrics", "monitored",
            "3 metrics", "four metrics", "6 metrics",
        ],
        "location_hints": ["results", "methodology", "metrics"],
        "severity": "critical",
        "description": (
            "Multiple metrics were tested without applying a multiple comparison "
            "correction, inflating the family-wise false positive rate."
        ),
        "impact": (
            "The reported p-value is unreliable; the actual false positive rate "
            "is significantly higher than the nominal alpha level."
        ),
        "correction": (
            "Apply Bonferroni correction (divide alpha by number of tests) "
            "or pre-specify a single primary metric before running the test."
        ),
    },
    "early_stopping_peeking": {
        "triggers": [
            "stopped early", "stopped when", "halted", "day 5", "day 3",
            "significance was reached", "p < 0.05 was reached", "up to",
            "monitoring", "peeking",
        ],
        "location_hints": ["methodology", "duration", "recommendation"],
        "severity": "major",
        "description": (
            "The test appears to have been stopped when statistical significance "
            "was first observed (peeking), which inflates the false positive rate."
        ),
        "impact": (
            "Optional stopping without pre-specified stopping rules makes the "
            "significance level unreliable and the result likely spurious."
        ),
        "correction": (
            "Pre-specify the required sample size or duration before starting. "
            "Use sequential testing methods (SPRT, O'Brien-Fleming) if monitoring "
            "during the test is required."
        ),
    },
    "simpsons_paradox": {
        "triggers": [
            "simpson", "all segment", "every group", "all groups",
            "age group", "overall positive", "negative in all",
            "composition", "aggregation",
        ],
        "location_hints": ["segment", "age", "subgroup"],
        "severity": "critical",
        "description": (
            "Simpson's Paradox: the overall metric is positive while all "
            "individual segments show a negative or neutral effect."
        ),
        "impact": (
            "The apparent positive result is an artifact of group composition "
            "differences, not a real treatment effect."
        ),
        "correction": (
            "Ensure groups are balanced across segments through proper randomization. "
            "Report segment-level effects as primary evidence."
        ),
    },
    "survivorship_bias": {
        "triggers": [
            "engaged users", "who engaged", "survivors", "only users",
            "active users only", "subset of users", "at least once",
            "filtered to",
        ],
        "location_hints": ["engagement", "analysis", "subset"],
        "severity": "major",
        "description": (
            "Analysis restricted to users who engaged with the feature at least once, "
            "excluding non-engagers — a survivorship bias that inflates apparent benefits."
        ),
        "impact": (
            "The effect estimates are biased upward; users who disliked or ignored "
            "the feature are excluded from the analysis."
        ),
        "correction": (
            "Use intent-to-treat analysis: include all users assigned to the treatment "
            "group, regardless of engagement."
        ),
    },
    "selection_bias": {
        "triggers": [
            "highly engaged", "engagement score", "selected customers",
            "most engaged", "non-random", "targeted", "cherry", "promoters",
        ],
        "location_hints": ["selection criteria", "methodology", "control"],
        "severity": "critical",
        "description": (
            "Treatment group was selected based on high engagement, creating a "
            "non-comparable control group and selection bias."
        ),
        "impact": (
            "The treatment and control groups differ systematically in their "
            "baseline purchase propensity, making causal estimates invalid."
        ),
        "correction": (
            "Randomize within the eligible segment. Use propensity score matching "
            "or a regression discontinuity design for causal identification."
        ),
    },
    "cluster_randomization_violated": {
        "triggers": [
            "per-impression", "impression level", "each impression",
            "independent observation", "independence", "clustering",
            "multiple impression", "same user",
        ],
        "location_hints": ["clustering", "randomization", "impressions"],
        "severity": "critical",
        "description": (
            "Each impression was treated as an independent observation, but multiple "
            "impressions per user are correlated — a cluster randomization violation."
        ),
        "impact": (
            "Standard errors are dramatically understated, making results appear "
            "far more significant than they actually are."
        ),
        "correction": (
            "Analyze at the user level or use cluster-robust standard errors "
            "clustered on user_id."
        ),
    },
    "omitted_variable_bias": {
        "triggers": [
            "location", "zip code", "neighborhood", "missing variable",
            "omitted", "not included", "not controlled", "confounder",
        ],
        "location_hints": ["model specification", "variables", "features"],
        "severity": "critical",
        "description": (
            "A major confounding variable (e.g., location/neighborhood) was omitted "
            "from the model, biasing all coefficient estimates."
        ),
        "impact": (
            "The included coefficients absorb the omitted variable's effect, making "
            "them uninterpretable and predictions systematically biased."
        ),
        "correction": (
            "Include location as a categorical fixed effect or use spatial regression "
            "methods to control for geographic variation."
        ),
    },
    "heteroskedasticity": {
        "triggers": [
            "heteroskedasticity", "heteroscedasticity", "variance",
            "residual std", "std dev", "increasing variance", "non-constant",
            "residual analysis", "std dev of residuals", "growing variance",
        ],
        "location_hints": ["residual", "variance", "diagnostic"],
        "severity": "major",
        "description": (
            "Residual variance increases systematically with the outcome, "
            "indicating heteroskedasticity that invalidates standard errors."
        ),
        "impact": (
            "Confidence intervals and significance tests are incorrect; "
            "predictions for high-value observations are unreliable."
        ),
        "correction": (
            "Use heteroskedasticity-robust (White) standard errors, or apply "
            "a log transformation to the dependent variable."
        ),
    },
    "endogeneity_reverse_causality": {
        "triggers": [
            "support ticket", "reverse causality", "endogeneity",
            "reverse causation", "causal direction", "already decided",
        ],
        "location_hints": ["key insights", "features", "coefficients"],
        "severity": "critical",
        "description": (
            "A key predictor (support tickets) is likely caused by the outcome "
            "(churn decision) rather than causing it — reverse causality."
        ),
        "impact": (
            "The coefficient on this predictor is biased and interventions based "
            "on it will not prevent churn."
        ),
        "correction": (
            "Use temporal separation (only include pre-decision tickets), "
            "or apply instrumental variables to isolate exogenous variation."
        ),
    },
    "multicollinearity": {
        "triggers": [
            "multicollinearity", "collinearity", "high correlation",
            "VIF", "0.91", "0.94", "correlated features",
        ],
        "location_hints": ["correlation", "matrix", "features"],
        "severity": "major",
        "description": (
            "Very high inter-correlations between predictors (e.g., ρ > 0.9) "
            "indicate severe multicollinearity that makes coefficients unstable."
        ),
        "impact": (
            "Individual coefficient estimates are unreliable; small data changes "
            "produce wildly different estimates."
        ),
        "correction": (
            "Compute VIF for all predictors. Remove redundant features or use "
            "Ridge regression to handle multicollinearity."
        ),
    },
    "overfitting_no_test_split": {
        "triggers": [
            "training", "R² = 0.97", "R2 = 0.97", "test set", "validation",
            "no holdout", "training data", "in-sample", "overfitting",
        ],
        "location_hints": ["model results", "performance", "training"],
        "severity": "critical",
        "description": (
            "Model performance reported only on training data (R²=0.97) with "
            "no holdout or cross-validation evaluation."
        ),
        "impact": (
            "The reported R² likely overstates true generalization performance; "
            "the model may perform poorly on new customers."
        ),
        "correction": (
            "Hold out 20-30% of data for testing before model training. "
            "Report test set performance alongside training metrics."
        ),
    },
    "spurious_regression": {
        "triggers": [
            "spurious", "non-stationary", "unit root", "stationarity",
            "trending", "time series", "ADF", "KPSS", "cointegration",
        ],
        "location_hints": ["trend", "time series", "variables"],
        "severity": "critical",
        "description": (
            "Time series variables used in levels without stationarity testing; "
            "the high R² may be a spurious regression artifact."
        ),
        "impact": (
            "Coefficients and significance tests are unreliable when non-stationary "
            "series are regressed on each other."
        ),
        "correction": (
            "Test all variables for stationarity (ADF/KPSS). Use first differences "
            "or test for cointegration before modeling."
        ),
    },
    "parallel_trends_violated": {
        "triggers": [
            "parallel trends", "pre-trend", "divergent", "different growth",
            "3.2%", "5.8%", "treatment grew", "control grew",
        ],
        "location_hints": ["parallel trends", "pre-period", "DiD"],
        "severity": "critical",
        "description": (
            "Pre-period trend analysis shows divergent growth rates between "
            "treatment and control groups, violating the DiD parallel trends assumption."
        ),
        "impact": (
            "The DiD estimate is biased; part of the post-treatment difference "
            "reflects pre-existing trend differences, not the treatment effect."
        ),
        "correction": (
            "Use synthetic control, matching on pre-period trends, or acknowledge "
            "the violated assumption and report estimates as bounds."
        ),
    },
    "iv_exclusion_restriction_violated": {
        "triggers": [
            "exclusion restriction", "instrumental variable", "distance",
            "IV assumption", "invalid instrument", "urban", "rural",
        ],
        "location_hints": ["instrumental variable", "IV", "distance"],
        "severity": "critical",
        "description": (
            "The chosen instrumental variable (distance to training center) likely "
            "violates the exclusion restriction by directly affecting earnings through "
            "channels other than program participation."
        ),
        "impact": (
            "The IV estimate is biased and inconsistent; the instrument is invalid."
        ),
        "correction": (
            "Use a valid instrument such as lottery-based allocation or a cutoff "
            "in eligibility criteria (regression discontinuity)."
        ),
    },
    "sutva_spillover_violation": {
        "triggers": [
            "spillover", "SUTVA", "network effect", "neighbors",
            "family members", "contamination", "interference",
        ],
        "location_hints": ["social network", "spillover", "neighbors"],
        "severity": "major",
        "description": (
            "Treated participants share skills and job leads with control group "
            "members, violating SUTVA and contaminating the control group."
        ),
        "impact": (
            "Control group outcomes rise due to spillover, understating the true "
            "treatment effect and biasing all causal estimates."
        ),
        "correction": (
            "Use cluster-level randomization by neighborhood to contain spillovers, "
            "or model spillover effects explicitly."
        ),
    },
    "nonstationarity_ignored": {
        "triggers": [
            "non-stationary", "world cup", "election", "varies",
            "sports ctr", "politics ctr", "temporal", "drift",
        ],
        "location_hints": ["stationarity", "time", "CTR"],
        "severity": "critical",
        "description": (
            "Arm reward distributions vary dramatically across time periods "
            "(e.g., Sports CTR during World Cup), violating the stationarity "
            "assumption of standard Thompson Sampling."
        ),
        "impact": (
            "The O(log T) regret bound does not hold; the system over-exploits "
            "stale posteriors and fails to adapt to changing environments."
        ),
        "correction": (
            "Use sliding-window or discounted Thompson Sampling that weights "
            "recent observations more heavily."
        ),
    },
}


class KeywordBaseline:
    """Keyword pattern scanner — no LLM required."""

    def audit_report(
        self,
        report_text: str,
        report_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Finding]:
        """Scan report text for error signatures and return Finding objects."""
        report_lower = report_text.lower()
        findings: List[Finding] = []

        for error_id, config in ERROR_PATTERNS.items():
            hits = sum(1 for t in config["triggers"] if t.lower() in report_lower)
            if hits < 2:
                continue  # Need at least 2 trigger hits for a finding

            # Try to locate the finding
            location = "Unspecified section"
            for hint in config["location_hints"]:
                # Find the nearest section header mentioning the hint
                pattern = rf"(?i)(#{1,3}[^\n]*{re.escape(hint)}[^\n]*)"
                m = re.search(pattern, report_text)
                if m:
                    location = m.group(1).strip("# ").strip()
                    break

            confidence = min(0.9, hits / max(len(config["triggers"]) * 0.4, 1))

            findings.append(
                Finding(
                    error_id=error_id,
                    severity=config["severity"],
                    location=location,
                    description=config["description"],
                    impact=config["impact"],
                    correction=config["correction"],
                    confidence=round(confidence, 2),
                )
            )

        return findings
