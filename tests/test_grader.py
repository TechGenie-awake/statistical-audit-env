"""Tests for the hybrid grader."""

import pytest
from server.graders.base_grader import (
    grade_episode,
    grade_finding,
    ErrorDefinition,
    _location_close,
    _severity_close,
    _semantic_similarity,
)

# Minimal ground-truth fixture (no sentence-transformers needed for unit tests)
SIMPLE_GT = ErrorDefinition(
    error_id="multiple_testing_violation",
    severity="critical",
    location="Results section",
    keywords_required=["multiple testing", "bonferroni", "false positive rate"],
    keywords_threshold=2,
    must_mention_concepts=["multiple", "testing"],
    canonical_explanation=(
        "Five metrics were tested without correction, inflating the family-wise "
        "false positive rate from 5% to 22.6%. Bonferroni correction is required."
    ),
    valid_corrections=[
        "Apply Bonferroni correction: require p < 0.05/5 = 0.01",
        "Pre-specify a single primary metric before running the test",
    ],
)


class TestLocationClose:
    def test_identical_locations(self):
        assert _location_close("Results section", "Results section") is True

    def test_overlapping_tokens(self):
        assert _location_close("Results section", "section methodology") is True

    def test_no_overlap(self):
        assert _location_close("Appendix", "Introduction methodology") is False

    def test_short_tokens_ignored(self):
        # Tokens <= 3 chars are filtered
        assert _location_close("the in a", "foo bar baz") is False


class TestSeverityClose:
    def test_exact_match_not_adjacent(self):
        assert _severity_close("critical", "critical") is False  # diff = 0

    def test_adjacent_levels(self):
        assert _severity_close("critical", "major") is True
        assert _severity_close("major", "minor") is True

    def test_non_adjacent(self):
        assert _severity_close("critical", "minor") is False  # diff = 2

    def test_invalid_severity(self):
        assert _severity_close("invalid", "critical") is False


class TestGradeFinding:
    def test_wrong_error_id_scores_zero(self):
        finding = {
            "error_id": "wrong_error_id",
            "severity": "critical",
            "location": "Results section",
            "description": "multiple testing bonferroni false positive rate",
            "impact": "inflated false positive rate",
            "correction": "Apply Bonferroni correction",
            "confidence": 0.9,
        }
        score, breakdown = grade_finding(finding, SIMPLE_GT)
        assert score == 0.0
        assert breakdown["detection"] == 0.0

    def test_correct_error_id_gets_detection_score(self):
        finding = {
            "error_id": "multiple_testing_violation",
            "severity": "critical",
            "location": "Results section",
            "description": "multiple testing bonferroni false positive rate",
            "impact": "inflated rate",
            "correction": "Apply Bonferroni correction pre-specify primary metric",
            "confidence": 0.9,
        }
        score, breakdown = grade_finding(finding, SIMPLE_GT)
        assert breakdown["detection"] == 1.0
        assert score > 0.0

    def test_correct_severity_scores_one(self):
        finding = {
            "error_id": "multiple_testing_violation",
            "severity": "critical",
            "location": "Results section",
            "description": "multiple testing false positive rate inflated",
            "impact": "high rate",
            "correction": "Bonferroni",
            "confidence": 0.9,
        }
        _, breakdown = grade_finding(finding, SIMPLE_GT)
        assert breakdown["severity"] == 1.0

    def test_adjacent_severity_scores_half(self):
        finding = {
            "error_id": "multiple_testing_violation",
            "severity": "major",  # adjacent to critical
            "location": "Results section",
            "description": "multiple testing false positive rate",
            "impact": "high",
            "correction": "Bonferroni correction",
            "confidence": 0.9,
        }
        _, breakdown = grade_finding(finding, SIMPLE_GT)
        assert breakdown["severity"] == 0.5

    def test_wrong_severity_scores_zero(self):
        finding = {
            "error_id": "multiple_testing_violation",
            "severity": "minor",  # two away from critical
            "location": "Results section",
            "description": "multiple testing false positive rate",
            "impact": "high",
            "correction": "Bonferroni",
            "confidence": 0.9,
        }
        _, breakdown = grade_finding(finding, SIMPLE_GT)
        assert breakdown["severity"] == 0.0

    def test_keyword_coverage(self):
        # description contains 2 of 3 required keywords — threshold is 2, so should be >= 1.0
        finding = {
            "error_id": "multiple_testing_violation",
            "severity": "critical",
            "location": "Results section",
            "description": "multiple testing without correction inflates the false positive rate",
            "impact": "bonferroni needed",
            "correction": "fix it",
            "confidence": 0.9,
        }
        _, breakdown = grade_finding(finding, SIMPLE_GT)
        # keywords: "multiple testing" ✓, "bonferroni" ✓ (in impact), "false positive rate" ✓
        assert breakdown["keywords"] >= 1.0

    def test_location_exact_match(self):
        finding = {
            "error_id": "multiple_testing_violation",
            "severity": "critical",
            "location": "Results section",
            "description": "multiple testing bonferroni false positive rate",
            "impact": "bad",
            "correction": "fix",
            "confidence": 0.9,
        }
        _, breakdown = grade_finding(finding, SIMPLE_GT)
        assert breakdown["location"] == 1.0

    def test_location_close_match(self):
        finding = {
            "error_id": "multiple_testing_violation",
            "severity": "critical",
            "location": "Results methodology section",  # overlapping
            "description": "multiple testing bonferroni false positive rate",
            "impact": "bad",
            "correction": "fix",
            "confidence": 0.9,
        }
        _, breakdown = grade_finding(finding, SIMPLE_GT)
        assert breakdown["location"] >= 0.5

    def test_score_is_bounded_01(self):
        finding = {
            "error_id": "multiple_testing_violation",
            "severity": "critical",
            "location": "Results section",
            "description": "multiple testing bonferroni false positive rate correction",
            "impact": "Bonferroni correction needed multiple testing false positive",
            "correction": "Apply Bonferroni correction require p < 0.05/5 = 0.01",
            "confidence": 0.95,
        }
        score, _ = grade_finding(finding, SIMPLE_GT)
        assert 0.0 <= score <= 1.0


class TestGradeEpisode:
    def _make_gt(self, error_id="err1", severity="critical"):
        return {
            "error_id": error_id,
            "severity": severity,
            "location": "Results section",
            "keywords_required": ["keyword_a", "keyword_b"],
            "keywords_threshold": 1,
            "must_mention_concepts": ["keyword"],
            "canonical_explanation": "keyword_a and keyword_b are present in analysis.",
            "valid_corrections": ["Fix by applying standard correction method."],
        }

    def test_empty_findings_gives_efficiency_only(self):
        gt = [self._make_gt()]
        reward, details = grade_episode([], gt)
        # No findings = no detection; efficiency = 0.05 (no false positives)
        assert details["errors_found"] == 0
        assert details["false_positives"] == 0
        assert reward == pytest.approx(0.05, abs=0.01)

    def test_false_positive_reduces_efficiency(self):
        gt = [self._make_gt()]
        findings = [
            {
                "error_id": "nonexistent_error",
                "severity": "critical",
                "location": "Results section",
                "description": "something",
                "impact": "bad",
                "correction": "fix it",
                "confidence": 0.5,
            }
        ]
        reward, details = grade_episode(findings, gt)
        assert details["false_positives"] == 1
        assert details["errors_found"] == 0
        # efficiency = max(0, 0.05 - 1*0.01) = 0.04
        assert details["component_rewards"]["efficiency"] == pytest.approx(0.04, abs=0.001)

    def test_five_false_positives_zero_efficiency(self):
        gt = [self._make_gt()]
        findings = [
            {"error_id": f"fake_{i}", "severity": "minor", "location": "x",
             "description": "x", "impact": "x", "correction": "x", "confidence": 0.5}
            for i in range(5)
        ]
        reward, details = grade_episode(findings, gt)
        assert details["component_rewards"]["efficiency"] == 0.0

    def test_total_reward_bounded_01(self):
        gt = [self._make_gt()]
        findings = [
            {
                "error_id": "err1",
                "severity": "critical",
                "location": "Results section",
                "description": "keyword_a and keyword_b present in analysis",
                "impact": "keyword_a matters",
                "correction": "Fix by applying standard correction method.",
                "confidence": 0.95,
            }
        ]
        reward, _ = grade_episode(findings, gt)
        assert 0.0 <= reward <= 1.0

    def test_multiple_errors_all_found(self):
        gt = [self._make_gt("err1"), self._make_gt("err2", "major")]
        findings = [
            {
                "error_id": "err1", "severity": "critical",
                "location": "Results section",
                "description": "keyword_a keyword_b present",
                "impact": "keyword_a matters", "correction": "Fix by applying standard correction method.",
                "confidence": 0.9,
            },
            {
                "error_id": "err2", "severity": "major",
                "location": "Results section",
                "description": "keyword_a keyword_b present",
                "impact": "keyword_a matters", "correction": "Fix by applying standard correction method.",
                "confidence": 0.9,
            },
        ]
        reward, details = grade_episode(findings, gt)
        assert details["errors_found"] == 2
        assert details["false_positives"] == 0

    def test_details_structure(self):
        gt = [self._make_gt()]
        reward, details = grade_episode([], gt)
        assert "total_reward" in details
        assert "component_rewards" in details
        assert "detailed_scores" in details
        assert "errors_found" in details
        assert "false_positives" in details
        assert "total_errors" in details
        cr = details["component_rewards"]
        for key in ("error_detection", "severity_accuracy", "explanation_quality",
                    "correction_validity", "efficiency"):
            assert key in cr


class TestSemanticSimilarity:
    def test_identical_texts_score_near_one(self):
        sim = _semantic_similarity("hello world", "hello world")
        assert sim > 0.99

    def test_different_texts_score_lower(self):
        sim = _semantic_similarity("hello world", "statistical regression analysis")
        assert sim < 0.8

    def test_similar_meaning_scores_higher_than_unrelated(self):
        related = _semantic_similarity(
            "Apply Bonferroni correction to control false positive rate.",
            "Use Bonferroni to fix the multiple testing false positive problem.",
        )
        unrelated = _semantic_similarity(
            "Apply Bonferroni correction to control false positive rate.",
            "The weather is sunny today in the park.",
        )
        assert related > unrelated
