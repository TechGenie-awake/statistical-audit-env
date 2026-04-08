"""
Hybrid Grader for StatAudit Environment.

Combines deterministic keyword/location matching with semantic similarity
scoring using sentence-transformers. All grading is deterministic and
reproducible given the same inputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Lazy import to avoid slow startup when not grading
_sentence_model = None


def _get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sentence_model


@dataclass
class ErrorDefinition:
    """Ground truth definition for a single planted error."""

    error_id: str
    severity: str  # "critical" | "major" | "minor"
    location: str
    keywords_required: List[str]
    keywords_threshold: int
    must_mention_concepts: List[str]
    canonical_explanation: str
    valid_corrections: List[str]


def _semantic_similarity(text1: str, text2: str) -> float:
    """Cosine similarity between two text strings using sentence-transformers."""
    model = _get_sentence_model()
    emb1 = model.encode([text1])[0]
    emb2 = model.encode([text2])[0]
    norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
    if norm == 0:
        return 0.0
    return float(np.dot(emb1, emb2) / norm)


def _location_close(loc1: str, loc2: str) -> bool:
    """Return True if two location strings share meaningful keywords."""
    def tokens(s: str) -> set:
        return {t.lower() for t in re.split(r"[\s,./\-]+", s) if len(t) > 3}

    t1, t2 = tokens(loc1), tokens(loc2)
    if not t1 or not t2:
        return False
    overlap = len(t1 & t2) / min(len(t1), len(t2))
    return overlap >= 0.4


def _severity_close(sev1: str, sev2: str) -> bool:
    """Return True if severity levels are adjacent."""
    levels = ["minor", "major", "critical"]
    try:
        return abs(levels.index(sev1) - levels.index(sev2)) == 1
    except ValueError:
        return False


def grade_finding(
    finding_dict: Dict[str, Any],
    gt: ErrorDefinition,
) -> Tuple[float, Dict[str, float]]:
    """
    Grade a single Finding against a ground-truth ErrorDefinition.

    Returns (overall_score, breakdown_dict).
    """
    scores: Dict[str, float] = {
        "detection": 0.0,
        "location": 0.0,
        "keywords": 0.0,
        "explanation": 0.0,
        "correction": 0.0,
        "severity": 0.0,
    }

    # 1. Detection — did the agent identify this error at all?
    if finding_dict.get("error_id") != gt.error_id:
        return 0.0, scores  # Wrong error; everything else irrelevant

    scores["detection"] = 1.0

    # 2. Location accuracy
    agent_loc = finding_dict.get("location", "")
    if gt.location.lower() in agent_loc.lower() or agent_loc.lower() in gt.location.lower():
        scores["location"] = 1.0
    elif _location_close(agent_loc, gt.location):
        scores["location"] = 0.5

    # 3. Keyword matching
    full_text = (
        finding_dict.get("description", "") + " " + finding_dict.get("impact", "")
    ).lower()
    hits = sum(1 for kw in gt.keywords_required if kw.lower() in full_text)
    scores["keywords"] = min(1.0, hits / gt.keywords_threshold) if gt.keywords_threshold > 0 else 1.0

    # 4. Explanation quality (semantic + must-mention concepts)
    semantic_sim = _semantic_similarity(full_text, gt.canonical_explanation)
    concepts_ok = all(c.lower() in full_text for c in gt.must_mention_concepts)
    scores["explanation"] = semantic_sim if concepts_ok else semantic_sim * 0.5

    # 5. Correction validity — max similarity against any valid correction
    agent_correction = finding_dict.get("correction", "")
    if agent_correction:
        correction_scores = [
            _semantic_similarity(agent_correction, vc) for vc in gt.valid_corrections
        ]
        scores["correction"] = max(correction_scores)

    # 6. Severity classification
    agent_severity = finding_dict.get("severity", "")
    if agent_severity == gt.severity:
        scores["severity"] = 1.0
    elif _severity_close(agent_severity, gt.severity):
        scores["severity"] = 0.5

    # Weighted total
    weights = {
        "detection": 0.15,
        "location": 0.10,
        "keywords": 0.15,
        "explanation": 0.25,
        "correction": 0.20,
        "severity": 0.15,
    }
    total = sum(scores[k] * weights[k] for k in weights)
    return total, scores


def grade_episode(
    agent_findings: List[Dict[str, Any]],
    ground_truth_errors: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    """
    Grade a full episode: agent findings vs. list of ground-truth error dicts.

    Returns (total_reward ∈ [0,1], details_dict).
    """
    # Build ErrorDefinition objects from raw dicts
    gt_definitions = [
        ErrorDefinition(
            error_id=e["error_id"],
            severity=e["severity"],
            location=e["location"],
            keywords_required=e["keywords_required"],
            keywords_threshold=e["keywords_threshold"],
            must_mention_concepts=e["must_mention_concepts"],
            canonical_explanation=e["canonical_explanation"],
            valid_corrections=e["valid_corrections"],
        )
        for e in ground_truth_errors
    ]

    gt_ids = {gt.error_id for gt in gt_definitions}

    # Component reward accumulators
    n = len(gt_definitions)
    component_rewards = {
        "error_detection": 0.0,
        "severity_accuracy": 0.0,
        "explanation_quality": 0.0,
        "correction_validity": 0.0,
    }
    detailed_scores: List[Dict[str, Any]] = []

    # Index agent findings by error_id for fast lookup
    findings_by_id = {f.get("error_id"): f for f in agent_findings}

    for gt in gt_definitions:
        matching = findings_by_id.get(gt.error_id)
        if matching is None:
            detailed_scores.append({"error_id": gt.error_id, "score": 0.0, "breakdown": {}})
            continue

        score, breakdown = grade_finding(matching, gt)
        detailed_scores.append({"error_id": gt.error_id, "score": score, "breakdown": breakdown})

        if n > 0:
            component_rewards["error_detection"] += breakdown.get("detection", 0.0) * (0.40 / n)
            component_rewards["severity_accuracy"] += breakdown.get("severity", 0.0) * (0.20 / n)
            component_rewards["explanation_quality"] += breakdown.get("explanation", 0.0) * (0.20 / n)
            component_rewards["correction_validity"] += breakdown.get("correction", 0.0) * (0.15 / n)

    # Efficiency bonus (5%) — penalise false positives
    false_positives = [f for f in agent_findings if f.get("error_id") not in gt_ids]
    fp_count = len(false_positives)
    efficiency = max(0.0, 0.05 - fp_count * 0.01)

    total_reward = sum(component_rewards.values()) + efficiency

    correctly_found = sum(
        1 for d in detailed_scores if d["score"] > 0.0
    )

    return total_reward, {
        "total_reward": total_reward,
        "component_rewards": {**component_rewards, "efficiency": efficiency},
        "detailed_scores": detailed_scores,
        "errors_found": correctly_found,
        "false_positives": fp_count,
        "total_errors": n,
    }
