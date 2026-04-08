"""Shared pytest fixtures for StatAudit tests."""

import sys
import os
import pytest

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from server.app import app
from server.environment import StatAuditEnvironment
from models import Finding, StatAuditAction


@pytest.fixture(scope="session")
def client():
    """FastAPI test client (session-scoped — one server instance per test run)."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def env():
    """Fresh environment instance for each test."""
    return StatAuditEnvironment()


@pytest.fixture()
def easy_findings():
    """Correct findings for ab_testing_easy."""
    return [
        Finding(
            error_id="multiple_testing_violation",
            severity="critical",
            location="Results section",
            description=(
                "Five metrics were tested and only the best p-value reported without "
                "multiple comparison correction. Family-wise false positive rate inflates "
                "from 5% to ~23%."
            ),
            impact="The p=0.041 result is unreliable; false positive rate is ~23%, not 5%.",
            correction=(
                "Apply Bonferroni correction (alpha = 0.05/5 = 0.01) or pre-specify a "
                "single primary metric before running the test."
            ),
            confidence=0.95,
        ),
        Finding(
            error_id="early_stopping_peeking",
            severity="major",
            location="Methodology section",
            description=(
                "Test was stopped on Day 5 when significance was first observed. "
                "Early stopping without pre-specified stopping rules inflates false positive rate."
            ),
            impact="Optional stopping makes the significance level unreliable.",
            correction=(
                "Pre-specify required sample size. Use sequential testing methods "
                "such as SPRT or O'Brien-Fleming boundaries."
            ),
            confidence=0.88,
        ),
    ]
