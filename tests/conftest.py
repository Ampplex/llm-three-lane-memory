"""Test configuration and shared fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_extraction() -> dict:
    """A realistic SemanticExtraction for testing."""
    return {
        "summary": "Speaker's dog Max is 3 years old and is a golden retriever",
        "entities": ["Max", "speaker"],
        "emotion": "neutral",
        "importance": 0.5,
        "actions": [{"verb": "is", "actor": "Max", "object": None}],
        "states": [
            {"entity": "Max", "attribute": "age", "value": "3 years old"},
            {"entity": "Max", "attribute": "breed", "value": "golden retriever"},
        ],
        "roles": [{"entity": "Max", "role": "pet"}],
        "location": None,
    }


@pytest.fixture
def sample_question() -> str:
    return "How old is Max?"
