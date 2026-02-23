"""Tests for the operator module (semantic extraction).

These tests require OPENAI_API_KEY and make real LLM calls.
Mark them with ``@pytest.mark.integration`` and skip in CI by default.
"""

from __future__ import annotations

import json

import pytest


@pytest.mark.integration
def test_operator_extract_returns_valid_structure():
    """operator_extract should return all required keys."""
    from threelane_memory.operator import operator_extract

    result = operator_extract("My dog Max is 3 years old")

    required_keys = {
        "summary", "entities", "emotion", "importance",
        "actions", "states", "roles", "location",
    }
    assert required_keys.issubset(result.keys())
    assert isinstance(result["entities"], list)
    assert len(result["entities"]) > 0
    assert 0 <= result["importance"] <= 1


@pytest.mark.integration
def test_operator_extract_entity_detection():
    """Entities should be extracted from input text."""
    from threelane_memory.operator import operator_extract

    result = operator_extract("Alice met Bob at the park")
    entities_lower = [e.lower() for e in result["entities"]]
    assert "alice" in entities_lower
    assert "bob" in entities_lower
