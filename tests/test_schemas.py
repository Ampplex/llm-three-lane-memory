"""Tests for schemas module — pure data structures, no external deps."""

from __future__ import annotations

from threelane_memory.schemas import SemanticExtraction


def test_semantic_extraction_required_keys():
    """SemanticExtraction TypedDict should accept valid data."""
    data: SemanticExtraction = {
        "summary": "test summary",
        "entities": ["Alice"],
        "emotion": "happy",
        "importance": 0.6,
        "actions": [],
        "states": [],
        "roles": [],
        "location": None,
    }
    assert data["summary"] == "test summary"
    assert isinstance(data["entities"], list)
    assert data["importance"] == 0.6
