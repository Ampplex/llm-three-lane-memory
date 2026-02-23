"""Tests for utility helpers."""

from threelane_memory.utils import normalize_text


def test_normalize_text_collapses_whitespace():
    assert normalize_text("  hello   world  ") == "hello world"


def test_normalize_text_handles_tabs_newlines():
    assert normalize_text("hello\t\n  world") == "hello world"


def test_normalize_text_empty_string():
    assert normalize_text("") == ""


def test_normalize_text_single_word():
    assert normalize_text("  hello  ") == "hello"
