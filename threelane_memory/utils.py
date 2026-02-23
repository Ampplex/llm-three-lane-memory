"""Shared utility helpers."""


def normalize_text(text: str) -> str:
    """Strip and collapse whitespace."""
    return " ".join(text.split())
