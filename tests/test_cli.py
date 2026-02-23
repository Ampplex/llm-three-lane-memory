"""Tests for the CLI module — argument parsing only, no external deps."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from threelane_memory.cli import main


def test_cli_no_args_exits(capsys):
    """Running with no subcommand should print help and exit."""
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code == 1


def test_cli_version(capsys):
    """--version should print version and exit."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "0.1.0" in captured.out


def test_cli_store_requires_text():
    """'store' subcommand should require text argument."""
    with pytest.raises(SystemExit):
        main(["store"])


def test_cli_query_requires_question():
    """'query' subcommand should require question argument."""
    with pytest.raises(SystemExit):
        main(["query"])


def test_cli_reindex_requires_old_model():
    """'reindex' subcommand should require --old-model."""
    with pytest.raises(SystemExit):
        main(["reindex"])
