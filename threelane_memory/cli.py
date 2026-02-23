"""CLI interface for threelane-memory.

Usage::

    threelane-memory chat                        # interactive chat
    threelane-memory store "I met Alice today"   # store a single memory
    threelane-memory query "Who did I meet?"     # query memories
    threelane-memory config                      # show active provider & config
    threelane-memory backup                      # export graph to JSON
    threelane-memory dedup                       # merge duplicate entities
    threelane-memory dedup --dry-run             # preview dedup merges
    threelane-memory consolidate                 # merge old low-importance episodes
    threelane-memory reindex                     # re-embed episodes after model change
"""

from __future__ import annotations

import argparse
import sys


def _preflight_check() -> None:
    """Run a dimension-mismatch check and warn the user if necessary."""
    from threelane_memory.database import check_index_dimension
    check_index_dimension()


def _cmd_chat(args: argparse.Namespace) -> None:
    """Launch interactive chat loop."""
    _preflight_check()
    from threelane_memory.chat import main as chat_main

    chat_main(speaker=args.speaker)


def _cmd_store(args: argparse.Namespace) -> None:
    """Store a single memory from the command line."""
    _preflight_check()
    from threelane_memory import store

    episode_id = store(args.text, speaker=args.speaker)
    print(f"Stored episode {episode_id}")


def _cmd_query(args: argparse.Namespace) -> None:
    """Query memories from the command line."""
    _preflight_check()
    from threelane_memory import query

    answer = query(args.question, speaker=args.speaker)
    print(answer)


def _cmd_config(args: argparse.Namespace) -> None:
    """Show active provider configuration and run health checks."""
    from threelane_memory.config import get_provider_summary, SUPPORTED_PROVIDERS
    from threelane_memory.database import get_index_dimension, check_index_dimension

    info = get_provider_summary()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║              threelane-memory · Configuration               ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Provider        : {info['provider']:<40} ║")
    print(f"║  Chat model      : {info['chat_model']:<40} ║")
    print(f"║  Embed model     : {info['embed_model']:<40} ║")
    print(f"║  Embedding dim   : {str(info['embedding_dim']):<40} ║")
    print(f"║  Neo4j URI       : {info['neo4j_uri'][:40]:<40} ║")
    print("╠══════════════════════════════════════════════════════════════╣")

    # Neo4j connectivity
    try:
        from threelane_memory.database import run_query
        run_query("RETURN 1 AS ok")
        print("║  Neo4j status    : ✅ connected                            ║")
    except Exception as e:
        msg = str(e)[:35]
        print(f"║  Neo4j status    : ❌ {msg:<38} ║")

    # Vector index dimension
    idx_dim = get_index_dimension()
    if idx_dim is None:
        print("║  Vector index    : ⚠  not found (will auto-create)        ║")
    elif idx_dim == info["embedding_dim"]:
        print(f"║  Vector index    : ✅ {idx_dim}-dim (matches config)          {'   ' if idx_dim < 1000 else '  '}║")
    else:
        print(f"║  Vector index    : ❌ {idx_dim}-dim (config expects {info['embedding_dim']})      ║")

    # Ollama connectivity (if applicable)
    if info["provider"] == "ollama":
        try:
            import urllib.request
            from threelane_memory.config import OLLAMA_BASE_URL
            req = urllib.request.urlopen(OLLAMA_BASE_URL, timeout=3)
            print("║  Ollama status   : ✅ running                             ║")
        except Exception:
            print("║  Ollama status   : ❌ not reachable (is ollama serve on?)  ║")

    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # Detailed mismatch warning (if any)
    check_index_dimension()

    # Switching help
    other = [p for p in SUPPORTED_PROVIDERS if p != info["provider"]]
    if other:
        print(f"  To switch to {other[0]}, see: .env.example")
        print(f"  After switching, run: threelane-memory config")
        print()

    # OpenAI key check
    if info["provider"] == "openai":
        from threelane_memory.config import OPENAI_API_KEY
        if not OPENAI_API_KEY:
            print("  ⚠  OPENAI_API_KEY is empty — set it in .env")
            print()


def _cmd_backup(args: argparse.Namespace) -> None:
    """Export the graph to JSON."""
    from threelane_memory.backup import save_backup

    path = save_backup(speaker=args.speaker, since=args.since)
    print(f"Done: {path}")


def _cmd_dedup(args: argparse.Namespace) -> None:
    """Run entity deduplication."""
    from threelane_memory.entity_dedup import deduplicate_entities

    result = deduplicate_entities(dry_run=args.dry_run)
    mode = "DRY-RUN" if args.dry_run else "APPLIED"
    print(f"[{mode}] Duplicates found: {result['duplicates_found']}, "
          f"Merged: {result['merged']}")


def _cmd_consolidate(args: argparse.Namespace) -> None:
    """Consolidate old low-importance episodes."""
    from threelane_memory.reconciler import consolidate

    result = consolidate(args.speaker)
    if result["merged"]:
        print(f"Merged {result['merged']} episodes → {result['consolidated_episode_id']}")
    else:
        print("Nothing to consolidate right now.")


def _cmd_reindex(args: argparse.Namespace) -> None:
    """Re-embed episodes after changing embedding model."""
    from threelane_memory.reconciler import reindex_embeddings

    count = reindex_embeddings(args.speaker, old_model=args.old_model)
    print(f"Re-indexed {count} episodes")


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``threelane-memory`` CLI."""
    parser = argparse.ArgumentParser(
        prog="threelane-memory",
        description="Personal long-term memory on Neo4j — supports Ollama (local) and OpenAI",
    )
    parser.add_argument(
        "--version", action="version",
        version="%(prog)s 0.1.0",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── config ────────────────────────────────────────────────────────────
    p_config = sub.add_parser("config",
                              help="Show active provider, models & health checks")
    p_config.set_defaults(func=_cmd_config)

    # ── chat ──────────────────────────────────────────────────────────────
    p_chat = sub.add_parser("chat", help="Interactive memory chat")
    p_chat.add_argument("--speaker", default="default",
                        help="Speaker identity (default: 'default')")
    p_chat.set_defaults(func=_cmd_chat)

    # ── store ─────────────────────────────────────────────────────────────
    p_store = sub.add_parser("store", help="Store a single memory")
    p_store.add_argument("text", help="The memory text to store")
    p_store.add_argument("--speaker", default="default",
                         help="Speaker identity (default: 'default')")
    p_store.set_defaults(func=_cmd_store)

    # ── query ─────────────────────────────────────────────────────────────
    p_query = sub.add_parser("query", help="Query memories")
    p_query.add_argument("question", help="The question to ask")
    p_query.add_argument("--speaker", default="default",
                         help="Speaker identity (default: 'default')")
    p_query.set_defaults(func=_cmd_query)

    # ── backup ────────────────────────────────────────────────────────────
    p_backup = sub.add_parser("backup", help="Export graph to JSON")
    p_backup.add_argument("--speaker", default=None,
                          help="Filter by speaker")
    p_backup.add_argument("--since", default=None,
                          help="Export episodes since this ISO date (e.g. 2025-01-01)")
    p_backup.set_defaults(func=_cmd_backup)

    # ── dedup ─────────────────────────────────────────────────────────────
    p_dedup = sub.add_parser("dedup", help="Merge duplicate entities")
    p_dedup.add_argument("--dry-run", action="store_true",
                         help="Preview merges without applying")
    p_dedup.set_defaults(func=_cmd_dedup)

    # ── consolidate ───────────────────────────────────────────────────────
    p_consolidate = sub.add_parser("consolidate",
                                   help="Merge old low-importance episodes")
    p_consolidate.add_argument("--speaker", default="default",
                               help="Speaker identity (default: 'default')")
    p_consolidate.set_defaults(func=_cmd_consolidate)

    # ── reindex ───────────────────────────────────────────────────────────
    p_reindex = sub.add_parser("reindex",
                               help="Re-embed episodes after model change")
    p_reindex.add_argument("--speaker", default="default",
                           help="Speaker identity (default: 'default')")
    p_reindex.add_argument("--old-model", required=True,
                           help="Previous embedding model version to re-index")
    p_reindex.set_defaults(func=_cmd_reindex)

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        from threelane_memory.database import close
        close()


if __name__ == "__main__":
    main()
