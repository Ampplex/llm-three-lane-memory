"""Advanced usage — direct access to the operator/reconciler/retriever pipeline.

Demonstrates how to use the lower-level API for fine-grained control.
"""

import json

from threelane_memory.operator import operator_extract
from threelane_memory.reconciler import reconcile, consolidate
from threelane_memory.retriever import retrieve
from threelane_memory.llm_interface import invoke_llm
from threelane_memory.backup import save_backup
from threelane_memory.entity_dedup import deduplicate_entities
from threelane_memory.database import close


def main() -> None:
    # ── Step 1: Semantic extraction ──────────────────────────────────────
    text = "Alice presented her research paper at MIT while Bob felt stressed."
    print(f"Input: {text}\n")

    semantics = operator_extract(text)
    print("Extracted semantics:")
    print(json.dumps(semantics, indent=2))

    # ── Step 2: Write to graph ───────────────────────────────────────────
    episode_id = reconcile(semantics, speaker="demo", raw_text=text)
    print(f"\nStored episode: {episode_id}")

    # ── Step 3: Retrieve context ─────────────────────────────────────────
    context = retrieve("What did Alice do?", speaker="demo")
    print(f"\nRetrieved context:\n{context}")

    # ── Step 4: Answer from context ──────────────────────────────────────
    answer = invoke_llm(
        f"Answer from context:\n{context}\n\nQuestion: What did Alice do?"
    )
    print(f"\nAnswer: {answer}")

    # ── Admin operations ─────────────────────────────────────────────────

    # Run entity deduplication (dry run)
    print("\n--- Entity dedup (dry run) ---")
    dedup_result = deduplicate_entities(dry_run=True)
    print(f"Duplicates found: {dedup_result['duplicates_found']}")

    # Consolidate old episodes
    print("\n--- Consolidation ---")
    consol_result = consolidate("demo")
    print(f"Merged: {consol_result['merged']}")

    # Backup
    print("\n--- Backup ---")
    backup_path = save_backup(speaker="demo")
    print(f"Saved: {backup_path}")

    close()


if __name__ == "__main__":
    main()
