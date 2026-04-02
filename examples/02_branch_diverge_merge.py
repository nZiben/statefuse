from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from statefuse import InMemoryStore, Memory, merge, materialize


def main() -> None:
    store_a = InMemoryStore()
    store_b = InMemoryStore()
    mem_a = Memory(store=store_a, replica_id="agentA")
    mem_b = Memory(store=store_b, replica_id="agentB")

    evidence_a = mem_a.add_evidence(pointer="doc://branchA", content="date=2026-03-25")
    evidence_b = mem_b.add_evidence(pointer="doc://branchB", content="date=2026-03-28")

    mem_a.add_claim(
        namespace="project",
        subject="deadline",
        predicate="date",
        value="2026-03-25",
        confidence=0.72,
        evidence_ids=[evidence_a],
    )
    mem_b.add_claim(
        namespace="project",
        subject="deadline",
        predicate="date",
        value="2026-03-28",
        confidence=0.76,
        evidence_ids=[evidence_b],
    )

    merged = merge(store_a.load_oplog(), store_b.load_oplog())
    merged_state = materialize(merged)
    print("Conflict count:", len(merged_state.conflicts))
    for conflict in merged_state.conflicts:
        print(json.dumps(conflict.to_dict(), indent=2))


if __name__ == "__main__":
    main()
