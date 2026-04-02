from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from statefuse import JsonlStore, Memory, materialize


def main() -> None:
    store = JsonlStore(pathlib.Path("tmp/basic_ops.jsonl"))
    mem = Memory(store=store, replica_id="agentA")

    evidence_id = mem.add_evidence(
        pointer="https://example.com/spec",
        content="Deadline is 2026-03-25",
        metadata={"tool": "web", "summary": "Product spec"},
    )
    mem.add_claim(
        namespace="project",
        subject="deadline",
        predicate="date",
        value="2026-03-25",
        confidence=0.8,
        evidence_ids=[evidence_id],
    )

    state = materialize(store.load_oplog())
    print("evidence_ids:", sorted(state.evidence_by_id))
    print("active_claim_keys:", [key.to_dict() for key in state.active_claims_by_key])
    print("conflicts:", json.dumps([conflict.to_dict() for conflict in state.conflicts], indent=2))


if __name__ == "__main__":
    main()
