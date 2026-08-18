from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from statefuse import (  # noqa: E402
    Claim,
    ClaimAdded,
    ClaimKey,
    OpLog,
    ResolutionContext,
    ResolverRegistry,
    SelectedState,
    UnresolvedConflict,
    materialize,
)


def main() -> None:
    key = ClaimKey(namespace="project", subject="release", predicate="status")
    claims = [
        Claim(
            claim_id="c1",
            key=key,
            value="draft",
            confidence=0.8,
            timestamp="2026-03-01T00:00:00Z",
            provenance={"replica_id": "a", "branch_id": "main"},
        ),
        Claim(
            claim_id="c2",
            key=key,
            value="ready",
            confidence=0.8,
            timestamp="2026-03-01T00:01:00Z",
            provenance={"replica_id": "a", "branch_id": "main"},
        ),
    ]
    oplog = OpLog(
        ClaimAdded(
            op_id=f"op-{claim.claim_id}",
            replica_id="a",
            timestamp=claim.timestamp,
            claim=claim,
        )
        for claim in claims
    )
    state = materialize(oplog)
    conflict = state.conflicts[0]
    original_ops = oplog.op_ids()
    original_snapshot = conflict.to_dict()
    registry = ResolverRegistry()

    default_result = registry.resolve(conflict, ResolutionContext())
    selected_result = registry.resolve(
        conflict,
        ResolutionContext(
            timestamps_trusted=True,
            concurrent=False,
            replacement_semantics=True,
        ),
        resolver_name="latest_write_wins",
    )

    assert isinstance(default_result, UnresolvedConflict)
    assert isinstance(selected_result, SelectedState)
    assert selected_result.selected_claim.claim_id == "c2"
    assert selected_result.conflict_set is conflict
    assert conflict.to_dict() == original_snapshot
    assert oplog.op_ids() == original_ops
    print("default:", type(default_result).__name__)
    print("opt-in selection:", selected_result.selected_claim.claim_id)
    print("original conflict and operation history preserved")


if __name__ == "__main__":
    main()
