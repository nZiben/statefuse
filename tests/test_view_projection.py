from __future__ import annotations

from statefuse.conflict import ConflictSet
from statefuse.materialize import materialize
from statefuse.model import Claim, ClaimKey
from statefuse.oplog import OpLog
from statefuse.ops import ClaimAdded
from statefuse.resolver import ConservativeHeuristicResolver, HeuristicResolver, Resolution, ViewConstraints
from statefuse.view import build_view


def _claim(op_id: str, claim_id: str, confidence: float, ts: str, value: str) -> ClaimAdded:
    return ClaimAdded(
        op_id=op_id,
        replica_id="replicaA",
        timestamp=ts,
        claim=Claim(
            claim_id=claim_id,
            key=ClaimKey(namespace="proj", subject="deadline", predicate="date"),
            value=value,
            confidence=confidence,
            timestamp=ts,
            evidence_ids=("sha256:x",),
            provenance={"replica_id": "replicaA"},
        ),
    )


class AbstainResolver:
    def resolve(self, conflict: ConflictSet, constraints: ViewConstraints, state):  # type: ignore[no-untyped-def]
        return Resolution(chosen_claim_id=None, reason="unable to choose")


def test_heuristic_resolver_selects_deterministically() -> None:
    oplog = OpLog(
        [
            _claim("op-1", "c1", 0.80, "2026-03-01T10:00:00.000000Z", "2026-03-25"),
            _claim("op-2", "c2", 0.90, "2026-03-01T10:00:00.000000Z", "2026-03-26"),
        ]
    )
    state = materialize(oplog)
    projection = build_view(
        state=state,
        constraints=ViewConstraints(scope="task-1"),
        resolver=HeuristicResolver(),
    )
    key = ClaimKey(namespace="proj", subject="deadline", predicate="date")
    assert projection.selected_claims[key].claim_id == "c2"
    assert projection.unresolved_conflicts == []
    assert key in projection.surfaced_conflicts
    assert "deterministic heuristic" in projection.explanations["proj:deadline:date"]


def test_unresolved_conflict_is_preserved() -> None:
    oplog = OpLog(
        [
            _claim("op-1", "c1", 0.80, "2026-03-01T10:00:00.000000Z", "2026-03-25"),
            _claim("op-2", "c2", 0.90, "2026-03-01T10:00:00.000000Z", "2026-03-26"),
        ]
    )
    state = materialize(oplog)
    projection = build_view(
        state=state,
        constraints=ViewConstraints(scope="task-2"),
        resolver=AbstainResolver(),
    )
    assert len(projection.unresolved_conflicts) == 1
    assert projection.selected_claims == {}


def test_build_view_does_not_mutate_materialized_state() -> None:
    oplog = OpLog(
        [
            _claim("op-1", "c1", 0.80, "2026-03-01T10:00:00.000000Z", "2026-03-25"),
            _claim("op-2", "c2", 0.90, "2026-03-01T10:00:00.000000Z", "2026-03-26"),
        ]
    )
    state = materialize(oplog)
    before_claim_ids = {
        key: tuple(claim.claim_id for claim in claims) for key, claims in state.active_claims_by_key.items()
    }
    before_conflicts = tuple(conflict.conflict_id for conflict in state.conflicts)

    _ = build_view(
        state=state,
        constraints=ViewConstraints(scope="task-3"),
        resolver=HeuristicResolver(),
    )

    after_claim_ids = {
        key: tuple(claim.claim_id for claim in claims) for key, claims in state.active_claims_by_key.items()
    }
    after_conflicts = tuple(conflict.conflict_id for conflict in state.conflicts)
    assert after_claim_ids == before_claim_ids
    assert after_conflicts == before_conflicts


def test_conservative_resolver_abstains_on_symmetric_conflict() -> None:
    oplog = OpLog(
        [
            _claim("op-1", "c1", 0.80, "2026-03-01T10:00:00.000000Z", "2026-03-25"),
            _claim("op-2", "c2", 0.80, "2026-03-01T10:00:01.000000Z", "2026-03-26"),
        ]
    )
    state = materialize(oplog)
    projection = build_view(
        state=state,
        constraints=ViewConstraints(scope="task-4"),
        resolver=ConservativeHeuristicResolver(),
    )
    assert len(projection.unresolved_conflicts) == 1
    assert projection.selected_claims == {}
