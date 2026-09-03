from __future__ import annotations

from collections.abc import Iterable

from statefuse.conflict import ConflictDetectionContext, ConflictSet, make_conflict
from statefuse.materialize import materialize
from statefuse.model import Claim, ClaimKey, ResolutionRecord
from statefuse.oplog import OpLog
from statefuse.ops import ClaimAdded, ResolutionAdded
from statefuse.resolver import (
    ConservativeHeuristicResolver,
    HeuristicResolver,
    Resolution,
    ViewConstraints,
)
from statefuse.view import build_view


def _claim(
    op_id: str,
    claim_id: str,
    confidence: float,
    ts: str,
    value: str,
    *,
    subject: str = "deadline",
) -> ClaimAdded:
    return ClaimAdded(
        op_id=op_id,
        replica_id="replicaA",
        timestamp=ts,
        claim=Claim(
            claim_id=claim_id,
            key=ClaimKey(namespace="proj", subject=subject, predicate="date"),
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


def test_heuristic_resolver_returns_provisional_suggestion() -> None:
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
    assert projection.selected_claims == {}
    assert projection.provisional_claims[key].claim_id == "c2"
    assert projection.selection_basis[key] == "provisional"
    assert len(projection.unresolved_conflicts) == 1
    assert key in projection.surfaced_conflicts
    assert "deterministic heuristic" in projection.explanations["proj:deadline:date"]


def test_default_resolver_abstains_without_selecting_a_conflict_side() -> None:
    state = materialize(
        OpLog(
            [
                _claim("op-1", "c1", 0.80, "2026-03-01T10:00:00.000000Z", "2026-03-25"),
                _claim("op-2", "c2", 0.90, "2026-03-01T10:00:00.000000Z", "2026-03-26"),
            ]
        )
    )

    projection = build_view(state=state, constraints=ViewConstraints(scope="task-default"))
    key = ClaimKey(namespace="proj", subject="deadline", predicate="date")

    assert projection.selected_claims == {}
    assert projection.provisional_claims == {}
    assert projection.selection_basis[key] == "abstained"
    assert projection.unresolved_conflicts == state.conflicts


def test_committed_cross_key_resolution_removes_a_provisional_suggestion() -> None:
    claims = [
        _claim("op-1", "c1", 0.80, "2026-03-01T10:00:00.000000Z", "2026-03-25"),
        _claim("op-2", "c2", 0.90, "2026-03-01T10:00:01.000000Z", "2026-03-26"),
        _claim(
            "op-3",
            "c3",
            1.0,
            "2026-03-01T10:00:02.000000Z",
            "2026-03-27",
            subject="release",
        ),
    ]

    def cross_key_detector(context: ConflictDetectionContext) -> Iterable[ConflictSet]:
        candidates = (context.claims_by_id["c2"], context.claims_by_id["c3"])
        return (
            make_conflict(
                candidates=candidates,
                key=candidates[0].key,
                conflict_type="execution.schedule",
                conflict_class="execution",
                conflict_subclass="schedule",
                detector_id="schedule/v1",
                reason="Dates cannot both be used.",
            ),
        )

    initial = materialize(OpLog(claims), conflict_detectors=(cross_key_detector,))
    cross_key = next(item for item in initial.conflicts if item.detector_id == "schedule/v1")
    resolution = ResolutionAdded(
        op_id="op-resolution",
        replica_id="reviewer",
        timestamp="2026-03-01T10:01:00.000000Z",
        resolution=ResolutionRecord(
            resolution_id="resolution-cross-key",
            conflict_ref=cross_key.conflict_ref,
            observed_conflict_id=cross_key.conflict_id,
            selected_claim_ids=("c3",),
            rejected_claim_ids=("c2",),
            retained_claim_ids=(),
            resolution_type="human_review",
            reason="Use the release date.",
            evidence_ids=(),
            actor_id="reviewer",
            timestamp="2026-03-01T10:01:00.000000Z",
            outcome="replace",
        ),
    )
    state = materialize(
        OpLog([*claims, resolution]), conflict_detectors=(cross_key_detector,)
    )

    projection = build_view(state, ViewConstraints(), resolver=HeuristicResolver())

    assert all(claim.claim_id != "c2" for claim in projection.provisional_claims.values())


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
        key: tuple(claim.claim_id for claim in claims)
        for key, claims in state.active_claims_by_key.items()
    }
    before_conflicts = tuple(conflict.conflict_id for conflict in state.conflicts)

    _ = build_view(
        state=state,
        constraints=ViewConstraints(scope="task-3"),
        resolver=HeuristicResolver(),
    )

    after_claim_ids = {
        key: tuple(claim.claim_id for claim in claims)
        for key, claims in state.active_claims_by_key.items()
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
