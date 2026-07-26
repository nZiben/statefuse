from __future__ import annotations

from statefuse.compaction import compact_oplog_with_report
from statefuse.materialize import materialize
from statefuse.merge import merge, merge_checked_authenticated
from statefuse.model import Claim, ClaimKey, ConflictLifecycleEvent, ResolutionRecord
from statefuse.oplog import OpLog
from statefuse.ops import (
    ClaimAdded,
    ClaimRetracted,
    ConflictLifecycleEventAdded,
    ResolutionAdded,
)
from statefuse.resolver import ViewConstraints
from statefuse.resolver_llm import LLMResolver
from statefuse.view import build_view

KEY = ClaimKey(namespace="project", subject="deadline", predicate="date")


def _claim(claim_id: str, value: str, second: int) -> ClaimAdded:
    timestamp = f"2026-03-01T00:00:{second:02d}.000000Z"
    return ClaimAdded(
        op_id=f"op-{claim_id}",
        replica_id=f"replica-{claim_id}",
        timestamp=timestamp,
        claim=Claim(
            claim_id=claim_id,
            key=KEY,
            value=value,
            confidence=0.8,
            timestamp=timestamp,
            provenance={"replica_id": f"replica-{claim_id}"},
        ),
    )


def _resolution(
    conflict_ref: str,
    conflict_id: str,
    *,
    resolution_id: str = "resolution-global",
    selected: tuple[str, ...] = ("c1",),
    rejected: tuple[str, ...] = ("c2",),
    scope: str | None = None,
    timestamp: str = "2026-03-01T00:01:00.000000Z",
) -> ResolutionAdded:
    return ResolutionAdded(
        op_id=f"op-{resolution_id}",
        replica_id="reviewer",
        timestamp=timestamp,
        resolution=ResolutionRecord(
            resolution_id=resolution_id,
            conflict_ref=conflict_ref,
            observed_conflict_id=conflict_id,
            selected_claim_ids=selected,
            rejected_claim_ids=rejected,
            retained_claim_ids=(),
            resolution_type="human",
            reason="Reviewed the source records.",
            evidence_ids=(),
            actor_id="reviewer-1",
            timestamp=timestamp,
            scope=scope,
        ),
    )


def _event(
    conflict_ref: str,
    conflict_id: str,
    *,
    event_id: str,
    status: str,
    timestamp: str,
    resolution_id: str | None = None,
    scope: str | None = None,
    op_id: str | None = None,
) -> ConflictLifecycleEventAdded:
    return ConflictLifecycleEventAdded(
        op_id=op_id or f"op-{event_id}",
        replica_id="reviewer",
        timestamp=timestamp,
        event=ConflictLifecycleEvent(
            event_id=event_id,
            conflict_ref=conflict_ref,
            observed_conflict_id=conflict_id,
            status=status,
            timestamp=timestamp,
            reason=f"Mark conflict {status}.",
            actor_id="reviewer-1",
            scope=scope,
            resolution_id=resolution_id,
        ),
    )


def _two_claim_conflict() -> tuple[list[ClaimAdded], str, str]:
    claims = [_claim("c1", "2026-03-25", 1), _claim("c2", "2026-03-26", 2)]
    conflict = materialize(OpLog(claims)).conflicts[0]
    return claims, conflict.conflict_ref, conflict.conflict_id


def test_conflict_ref_is_stable_while_snapshot_id_tracks_candidates() -> None:
    claims, conflict_ref, conflict_id = _two_claim_conflict()
    expanded = materialize(OpLog([*claims, _claim("c3", "2026-03-27", 3)])).conflicts[0]

    assert expanded.conflict_ref == conflict_ref
    assert expanded.conflict_id != conflict_id


def test_conflict_created_only_by_merge_is_open_without_lifecycle_event() -> None:
    left = OpLog([_claim("c1", "2026-03-25", 1)])
    right = OpLog([_claim("c2", "2026-03-26", 2)])

    state = materialize(merge(left, right))
    conflict = state.conflicts[0]

    assert state.lifecycle_status_by_conflict_ref_and_scope[(conflict.conflict_ref, None)] == "open"
    assert conflict.conflict_ref not in state.lifecycle_history_by_conflict_ref


def test_exact_scoped_resolution_wins_and_does_not_leak() -> None:
    claims, conflict_ref, conflict_id = _two_claim_conflict()
    global_resolution = _resolution(conflict_ref, conflict_id)
    scoped_resolution = _resolution(
        conflict_ref,
        conflict_id,
        resolution_id="resolution-task",
        selected=("c2",),
        rejected=("c1",),
        scope="task-1",
    )
    state = materialize(OpLog([*claims, global_resolution, scoped_resolution]))

    global_view = build_view(state, ViewConstraints())
    scoped_view = build_view(state, ViewConstraints(scope="task-1"))
    other_view = build_view(state, ViewConstraints(scope="task-2"))

    assert global_view.selected_claims[KEY].claim_id == "c1"
    assert scoped_view.selected_claims[KEY].claim_id == "c2"
    assert other_view.selected_claims[KEY].claim_id == "c1"
    assert KEY in global_view.surfaced_conflicts
    assert KEY in scoped_view.surfaced_conflicts


def test_uncovered_candidate_reopens_resolution_without_resolver_fallback() -> None:
    claims, conflict_ref, conflict_id = _two_claim_conflict()
    resolution = _resolution(conflict_ref, conflict_id)
    state = materialize(OpLog([*claims, resolution, _claim("c3", "2026-03-27", 3)]))

    class MustNotRun:
        def resolve(self, conflict, constraints, current_state):  # type: ignore[no-untyped-def]
            raise AssertionError("stale committed resolution must not fall through to resolver")

    projection = build_view(state, ViewConstraints(), resolver=MustNotRun())

    assert state.lifecycle_status_by_conflict_ref_and_scope[(conflict_ref, None)] == "reopened"
    assert (conflict_ref, None) not in state.effective_resolutions_by_conflict_ref_and_scope
    assert projection.selected_claims == {}
    assert [item.conflict_ref for item in projection.unresolved_conflicts] == [conflict_ref]
    assert "uncovered candidates: c3" in projection.explanations["project:deadline:date"]


def test_observed_snapshot_must_match_classified_claims_but_allows_covered_shrink() -> None:
    claims, conflict_ref, conflict_id = _two_claim_conflict()
    bogus = _resolution(conflict_ref, conflict_id)
    bogus = ResolutionAdded(
        op_id=bogus.op_id,
        replica_id=bogus.replica_id,
        timestamp=bogus.timestamp,
        resolution=ResolutionRecord.from_dict(
            {**bogus.resolution.to_dict(), "observed_conflict_id": "conflict:bogus"}
        ),
    )
    bogus_state = materialize(OpLog([*claims, bogus]))
    bogus_view = build_view(bogus_state, ViewConstraints())

    assert (conflict_ref, None) not in bogus_state.effective_resolutions_by_conflict_ref_and_scope
    assert (
        bogus_state.lifecycle_status_by_conflict_ref_and_scope[(conflict_ref, None)] == "reopened"
    )
    assert bogus_view.selected_claims == {}

    third = _claim("c3", "2026-03-27", 3)
    expanded = materialize(OpLog([*claims, third])).conflicts[0]
    reviewed = _resolution(
        expanded.conflict_ref,
        expanded.conflict_id,
        selected=("c1",),
        rejected=("c2", "c3"),
    )
    retract_third = ClaimRetracted(
        op_id="op-retract-c3",
        replica_id="reviewer",
        timestamp="2026-03-01T00:02:00.000000Z",
        target_claim_id="c3",
        reason="Candidate withdrawn.",
    )
    shrunk_state = materialize(OpLog([*claims, third, reviewed, retract_third]))
    shrunk_view = build_view(shrunk_state, ViewConstraints())

    assert shrunk_state.conflicts[0].conflict_id != expanded.conflict_id
    assert shrunk_view.selected_claims[KEY].claim_id == "c1"


def test_missing_resolution_fails_open() -> None:
    claims, conflict_ref, conflict_id = _two_claim_conflict()
    event = _event(
        conflict_ref,
        conflict_id,
        event_id="event-resolved",
        status="resolved",
        timestamp="2026-03-01T00:01:01.000000Z",
        resolution_id="resolution-missing",
    )

    state = materialize(OpLog([*claims, event]))

    assert state.lifecycle_status_by_conflict_ref_and_scope[(conflict_ref, None)] == "open"
    assert (conflict_ref, None) not in state.effective_resolutions_by_conflict_ref_and_scope
    assert state.lifecycle_history_by_conflict_ref[conflict_ref] == (event.event,)


def test_lifecycle_tie_break_invalidation_and_history_survive_retraction() -> None:
    claims, conflict_ref, conflict_id = _two_claim_conflict()
    timestamp = "2026-03-01T00:01:01.000000Z"
    resolution = _resolution(conflict_ref, conflict_id, timestamp="2026-03-01T00:01:00.000000Z")
    resolved = _event(
        conflict_ref,
        conflict_id,
        event_id="event-a",
        status="resolved",
        timestamp=timestamp,
        resolution_id=resolution.resolution.resolution_id,
        op_id="op-z",
    )
    invalidated = _event(
        conflict_ref,
        conflict_id,
        event_id="event-z",
        status="invalidated",
        timestamp=timestamp,
        resolution_id=resolution.resolution.resolution_id,
        op_id="op-a",
    )
    retractions = [
        ClaimRetracted(
            op_id=f"op-retract-{claim_id}",
            replica_id="reviewer",
            timestamp="2026-03-01T00:02:00.000000Z",
            target_claim_id=claim_id,
            reason="Remove current claim.",
        )
        for claim_id in ("c1", "c2")
    ]

    current = materialize(OpLog([*claims, resolution, resolved, invalidated]))
    retracted = materialize(
        OpLog([*claims, resolution, invalidated, resolved, *retractions])
    )

    assert [
        event.event_id for event in current.lifecycle_history_by_conflict_ref[conflict_ref]
    ] == ["event-a", "event-z"]
    assert current.lifecycle_status_by_conflict_ref_and_scope[(conflict_ref, None)] == "invalidated"
    assert (conflict_ref, None) not in current.effective_resolutions_by_conflict_ref_and_scope
    assert retracted.conflicts == []
    assert conflict_ref in retracted.lifecycle_history_by_conflict_ref
    assert resolution.resolution.resolution_id in retracted.resolutions_by_id


def test_llm_recommendation_is_not_persisted() -> None:
    claims, _, _ = _two_claim_conflict()
    oplog = OpLog(claims)
    state = materialize(oplog)

    class FakeClient:
        def resolve_json(self, **kwargs):  # type: ignore[no-untyped-def]
            return '{"chosen_claim_id":"c2","reason":"recommend c2"}'

    projection = build_view(
        state,
        ViewConstraints(scope="task"),
        resolver=LLMResolver(client=FakeClient()),
    )

    assert projection.selected_claims[KEY].claim_id == "c2"
    assert state.resolutions_by_id == {}
    assert state.lifecycle_events_by_id == {}
    assert len(oplog) == 2


def test_compaction_preserves_resolution_and_lifecycle_history() -> None:
    claims, conflict_ref, conflict_id = _two_claim_conflict()
    resolution = _resolution(conflict_ref, conflict_id)
    event = _event(
        conflict_ref,
        conflict_id,
        event_id="event-resolved",
        status="resolved",
        timestamp="2026-03-01T00:01:01.000000Z",
        resolution_id=resolution.resolution.resolution_id,
    )
    original = OpLog([*claims, resolution, event])

    report = compact_oplog_with_report(original)
    compacted_state = materialize(report.compacted)

    assert report.compacted.get(resolution.op_id) == resolution
    assert report.compacted.get(event.op_id) == event
    assert compacted_state.resolutions_by_id == materialize(original).resolutions_by_id
    assert (
        compacted_state.lifecycle_history_by_conflict_ref
        == materialize(original).lifecycle_history_by_conflict_ref
    )


def test_authenticated_merge_quarantines_unsigned_authority_operations() -> None:
    resolution = _resolution("conflict-ref:x", "conflict:x")
    event = _event(
        "conflict-ref:x",
        "conflict:x",
        event_id="event-resolved",
        status="resolved",
        timestamp="2026-03-01T00:01:01.000000Z",
        resolution_id=resolution.resolution.resolution_id,
    )

    report = merge_checked_authenticated(
        OpLog(),
        OpLog([resolution, event]),
        key_secrets={},
        require_signed=True,
    )

    assert len(report.merged) == 0
    assert {item.op_id for item in report.quarantined} == {resolution.op_id, event.op_id}
    assert {item.reason for item in report.quarantined} == {"authority_signature_unsupported"}

    permissive = merge_checked_authenticated(
        OpLog(),
        OpLog([resolution, event]),
        key_secrets={},
        require_signed=False,
    )
    assert permissive.merged.op_ids() == tuple(sorted((resolution.op_id, event.op_id)))
