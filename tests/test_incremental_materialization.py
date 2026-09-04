from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest

from statefuse import (
    Claim,
    ClaimAdded,
    ClaimKey,
    ClaimRetracted,
    ConflictLifecycleEvent,
    ConflictLifecycleEventAdded,
    InMemoryStore,
    JsonlStore,
    MaterializationCheckpoint,
    Memory,
    OpLog,
    ResolutionAdded,
    ResolutionRecord,
    SQLiteStore,
    materialize,
)
from statefuse.integrations import (
    FakeMemoryRepositoryAdapter,
    InMemoryExternalReferenceStore,
    ProjectionService,
    project_state_delta,
)
from statefuse.materialize import apply_operations_incrementally


def _claim_op(op_id: str, claim_id: str, subject: str, value: str) -> ClaimAdded:
    timestamp = f"2026-03-01T00:00:0{int(op_id[-1])}.000000Z"
    return ClaimAdded(
        op_id=op_id,
        replica_id="test",
        timestamp=timestamp,
        claim=Claim(
            claim_id=claim_id,
            key=ClaimKey("project", subject, "value"),
            value=value,
            confidence=0.8,
            timestamp=timestamp,
            provenance={"replica_id": "test"},
        ),
    )


def test_incremental_state_matches_full_materialization_for_mixed_operations() -> None:
    memory = Memory(store=InMemoryStore(), replica_id="test")
    source_id = memory.add_source(source_type="message", message_id="m1")
    evidence_id = memory.add_evidence(pointer="message://m1", source_id=source_id)
    first_claim_id = memory.add_claim(
        namespace="project",
        subject="deadline",
        predicate="date",
        value="May 12",
        confidence=0.8,
        evidence_ids=(evidence_id,),
    )
    memory.materialize()

    second_claim_id = memory.add_claim(
        namespace="project",
        subject="deadline",
        predicate="date",
        value="May 15",
        confidence=0.9,
        evidence_ids=(evidence_id,),
    )
    conflict = memory.materialize().conflicts[0]
    memory.add_resolution(
        conflict_ref=conflict.conflict_ref,
        observed_conflict_id=conflict.conflict_id,
        selected_claim_ids=(second_claim_id,),
        rejected_claim_ids=(first_claim_id,),
        resolution_type="human",
        reason="Confirmed",
        actor_id="reviewer",
    )
    memory.append_op(
        ConflictLifecycleEventAdded(
            op_id="lifecycle-1",
            replica_id="test",
            timestamp="2026-03-01T00:01:00.000000Z",
            event=ConflictLifecycleEvent(
                event_id="event-1",
                conflict_ref=conflict.conflict_ref,
                observed_conflict_id=conflict.conflict_id,
                status="reopened",
                timestamp="2026-03-01T00:01:00.000000Z",
                reason="Needs another review",
            ),
        )
    )
    memory.add_decision(scope="project", payload={"ship": True})
    memory.add_derivation(
        rule_id="copy",
        input_claim_ids=(second_claim_id,),
        output_claim_ids=(),
        engine="test",
        explanation="test derivation",
    )
    memory.retract_claim(
        target_claim_id=first_claim_id,
        evidence_ids=(evidence_id,),
        reason="Superseded",
        supersedes_claim_id=second_claim_id,
    )

    incremental = memory.materialize()
    full = materialize(memory.load_oplog(), predicate_registry=memory.predicate_registry)

    assert incremental == full


def test_retraction_received_before_claim_converges_incrementally() -> None:
    store = InMemoryStore()
    store.append(
        ClaimRetracted(
            op_id="retract-1",
            replica_id="test",
            timestamp="2026-03-01T00:00:00.000000Z",
            target_claim_id="c1",
            reason="Withdrawn",
        )
    )
    memory = Memory(store=store, replica_id="test")
    before_claim = memory.materialize()

    assert before_claim == materialize(store.load_oplog())
    assert before_claim.inactive_claim_ids == {"c1"}

    store.append(_claim_op("op-1", "c1", "deadline", "May 12"))

    incremental = memory.materialize()

    assert incremental == materialize(store.load_oplog())
    assert incremental.active_claims_by_key == {}
    assert incremental.inactive_claim_ids == {"c1"}


def test_incremental_conflict_detection_only_receives_the_affected_key() -> None:
    store = InMemoryStore()
    memory = Memory(store=store, replica_id="test")
    store.append(_claim_op("op-1", "c1", "deadline", "May 12"))
    store.append(_claim_op("op-2", "c2", "owner", "Alice"))
    memory.materialize()
    store.append(_claim_op("op-3", "c3", "deadline", "May 15"))

    from statefuse.conflict import update_direct_conflict_index

    materialize_module = importlib.import_module("statefuse.materialize")
    with patch.object(
        materialize_module,
        "update_direct_conflict_index",
        wraps=update_direct_conflict_index,
    ) as update_index:
        memory.materialize()

    assert update_index.call_count == 1
    assert {claim.key for claim in update_index.call_args.args[1]} == {
        ClaimKey("project", "deadline", "value")
    }


@pytest.mark.parametrize("store_type", [JsonlStore, SQLiteStore])
def test_restart_loads_checkpoint_and_applies_only_post_cursor_ops(tmp_path, store_type) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / ("ops.jsonl" if store_type is JsonlStore else "ops.sqlite")
    store = store_type(path)
    store.append(_claim_op("op-1", "c1", "deadline", "May 12"))
    first_memory = Memory(store=store, replica_id="test")
    first_memory.materialize()

    reopened_store = store_type(path)
    reopened_store.append(_claim_op("op-2", "c2", "deadline", "May 15"))
    restored_memory = Memory(store=reopened_store, replica_id="test")

    with patch(
        "statefuse.memory.apply_operations_incrementally",
        wraps=apply_operations_incrementally,
    ) as apply:
        restored = restored_memory.materialize()

    assert apply.call_count == 1
    assert [op.op_id for op in apply.call_args.args[1]] == ["op-2"]
    assert restored == materialize(reopened_store.load_oplog())


def test_invalid_checkpoint_version_falls_back_to_full_materialization(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "ops.sqlite")
    store.append(_claim_op("op-1", "c1", "deadline", "May 12"))
    store.save_materialization_checkpoint(
        MaterializationCheckpoint(
            version="invalid",
            config_token="default",
            cursor=1,
            payload="{}",
        )
    )
    memory = Memory(store=store, replica_id="test")

    with patch("statefuse.memory.materialize", wraps=materialize) as full_materialize:
        state = memory.materialize()

    assert full_materialize.call_count == 1
    assert set(state.claims_by_id) == {"c1"}


def test_mutating_the_predicate_registry_invalidates_the_cached_state() -> None:
    memory = Memory(store=InMemoryStore(), replica_id="test")
    memory.append_op(_claim_op("op-1", "c1", "deadline", "ALPHA"))
    memory.append_op(_claim_op("op-2", "c2", "deadline", "alpha"))
    assert len(memory.materialize().conflicts) == 1

    memory.predicate_registry.register("value", normalize=lambda value: str(value).casefold())

    assert memory.materialize().conflicts == []


def test_mutating_a_returned_state_does_not_corrupt_the_cache() -> None:
    memory = Memory(store=InMemoryStore(), replica_id="test")
    memory.append_op(_claim_op("op-1", "c1", "deadline", "May 12"))
    state = memory.materialize()
    state.active_claims_by_key.clear()
    state.claims_by_id.clear()

    restored = memory.materialize()

    assert set(restored.claims_by_id) == {"c1"}
    assert list(restored.active_claims_by_key.values())[0][0].claim_id == "c1"


def test_late_claim_projects_an_earlier_resolution() -> None:
    claims = (
        _claim_op("op-1", "c1", "deadline", "May 12"),
        _claim_op("op-2", "c2", "deadline", "May 15"),
    )
    conflict = materialize(OpLog(claims)).conflicts[0]
    resolution = ResolutionAdded(
        op_id="resolution-op",
        replica_id="test",
        timestamp="2026-03-01T00:01:00.000000Z",
        resolution=ResolutionRecord(
            resolution_id="r1",
            conflict_ref=conflict.conflict_ref,
            observed_conflict_id=conflict.conflict_id,
            selected_claim_ids=("c2",),
            rejected_claim_ids=("c1",),
            retained_claim_ids=(),
            resolution_type="human",
            reason="Confirmed",
            evidence_ids=(),
            actor_id="reviewer",
            timestamp="2026-03-01T00:01:00.000000Z",
        ),
    )
    memory = Memory(store=InMemoryStore(), replica_id="test")
    memory.append_op(resolution)
    memory.materialize()
    cursor = memory.materialization_cursor
    assert cursor is not None
    for claim in claims:
        memory.append_op(claim)

    state, delta = memory.materialize_with_delta(cursor)
    projected = project_state_delta(state, "project", delta)

    assert "statefuse:resolution:r1" in {
        record.projection_id for record in projected.upserts
    }


def test_projection_sync_does_not_rebuild_unrelated_records() -> None:
    memory = Memory(store=InMemoryStore(), replica_id="test")
    memory.add_claim(
        namespace="project",
        subject="deadline",
        predicate="date",
        value="May 12",
        confidence=0.8,
        evidence_ids=(),
        claim_id="deadline-1",
    )
    memory.add_claim(
        namespace="project",
        subject="owner",
        predicate="name",
        value="Alice",
        confidence=0.8,
        evidence_ids=(),
        claim_id="owner-1",
    )
    adapter = FakeMemoryRepositoryAdapter()
    service = ProjectionService(memory, adapter, InMemoryExternalReferenceStore())
    service.synchronize("project")
    writes_before = adapter.write_count

    memory.add_claim(
        namespace="project",
        subject="deadline",
        predicate="date",
        value="May 15",
        confidence=0.9,
        evidence_ids=(),
        claim_id="deadline-2",
    )
    report = service.synchronize("project")

    touched = set(report.created) | set(report.updated) | set(report.deleted)
    assert "statefuse:claim:owner-1" not in touched
    assert adapter.write_count - writes_before == 3


def test_projection_sync_tracks_an_independent_cursor_per_namespace() -> None:
    memory = Memory(store=InMemoryStore(), replica_id="test")
    memory.add_claim(
        namespace="project",
        subject="deadline",
        predicate="date",
        value="May 12",
        confidence=0.8,
        evidence_ids=(),
        claim_id="project-1",
    )
    memory.add_claim(
        namespace="profile",
        subject="user",
        predicate="city",
        value="Shenzhen",
        confidence=0.8,
        evidence_ids=(),
        claim_id="profile-1",
    )
    service = ProjectionService(
        memory,
        FakeMemoryRepositoryAdapter(),
        InMemoryExternalReferenceStore(),
    )

    project = service.synchronize("project")
    profile = service.synchronize("profile")

    assert project.created == ("statefuse:claim:project-1",)
    assert profile.created == ("statefuse:claim:profile-1",)


def test_resolution_delta_updates_only_its_conflict_and_resolution_records() -> None:
    memory = Memory(store=InMemoryStore(), replica_id="test")
    for claim_id, value in (("c1", "May 12"), ("c2", "May 15")):
        memory.add_claim(
            namespace="project",
            subject="deadline",
            predicate="date",
            value=value,
            confidence=0.8,
            evidence_ids=(),
            claim_id=claim_id,
        )
    memory.add_claim(
        namespace="project",
        subject="owner",
        predicate="name",
        value="Alice",
        confidence=0.8,
        evidence_ids=(),
        claim_id="owner-1",
    )
    adapter = FakeMemoryRepositoryAdapter()
    service = ProjectionService(memory, adapter, InMemoryExternalReferenceStore())
    service.synchronize("project")
    conflict = memory.materialize().conflicts[0]

    memory.add_resolution(
        conflict_ref=conflict.conflict_ref,
        observed_conflict_id=conflict.conflict_id,
        selected_claim_ids=("c2",),
        rejected_claim_ids=("c1",),
        resolution_type="human",
        reason="Confirmed",
        actor_id="reviewer",
        resolution_id="r1",
    )
    report = service.synchronize("project")

    assert report.created == ("statefuse:resolution:r1",)
    assert report.updated == (f"statefuse:conflict:{conflict.conflict_id}",)
    assert "statefuse:claim:owner-1" not in report.unchanged


def test_custom_detectors_keep_the_full_materialization_fallback() -> None:
    def detector(_context):  # type: ignore[no-untyped-def]
        return ()

    memory = Memory(
        store=InMemoryStore(),
        replica_id="test",
        conflict_detectors=(detector,),
    )
    memory.append_op(_claim_op("op-1", "c1", "deadline", "May 12"))

    _, delta = memory.materialize_with_delta(None)

    assert delta.full_rebuild is True


def test_incremental_conflict_metadata_matches_full_materialization() -> None:
    memory = Memory(store=InMemoryStore(), replica_id="test")
    source_id = memory.add_source(source_type="message", source_id="source-1")
    evidence_id = memory.add_evidence(
        pointer="message://1",
        source_id=source_id,
        evidence_id="evidence-1",
    )
    derivation_id = memory.add_derivation(
        rule_id="combine",
        input_claim_ids=("input-1", "input-2"),
        output_claim_ids=("c1",),
        engine="test",
        explanation="Combined two inputs.",
        derivation_id="derivation-1",
    )
    memory.add_claim(
        namespace="project",
        subject="deadline",
        predicate="date",
        value="May 12",
        confidence=0.8,
        evidence_ids=(evidence_id,),
        derivation_id=derivation_id,
        claim_id="c1",
    )
    memory.materialize()

    memory.add_claim(
        namespace="project",
        subject="deadline",
        predicate="date",
        value="May 15",
        confidence=0.9,
        evidence_ids=(evidence_id,),
        claim_id="c2",
    )

    incremental = memory.materialize()
    full = materialize(memory.load_oplog(), predicate_registry=memory.predicate_registry)

    assert incremental == full
    assert incremental.conflicts[0].annotations["provenance"] == ["message"]
    assert incremental.conflicts[0].annotations["dependency_depth"] == "multi_hop"


def test_late_conflict_dependencies_recompute_only_their_claim_keys() -> None:
    memory = Memory(store=InMemoryStore(), replica_id="test")
    for claim_id, value in (("c1", "May 12"), ("c2", "May 15")):
        memory.add_claim(
            namespace="project",
            subject="deadline",
            predicate="date",
            value=value,
            confidence=0.8,
            evidence_ids=("evidence-1",),
            derivation_id="derivation-1" if claim_id == "c1" else None,
            claim_id=claim_id,
        )
    memory.materialize()

    memory.add_evidence(
        pointer="message://1",
        source_id="source-1",
        evidence_id="evidence-1",
    )
    assert memory.materialize() == materialize(memory.load_oplog())

    memory.add_source(source_type="message", source_id="source-1")
    assert memory.materialize() == materialize(memory.load_oplog())

    memory.add_derivation(
        rule_id="combine",
        input_claim_ids=("input-1", "input-2"),
        output_claim_ids=("c1",),
        engine="test",
        explanation="Combined two inputs.",
        derivation_id="derivation-1",
    )
    incremental = memory.materialize()
    full = materialize(memory.load_oplog())

    assert incremental == full
    assert incremental.conflicts[0].annotations == {
        "dependency_depth": "multi_hop",
        "provenance": ["message"],
        "representation": "categorical",
    }
