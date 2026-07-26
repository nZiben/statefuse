from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from statefuse import (
    Claim,
    ClaimAdded,
    ClaimKey,
    ConflictLifecycleEvent,
    ConflictLifecycleEventAdded,
    Derivation,
    DerivationAdded,
    Evidence,
    EvidenceAdded,
    InMemoryStore,
    JsonlStore,
    Memory,
    Op,
    OpLog,
    ResolutionAdded,
    ResolutionRecord,
    Source,
    SourceAdded,
    SQLiteStore,
    ValidityInterval,
    materialize,
    merge,
    sign_claim,
    verify_claim_signature,
)
from statefuse.compaction import compact_oplog
from statefuse.langgraph import append_op_to_graph_state, oplog_from_graph_state
from statefuse.ops import AnyOp
from statefuse.utils import canonical_json_dumps

TS = "2026-03-01T00:00:00.000000Z"


def _claim(claim_id: str, value: str, **kwargs: object) -> Claim:
    return Claim(
        claim_id=claim_id,
        key=ClaimKey(namespace="project", subject="deadline", predicate="date"),
        value=value,
        confidence=0.8,
        timestamp=TS,
        evidence_ids=("e1",),
        provenance={"replica_id": "a"},
        **kwargs,
    )


def _fixture_ops() -> list[AnyOp]:
    source_op = SourceAdded(
        op_id="op-1-source",
        replica_id="a",
        timestamp=TS,
        source=Source(
            source_id="s1",
            source_type="document",
            uri="doc://plan",
            system="files",
            actor_id="human-1",
            session_id="session-1",
            message_id="message-1",
            timestamp=TS,
            metadata={"title": "Plan"},
        ),
    )
    evidence_op = EvidenceAdded(
        op_id="op-2-evidence",
        replica_id="a",
        timestamp=TS,
        evidence=Evidence(
            evidence_id="e1",
            pointer="doc://plan#deadline",
            metadata={"page": 1},
            source_id="s1",
            content_digest="sha256:content",
        ),
    )
    claim_one_op = ClaimAdded(
        op_id="op-3-claim-one",
        replica_id="a",
        timestamp=TS,
        claim=_claim(
            "c1",
            "2026-03-25",
            validity=ValidityInterval(valid_from="2026-03-01T00:00:00Z"),
        ),
    )
    claim_two_op = ClaimAdded(
        op_id="op-4-claim-two",
        replica_id="b",
        timestamp=TS,
        claim=_claim("c2", "2026-03-28", derivation_id="d1"),
    )
    derivation_op = DerivationAdded(
        op_id="op-5-derivation",
        replica_id="b",
        timestamp=TS,
        derivation=Derivation(
            derivation_id="d1",
            rule_id="deadline-adjustment",
            input_claim_ids=("c1",),
            output_claim_ids=("c2",),
            engine="rules-v1",
            explanation="Adjusted the deadline.",
            timestamp=TS,
            confidence=0.7,
            metadata={"version": 1},
        ),
    )
    conflict = materialize(OpLog([claim_one_op, claim_two_op])).conflicts[0]
    resolution_op = ResolutionAdded(
        op_id="op-6-resolution",
        replica_id="reviewer",
        timestamp="2026-03-01T00:00:01.000000Z",
        resolution=ResolutionRecord(
            resolution_id="r1",
            conflict_ref=conflict.conflict_ref,
            observed_conflict_id=conflict.conflict_id,
            selected_claim_ids=("c1",),
            rejected_claim_ids=("c2",),
            retained_claim_ids=(),
            resolution_type="human",
            reason="Confirmed against the plan.",
            evidence_ids=("e1",),
            actor_id="reviewer",
            timestamp="2026-03-01T00:00:01.000000Z",
            scope="planning",
            metadata={"ticket": "T-1"},
        ),
    )
    event_op = ConflictLifecycleEventAdded(
        op_id="op-7-event",
        replica_id="reviewer",
        timestamp="2026-03-01T00:00:02.000000Z",
        event=ConflictLifecycleEvent(
            event_id="event-1",
            conflict_ref=conflict.conflict_ref,
            observed_conflict_id=conflict.conflict_id,
            status="resolved",
            timestamp="2026-03-01T00:00:02.000000Z",
            reason="Resolution committed.",
            actor_id="reviewer",
            scope="planning",
            resolution_id="r1",
            metadata={"ticket": "T-1"},
        ),
    )
    return [
        source_op,
        evidence_op,
        claim_one_op,
        claim_two_op,
        derivation_op,
        resolution_op,
        event_op,
    ]


def test_old_evidence_and_claim_operation_payloads_round_trip_unchanged() -> None:
    old_payloads = [
        {
            "op_id": "old-evidence-op",
            "op_type": "EvidenceAdded",
            "replica_id": "legacy",
            "timestamp": TS,
            "evidence": {
                "evidence_id": "old-evidence",
                "pointer": "doc://legacy",
                "metadata": {"page": 2},
            },
        },
        {
            "op_id": "old-claim-op",
            "op_type": "ClaimAdded",
            "replica_id": "legacy",
            "timestamp": TS,
            "claim": {
                "claim_id": "old-claim",
                "claim_ref": "sha256:legacy-ref",
                "key": {
                    "namespace": "project",
                    "subject": "deadline",
                    "predicate": "date",
                },
                "value": "2026-03-25",
                "confidence": 0.8,
                "timestamp": TS,
                "evidence_ids": ["old-evidence"],
                "provenance": {"replica_id": "legacy"},
            },
        },
    ]

    for payload in old_payloads:
        encoded = canonical_json_dumps(payload)
        assert Op.from_dict(payload).to_dict() == payload
        assert Op.from_json(encoded).to_json() == encoded


def test_new_models_and_operations_round_trip() -> None:
    operations = _fixture_ops()
    models = [
        operations[0].source,
        operations[1].evidence,
        operations[2].claim.validity,
        operations[3].claim,
        operations[4].derivation,
        operations[5].resolution,
        operations[6].event,
    ]

    for model in models:
        assert model is not None
        assert type(model).from_dict(model.to_dict()) == model
    for operation in operations:
        assert Op.from_dict(operation.to_dict()) == operation
        assert Op.from_json(operation.to_json()) == operation


def test_merge_laws_hold_with_new_operations() -> None:
    operations = _fixture_ops()
    left = OpLog(operations[:2])
    middle = OpLog(operations[2:5])
    right = OpLog(operations[5:])

    assert merge(left, middle) == merge(middle, left)
    assert merge(merge(left, middle), right) == merge(left, merge(middle, right))
    assert merge(left, left) == left


def test_shuffled_operations_materialize_identically() -> None:
    operations = _fixture_ops()
    expected = materialize(OpLog(operations))
    shuffled = list(operations)
    random.Random(7).shuffle(shuffled)
    actual = materialize(OpLog(shuffled))

    assert actual == expected
    assert tuple(actual.sources_by_id) == ("s1",)
    assert tuple(actual.derivations_by_id) == ("d1",)
    assert len(actual.conflicts) == 1
    assert actual.lifecycle_history_by_conflict_ref
    assert actual.resolutions_by_conflict_ref


def test_new_entity_identifier_collisions_are_rejected() -> None:
    operations = _fixture_ops()
    source_op = operations[0]
    derivation_op = operations[4]
    resolution_op = operations[5]
    event_op = operations[6]
    collisions = [
        (
            source_op,
            replace(
                source_op,
                op_id="collision-source",
                source=replace(source_op.source, uri="doc://different"),
            ),
            "source_id collision",
        ),
        (
            derivation_op,
            replace(
                derivation_op,
                op_id="collision-derivation",
                derivation=replace(derivation_op.derivation, explanation="Different"),
            ),
            "derivation_id collision",
        ),
        (
            resolution_op,
            replace(
                resolution_op,
                op_id="collision-resolution",
                resolution=replace(resolution_op.resolution, reason="Different"),
            ),
            "resolution_id collision",
        ),
        (
            event_op,
            replace(
                event_op,
                op_id="collision-event",
                event=replace(event_op.event, reason="Different"),
            ),
            "event_id collision",
        ),
    ]

    for original, collision, message in collisions:
        try:
            materialize(OpLog([original, collision]))
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"Expected {message}.")


def test_related_records_can_arrive_out_of_order() -> None:
    operations = _fixture_ops()
    source_op, evidence_op, claim_one_op, claim_two_op, derivation_op = operations[:5]
    out_of_order = [
        replace(derivation_op, op_id="op-01-derivation"),
        replace(claim_two_op, op_id="op-02-output-claim"),
        replace(evidence_op, op_id="op-03-evidence"),
        replace(claim_one_op, op_id="op-04-input-claim"),
        replace(source_op, op_id="op-05-source"),
    ]

    partial_log = OpLog(out_of_order[:3])
    partial = materialize(partial_log)
    assert "d1" in partial.derivations_by_id
    assert "c2" in partial.claims_by_id
    assert "e1" in partial.evidence_by_id
    assert "c1" not in partial.claims_by_id
    assert "s1" not in partial.sources_by_id

    merged = merge(partial_log, OpLog(out_of_order[3:]))
    complete = materialize(merged)
    assert set(complete.claims_by_id) == {"c1", "c2"}
    assert complete.evidence_by_id["e1"].source_id == "s1"
    assert complete.derivations_by_id["d1"].input_claim_ids == ("c1",)
    assert complete.derivations_by_id["d1"].output_claim_ids == ("c2",)


def test_validity_and_derivation_links_do_not_change_claim_ref() -> None:
    assert ValidityInterval(valid_from=TS)
    assert ValidityInterval(valid_until=TS)
    assert ValidityInterval(
        valid_from="2026-03-01T01:00:00+01:00",
        valid_until="2026-03-01T00:00:00Z",
    )
    try:
        ValidityInterval(valid_from="2026-03-02T00:00:00Z", valid_until=TS)
    except ValueError as exc:
        assert "valid_from" in str(exc)
    else:
        raise AssertionError("Expected an inverted validity interval to fail.")

    plain = _claim("plain", "2026-03-25")
    linked = Claim(
        claim_id="linked",
        key=plain.key,
        value=plain.value,
        confidence=0.1,
        timestamp="2026-04-01T00:00:00Z",
        evidence_ids=("other-evidence",),
        provenance={"replica_id": "other"},
        validity=ValidityInterval(valid_until="2026-04-30T00:00:00Z"),
        derivation_id="d-other",
    )

    assert linked.claim_ref == plain.claim_ref
    assert "validity" not in plain.to_dict()
    assert "derivation_id" not in plain.to_dict()


def test_new_operations_persist_and_survive_compaction() -> None:
    operations = _fixture_ops()
    expected = OpLog(operations)
    with TemporaryDirectory() as directory:
        stores = [
            JsonlStore(Path(directory) / "ops.jsonl"),
            SQLiteStore(Path(directory) / "ops.db"),
        ]
        for store in stores:
            for operation in operations:
                store.append(operation)
            assert store.load_oplog() == expected

    graph_state: dict[str, object] = {}
    for operation in operations:
        append_op_to_graph_state(graph_state, operation)
    assert oplog_from_graph_state(graph_state) == expected

    compacted = compact_oplog(expected)
    preserved = {
        operation.op_id
        for operation in operations
        if isinstance(
            operation,
            (SourceAdded, DerivationAdded, ResolutionAdded, ConflictLifecycleEventAdded),
        )
    }
    assert preserved <= set(compacted.op_ids())

    resolution_only_evidence = EvidenceAdded(
        op_id="op-resolution-evidence",
        replica_id="reviewer",
        timestamp=TS,
        evidence=Evidence("resolution-evidence", "review://note"),
    )
    resolution = operations[5]
    assert isinstance(resolution, ResolutionAdded)
    resolution = replace(
        resolution,
        op_id="op-resolution-only",
        resolution=replace(resolution.resolution, evidence_ids=("resolution-evidence",)),
    )
    compacted_resolution = compact_oplog(OpLog([resolution_only_evidence, resolution]))
    assert compacted_resolution.get(resolution_only_evidence.op_id) == resolution_only_evidence


def test_memory_source_ids_and_resolution_validation() -> None:
    left = Memory(InMemoryStore(), replica_id="a")
    right = Memory(InMemoryStore(), replica_id="b")
    left_id = left.add_source(source_type="document", uri="doc://x", metadata={"a": 1, "b": 2})
    right_id = right.add_source(
        source_type="document",
        uri="doc://x",
        metadata={"b": 2, "a": 1},
    )
    assert left_id == right_id
    assert left.add_source(source_type="document", source_id="caller-id") == "caller-id"
    evidence_id = left.add_evidence(pointer="doc://x", source_id=left_id)
    other_source_id = left.add_source(source_type="document", uri="doc://other")
    other_evidence_id = left.add_evidence(pointer="doc://x", source_id=other_source_id)
    assert evidence_id != other_evidence_id
    assert left.materialize().evidence_by_id[evidence_id].source_id == left_id

    common = {
        "resolution_id": "r-invalid",
        "conflict_ref": "conflict-ref:x",
        "observed_conflict_id": "conflict:x",
        "selected_claim_ids": ("c1",),
        "rejected_claim_ids": ("c1",),
        "retained_claim_ids": (),
        "resolution_type": "human",
        "reason": "Reviewed.",
        "evidence_ids": (),
        "actor_id": "reviewer",
        "timestamp": TS,
    }
    try:
        ResolutionRecord(**common)
    except ValueError as exc:
        assert "must not overlap" in str(exc)
    else:
        raise AssertionError("Expected overlapping resolution claim sets to fail.")

    common["rejected_claim_ids"] = ("c2",)
    common["valid_from"] = "2026-03-02T00:00:00Z"
    common["valid_until"] = TS
    try:
        ResolutionRecord(**common)
    except ValueError as exc:
        assert "valid_from" in str(exc)
    else:
        raise AssertionError("Expected inverted resolution validity to fail.")

    valid_resolution = _fixture_ops()[5]
    valid_event = _fixture_ops()[6]
    assert isinstance(valid_resolution, ResolutionAdded)
    assert isinstance(valid_event, ConflictLifecycleEventAdded)
    for record in (valid_resolution.resolution, valid_event.event):
        try:
            replace(record, timestamp="not-a-timestamp")
        except ValueError:
            pass
        else:
            raise AssertionError("Expected malformed authority timestamps to fail early.")

    invalid_payload = valid_resolution.resolution.to_dict()
    invalid_payload["actor_id"] = None
    try:
        ResolutionRecord.from_dict(invalid_payload)
    except ValueError as exc:
        assert "actor_id must be a string" in str(exc)
    else:
        raise AssertionError("Expected null required resolution fields to fail.")


def test_claim_signatures_cover_validity_and_derivation_links() -> None:
    claim = _claim(
        "signed",
        "2026-03-25",
        validity=ValidityInterval(valid_from=TS),
        derivation_id="d1",
    )
    signed = sign_claim(claim, secret="secret", key_id="key-1")
    secrets = {"key-1": "secret"}

    assert verify_claim_signature(signed, key_secrets=secrets)
    assert not verify_claim_signature(
        replace(signed, validity=ValidityInterval(valid_until=TS)),
        key_secrets=secrets,
    )
    assert not verify_claim_signature(
        replace(signed, derivation_id="d2"),
        key_secrets=secrets,
    )
