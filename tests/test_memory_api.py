from __future__ import annotations

from unittest.mock import patch

from statefuse import InMemoryStore, Memory, PredicateRegistry, ViewConstraints
from statefuse.model import Evidence
from statefuse.ops import EvidenceAdded


def test_add_evidence_generates_sha256_identifier() -> None:
    mem = Memory(store=InMemoryStore(), replica_id="agentA")
    evidence_id = mem.add_evidence(pointer="doc://x", content="hello")
    assert evidence_id.startswith("sha256:")
    assert len(evidence_id.split(":", maxsplit=1)[1]) == 64


def test_add_retract_supersede_lifecycle() -> None:
    mem = Memory(store=InMemoryStore(), replica_id="agentA")
    evid = mem.add_evidence(pointer="doc://deadline", content="draft")
    old_claim = mem.add_claim(
        namespace="proj",
        subject="deadline",
        predicate="date",
        value="2026-03-20",
        confidence=0.6,
        evidence_ids=[evid],
    )
    new_claim = mem.add_claim(
        namespace="proj",
        subject="deadline",
        predicate="date",
        value="2026-03-25",
        confidence=0.9,
        evidence_ids=[evid],
    )
    mem.retract_claim(
        target_claim_id=old_claim,
        evidence_ids=[evid],
        reason="updated plan",
        supersedes_claim_id=new_claim,
    )

    state = mem.materialize()
    key = next(iter(state.active_claims_by_key))
    active_ids = [claim.claim_id for claim in state.active_claims_by_key[key]]
    assert old_claim not in active_ids
    assert new_claim in active_ids


def test_materialize_and_build_view_end_to_end() -> None:
    mem = Memory(store=InMemoryStore(), replica_id="agentA")
    evid = mem.add_evidence(pointer="doc://hq", content="city=New York")
    mem.add_claim(
        namespace="proj",
        subject="hq",
        predicate="city",
        value="New York",
        confidence=0.88,
        evidence_ids=[evid],
    )

    state = mem.materialize()
    projection = mem.build_view(constraints=ViewConstraints(scope="planning"))
    assert len(state.conflicts) == 0
    assert len(projection.selected_claims) == 1


def test_append_op_rejects_op_id_collision_with_different_payload() -> None:
    mem = Memory(store=InMemoryStore(), replica_id="agentA")
    op1 = EvidenceAdded(
        op_id="op-collision",
        replica_id="agentA",
        timestamp="2026-03-01T00:00:00.000000Z",
        evidence=Evidence(evidence_id="sha256:1", pointer="doc://a", metadata={}),
    )
    op2 = EvidenceAdded(
        op_id="op-collision",
        replica_id="agentA",
        timestamp="2026-03-01T00:00:01.000000Z",
        evidence=Evidence(evidence_id="sha256:2", pointer="doc://b", metadata={}),
    )

    assert mem.append_op(op1) is True
    try:
        mem.append_op(op2)
    except ValueError as exc:
        assert "op_id collision" in str(exc)
    else:
        raise AssertionError("Expected ValueError for op_id collision with different payload.")


def test_content_addressed_op_id_mode_is_available() -> None:
    with patch("statefuse.memory.utc_now_iso", return_value="2026-03-01T00:00:00.000000Z"):
        left = Memory(store=InMemoryStore(), replica_id="agentA", op_id_mode="content-addressed")
        right = Memory(store=InMemoryStore(), replica_id="agentA", op_id_mode="content-addressed")
        left.add_evidence(pointer="doc://x", content="hello")
        right.add_evidence(pointer="doc://x", content="hello")

    left_op_id = left.load_oplog().op_ids()[0]
    right_op_id = right.load_oplog().op_ids()[0]
    assert left_op_id.startswith("sha256:")
    assert left_op_id == right_op_id


def test_claim_ref_helper_respects_predicate_contract_normalization() -> None:
    registry = PredicateRegistry()
    registry.register("city", normalize=lambda value: str(value).strip().lower(), normalize_for_claim_ref=True)
    mem = Memory(store=InMemoryStore(), replica_id="agentA", predicate_registry=registry)
    ts = "2026-03-01T00:00:00.000000Z"
    left = mem.claim_ref_for(
        namespace="proj",
        subject="hq",
        predicate="city",
        value=" New York ",
        confidence=0.7,
        timestamp=ts,
        evidence_ids=[],
    )
    right = mem.claim_ref_for(
        namespace="proj",
        subject="hq",
        predicate="city",
        value="new york",
        confidence=0.7,
        timestamp=ts,
        evidence_ids=[],
    )
    assert left == right
