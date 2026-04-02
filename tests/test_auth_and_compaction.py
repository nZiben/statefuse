from __future__ import annotations

from statefuse import InMemoryStore, Memory
from dataclasses import replace

from statefuse.auth import (
    claim_signature_status,
    retraction_signature_status,
    sign_claim,
    sign_retraction,
    verify_claim_signature,
    verify_retraction_signature,
)
from statefuse.compaction import compact_oplog_with_report, compact_projection_equivalent_with_report
from statefuse.materialize import materialize
from statefuse.merge import merge, merge_checked_authenticated
from statefuse.model import Claim, ClaimKey
from statefuse.oplog import OpLog
from statefuse.ops import ClaimAdded, ClaimRetracted


def test_sign_claim_and_verify_signature() -> None:
    claim = Claim(
        claim_id="c1",
        key=ClaimKey(namespace="auth", subject="item", predicate="status"),
        value="open",
        confidence=0.8,
        timestamp="2026-03-01T00:00:00.000000Z",
        evidence_ids=("e1",),
        provenance={"replica_id": "agent-a"},
    )
    signed = sign_claim(claim, secret="top-secret", key_id="k1")
    assert verify_claim_signature(signed, key_secrets={"k1": "top-secret"})
    assert claim_signature_status(signed, key_secrets={"k1": "top-secret"}) == "verified"
    assert claim_signature_status(signed, key_secrets={"k2": "other"}) == "unknown_key"

    tampered = replace(signed, value="closed")
    assert claim_signature_status(tampered, key_secrets={"k1": "top-secret"}) == "invalid"


def test_memory_add_claim_can_attach_signature() -> None:
    mem = Memory(store=InMemoryStore(), replica_id="agentA")
    evidence_id = mem.add_evidence(pointer="doc://x", content="hello")
    mem.add_claim(
        namespace="proj",
        subject="deadline",
        predicate="date",
        value="2026-03-25",
        confidence=0.8,
        evidence_ids=[evidence_id],
        signing_key="shared-secret",
        signing_key_id="key-1",
    )
    state = mem.materialize()
    claim = next(iter(next(iter(state.active_claims_by_key.values()))))
    assert verify_claim_signature(claim, key_secrets={"key-1": "shared-secret"})


def test_sign_retraction_and_verify_signature() -> None:
    retraction = ClaimRetracted(
        op_id="op-r1",
        replica_id="agent-a",
        timestamp="2026-03-01T00:00:00.000000Z",
        target_claim_id="c1",
        evidence_ids=("e1",),
        reason="updated plan",
    )
    signed = sign_retraction(retraction, secret="top-secret", key_id="k1")
    assert verify_retraction_signature(signed, key_secrets={"k1": "top-secret"})
    assert retraction_signature_status(signed, key_secrets={"k1": "top-secret"}) == "verified"

    tampered = replace(signed, reason="tampered")
    assert retraction_signature_status(tampered, key_secrets={"k1": "top-secret"}) == "invalid"


def test_compaction_preserves_materialized_state_and_tombstones() -> None:
    mem = Memory(store=InMemoryStore(), replica_id="agentA")
    evidence_id = mem.add_evidence(pointer="doc://deadline", content="draft")
    old_claim_id = mem.add_claim(
        namespace="proj",
        subject="deadline",
        predicate="date",
        value="2026-03-20",
        confidence=0.6,
        evidence_ids=[evidence_id],
    )
    new_claim_id = mem.add_claim(
        namespace="proj",
        subject="deadline",
        predicate="date",
        value="2026-03-25",
        confidence=0.9,
        evidence_ids=[evidence_id],
    )
    mem.retract_claim(
        target_claim_id=old_claim_id,
        evidence_ids=[evidence_id],
        reason="updated plan",
        supersedes_claim_id=new_claim_id,
    )

    original = mem.load_oplog()
    report = compact_oplog_with_report(original)
    compacted = report.compacted
    assert report.compacted_ops < report.original_ops
    assert report.active_claims_equivalent
    assert report.conflicts_equivalent
    assert report.projections_equivalent

    original_state = materialize(original)
    compacted_state = materialize(compacted)
    assert original_state.active_claims_by_key == compacted_state.active_claims_by_key
    assert original_state.conflicts == compacted_state.conflicts

    old_claim_op = next(
        op for op in original.iter_ops() if isinstance(op, ClaimAdded) and op.claim.claim_id == old_claim_id
    )
    merged = merge(compacted, OpLog([old_claim_op]))
    merged_state = materialize(merged)
    key = ClaimKey(namespace="proj", subject="deadline", predicate="date")
    active_ids = {claim.claim_id for claim in merged_state.active_claims_by_key[key]}
    assert old_claim_id not in active_ids
    assert new_claim_id in active_ids


def test_projection_equivalent_compaction_preserves_view_signature() -> None:
    mem = Memory(store=InMemoryStore(), replica_id="agentA")
    evidence_id = mem.add_evidence(pointer="doc://x", content="hello")
    old_claim_id = mem.add_claim(
        namespace="proj",
        subject="hq",
        predicate="city",
        value="New York",
        confidence=0.7,
        evidence_ids=[evidence_id],
    )
    mem.add_claim(
        namespace="proj",
        subject="hq",
        predicate="city",
        value="San Francisco",
        confidence=0.9,
        evidence_ids=[evidence_id],
    )
    mem.retract_claim(
        target_claim_id=old_claim_id,
        evidence_ids=[evidence_id],
        reason="verified correction",
    )

    report = compact_projection_equivalent_with_report(mem.load_oplog())
    assert report.active_claims_equivalent
    assert report.conflicts_equivalent
    assert report.projections_equivalent


def test_merge_checked_authenticated_quarantines_invalid_and_revoked_traffic() -> None:
    base = OpLog([])
    trusted_claim = sign_claim(
        Claim(
            claim_id="c1",
            key=ClaimKey(namespace="auth", subject="item", predicate="status"),
            value="open",
            confidence=0.8,
            timestamp="2026-03-01T00:00:00.000000Z",
            evidence_ids=(),
            provenance={"replica_id": "agent-a"},
        ),
        secret="trusted-secret",
        key_id="k-trusted",
    )
    trusted_op = ClaimAdded(
        op_id="op-1",
        replica_id="agent-a",
        timestamp="2026-03-01T00:00:00.000000Z",
        claim=trusted_claim,
    )
    revoked_claim = sign_claim(
        Claim(
            claim_id="c2",
            key=ClaimKey(namespace="auth", subject="item", predicate="status"),
            value="closed",
            confidence=0.8,
            timestamp="2026-03-01T00:00:01.000000Z",
            evidence_ids=(),
            provenance={"replica_id": "agent-b"},
        ),
        secret="revoked-secret",
        key_id="k-revoked",
    )
    revoked_op = ClaimAdded(
        op_id="op-2",
        replica_id="agent-b",
        timestamp="2026-03-01T00:00:01.000000Z",
        claim=revoked_claim,
    )
    invalid_retraction = sign_retraction(
        ClaimRetracted(
            op_id="op-r1",
            replica_id="agent-b",
            timestamp="not-a-timestamp",
            target_claim_id="c1",
            evidence_ids=(),
            reason="tampered",
        ),
        secret="revoked-secret",
        key_id="k-revoked",
    )
    report = merge_checked_authenticated(
        base,
        OpLog([trusted_op, revoked_op, invalid_retraction]),
        key_secrets={"k-trusted": "trusted-secret", "k-revoked": "revoked-secret"},
        revoked_keys={"k-revoked"},
    )
    assert report.merged.get("op-1") == trusted_op
    assert report.merged.get("op-2") is None
    assert report.merged.get("op-r1") is None
    reasons = {item.reason for item in report.quarantined}
    assert "claim_signature_revoked" in reasons
    assert "retraction_signature_revoked" in reasons or "invalid_timestamp" in reasons
