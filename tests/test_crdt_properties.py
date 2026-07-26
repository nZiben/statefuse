from __future__ import annotations

from statefuse.merge import merge
from statefuse.model import Claim, ClaimKey, Evidence
from statefuse.oplog import OpLog
from statefuse.ops import ClaimAdded, EvidenceAdded


def _evidence_op(op_id: str, evidence_id: str, replica: str) -> EvidenceAdded:
    return EvidenceAdded(
        op_id=op_id,
        replica_id=replica,
        timestamp="2026-03-01T00:00:00.000000Z",
        evidence=Evidence(evidence_id=evidence_id, pointer=f"doc://{evidence_id}", metadata={}),
    )


def _claim_op(op_id: str, claim_id: str, value: str, replica: str) -> ClaimAdded:
    return ClaimAdded(
        op_id=op_id,
        replica_id=replica,
        timestamp="2026-03-01T00:00:00.000000Z",
        claim=Claim(
            claim_id=claim_id,
            key=ClaimKey(namespace="project", subject="deadline", predicate="date"),
            value=value,
            confidence=0.7,
            timestamp="2026-03-01T00:00:00.000000Z",
            evidence_ids=("sha256:x",),
            provenance={"replica_id": replica},
        ),
    )


def test_merge_commutative_associative_idempotent() -> None:
    oplog_a = OpLog(
        [_evidence_op("op-1", "sha256:1", "a"), _claim_op("op-2", "c1", "2026-03-25", "a")]
    )
    oplog_b = OpLog(
        [_evidence_op("op-3", "sha256:2", "b"), _claim_op("op-4", "c2", "2026-03-26", "b")]
    )
    oplog_c = OpLog([_evidence_op("op-5", "sha256:3", "c")])

    assert merge(oplog_a, oplog_b) == merge(oplog_b, oplog_a)
    assert merge(merge(oplog_a, oplog_b), oplog_c) == merge(oplog_a, merge(oplog_b, oplog_c))
    assert merge(oplog_a, oplog_a) == oplog_a


def test_duplicate_op_id_with_same_payload_is_noop() -> None:
    op = _evidence_op("op-1", "sha256:1", "a")
    oplog = OpLog([op])
    added = oplog.add(op)
    assert not added
    assert len(oplog) == 1


def test_duplicate_op_id_with_different_payload_raises() -> None:
    op1 = _evidence_op("op-1", "sha256:1", "a")
    op2 = _evidence_op("op-1", "sha256:DIFF", "a")
    oplog = OpLog([op1])
    try:
        oplog.add(op2)
    except ValueError as exc:
        assert "op_id collision" in str(exc)
    else:
        raise AssertionError("Expected ValueError for duplicate op_id with different payload.")
