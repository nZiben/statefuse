from __future__ import annotations

from statefuse.merge import merge_checked
from statefuse.model import Claim, ClaimKey, Evidence
from statefuse.oplog import OpLog
from statefuse.ops import ClaimAdded, EvidenceAdded


def _evidence(op_id: str, evidence_id: str) -> EvidenceAdded:
    return EvidenceAdded(
        op_id=op_id,
        replica_id="replica-a",
        timestamp="2026-03-01T00:00:00.000000Z",
        evidence=Evidence(evidence_id=evidence_id, pointer=f"doc://{evidence_id}", metadata={}),
    )


def _claim(op_id: str, claim_id: str, value: str) -> ClaimAdded:
    return ClaimAdded(
        op_id=op_id,
        replica_id="replica-a",
        timestamp="2026-03-01T00:00:00.000000Z",
        claim=Claim(
            claim_id=claim_id,
            key=ClaimKey(namespace="sync", subject="item", predicate="status"),
            value=value,
            confidence=0.7,
            timestamp="2026-03-01T00:00:00.000000Z",
            evidence_ids=(),
            provenance={"replica_id": "replica-a"},
        ),
    )


def test_merge_checked_records_duplicates_without_error() -> None:
    op = _evidence("op-1", "sha256:1")
    report = merge_checked(OpLog([op]), OpLog([op]))
    assert report.merged.op_ids() == ("op-1",)
    assert report.duplicate_op_ids == ("op-1",)
    assert report.quarantined == ()


def test_merge_checked_quarantines_payload_collision() -> None:
    left = _claim("op-shared", "c1", "open")
    right = _claim("op-shared", "c2", "closed")
    report = merge_checked(OpLog([left]), OpLog([right]))
    assert report.merged.get("op-shared") == left
    assert report.duplicate_op_ids == ()
    assert len(report.quarantined) == 1
    quarantined = report.quarantined[0]
    assert quarantined.op_id == "op-shared"
    assert quarantined.kept_op == left
    assert quarantined.rejected_op == right
