from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field

from .auth import claim_signature_status, retraction_signature_status
from .oplog import OpLog
from .ops import (
    AnyOp,
    ClaimAdded,
    ClaimRetracted,
    ConflictLifecycleEventAdded,
    ResolutionAdded,
)
from .utils import parse_utc_iso


def merge(oplog_a: OpLog, oplog_b: OpLog) -> OpLog:
    """CRDT merge is plain set union over immutable operation IDs."""
    merged = OpLog(oplog_a.iter_ops())
    for op in oplog_b.iter_ops():
        merged.add(op)
    return merged


@dataclass(frozen=True)
class QuarantinedOp:
    op_id: str
    kept_op: AnyOp
    rejected_op: AnyOp
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "op_id": self.op_id,
            "reason": self.reason,
            "kept_op": self.kept_op.to_dict(),
            "rejected_op": self.rejected_op.to_dict(),
        }


@dataclass(frozen=True)
class MergeReport:
    merged: OpLog
    added_op_ids: tuple[str, ...] = field(default_factory=tuple)
    duplicate_op_ids: tuple[str, ...] = field(default_factory=tuple)
    quarantined: tuple[QuarantinedOp, ...] = field(default_factory=tuple)


def merge_checked(oplog_a: OpLog, oplog_b: OpLog) -> MergeReport:
    """Merge two op-logs while quarantining invalid payload collisions."""

    merged = OpLog(oplog_a.iter_ops())
    added_op_ids: list[str] = []
    duplicate_op_ids: list[str] = []
    quarantined: list[QuarantinedOp] = []

    for op in oplog_b.iter_ops():
        existing = merged.get(op.op_id)
        if existing is None:
            merged.add(op)
            added_op_ids.append(op.op_id)
            continue
        if existing == op:
            duplicate_op_ids.append(op.op_id)
            continue
        quarantined.append(
            QuarantinedOp(
                op_id=op.op_id,
                kept_op=existing,
                rejected_op=op,
                reason="op_id collision with different payload",
            )
        )

    return MergeReport(
        merged=merged,
        added_op_ids=tuple(added_op_ids),
        duplicate_op_ids=tuple(duplicate_op_ids),
        quarantined=tuple(quarantined),
    )


def _authenticated_reason(
    op: AnyOp,
    *,
    key_secrets: Mapping[str, str],
    revoked_keys: Collection[str],
    require_signed: bool,
    require_valid_timestamps: bool,
) -> str | None:
    if require_valid_timestamps:
        try:
            parse_utc_iso(op.timestamp)
            if isinstance(op, ClaimAdded):
                parse_utc_iso(op.claim.timestamp)
            if isinstance(op, ResolutionAdded):
                parse_utc_iso(op.resolution.timestamp)
            if isinstance(op, ConflictLifecycleEventAdded):
                parse_utc_iso(op.event.timestamp)
        except Exception:
            return "invalid_timestamp"

    if require_signed and isinstance(op, (ResolutionAdded, ConflictLifecycleEventAdded)):
        return "authority_signature_unsupported"

    if isinstance(op, ClaimAdded):
        key_id = str(op.claim.provenance.get("signing_key_id", "") or "")
        if key_id and key_id in revoked_keys:
            return "claim_signature_revoked"
        status = claim_signature_status(op.claim, key_secrets=key_secrets)
        if require_signed and status != "verified":
            return f"claim_signature_{status}"
        if not require_signed and status == "invalid":
            return "claim_signature_invalid"
        return None

    if isinstance(op, ClaimRetracted):
        key_id = str(op.signing_key_id or "")
        if key_id and key_id in revoked_keys:
            return "retraction_signature_revoked"
        status = retraction_signature_status(op, key_secrets=key_secrets)
        if require_signed and status != "verified":
            return f"retraction_signature_{status}"
        if not require_signed and status == "invalid":
            return "retraction_signature_invalid"
        return None

    return None


def merge_checked_authenticated(
    oplog_a: OpLog,
    oplog_b: OpLog,
    *,
    key_secrets: Mapping[str, str],
    revoked_keys: Collection[str] = (),
    require_signed: bool = True,
    require_valid_timestamps: bool = True,
) -> MergeReport:
    """
    Merge two op-logs while quarantining payload collisions and unauthenticated traffic.

    This is a minimal authenticated sync policy, not a Byzantine protocol: it
    verifies signatures on claims and retractions, rejects revoked keys, and can
    optionally reject malformed timestamps before accepting incoming ops.
    """

    merged = OpLog(oplog_a.iter_ops())
    added_op_ids: list[str] = []
    duplicate_op_ids: list[str] = []
    quarantined: list[QuarantinedOp] = []

    for op in oplog_b.iter_ops():
        reason = _authenticated_reason(
            op,
            key_secrets=key_secrets,
            revoked_keys=revoked_keys,
            require_signed=require_signed,
            require_valid_timestamps=require_valid_timestamps,
        )
        if reason is not None:
            quarantined.append(
                QuarantinedOp(
                    op_id=op.op_id,
                    kept_op=op,
                    rejected_op=op,
                    reason=reason,
                )
            )
            continue

        existing = merged.get(op.op_id)
        if existing is None:
            merged.add(op)
            added_op_ids.append(op.op_id)
            continue
        if existing == op:
            duplicate_op_ids.append(op.op_id)
            continue
        quarantined.append(
            QuarantinedOp(
                op_id=op.op_id,
                kept_op=existing,
                rejected_op=op,
                reason="op_id collision with different payload",
            )
        )

    return MergeReport(
        merged=merged,
        added_op_ids=tuple(added_op_ids),
        duplicate_op_ids=tuple(duplicate_op_ids),
        quarantined=tuple(quarantined),
    )
