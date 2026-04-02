from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, TypeVar, cast

from .model import Claim, Decision, Evidence
from .utils import canonical_json_dumps, canonical_json_loads


def _require_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string.")
    return value


def _optional_str(data: Mapping[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string or null.")
    return value


def _list_of_str(data: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = data.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list of strings.")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{key} must be a list of strings.")
        result.append(item)
    return tuple(result)


def _mapping_value(data: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object.")
    return dict(value)


@dataclass(frozen=True)
class Op:
    op_id: str
    replica_id: str
    timestamp: str
    op_type: ClassVar[str] = "Op"

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_id": self.op_id,
            "op_type": self.op_type,
            "replica_id": self.replica_id,
            "timestamp": self.timestamp,
            **self._payload_to_dict(),
        }

    def _payload_to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnyOp":
        op_type = payload.get("op_type")
        if not isinstance(op_type, str):
            raise ValueError("op_type must be present and must be a string.")
        op_cls = _OP_TYPES.get(op_type)
        if op_cls is None:
            raise ValueError(f"Unknown op_type: {op_type}")
        return op_cls._from_dict(payload)

    @classmethod
    def from_json(cls, payload: str) -> "AnyOp":
        data = canonical_json_loads(payload)
        if not isinstance(data, dict):
            raise ValueError("Operation JSON must decode to an object.")
        return cls.from_dict(data)

    @classmethod
    def _from_dict(cls, payload: Mapping[str, Any]) -> "Op":
        raise NotImplementedError


@dataclass(frozen=True)
class EvidenceAdded(Op):
    evidence: Evidence
    op_type: ClassVar[str] = "EvidenceAdded"

    def _payload_to_dict(self) -> dict[str, Any]:
        return {"evidence": self.evidence.to_dict()}

    @classmethod
    def _from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceAdded":
        return cls(
            op_id=_require_str(payload, "op_id"),
            replica_id=_require_str(payload, "replica_id"),
            timestamp=_require_str(payload, "timestamp"),
            evidence=Evidence.from_dict(_mapping_value(payload, "evidence")),
        )


@dataclass(frozen=True)
class ClaimAdded(Op):
    claim: Claim
    op_type: ClassVar[str] = "ClaimAdded"

    def _payload_to_dict(self) -> dict[str, Any]:
        return {"claim": self.claim.to_dict()}

    @classmethod
    def _from_dict(cls, payload: Mapping[str, Any]) -> "ClaimAdded":
        return cls(
            op_id=_require_str(payload, "op_id"),
            replica_id=_require_str(payload, "replica_id"),
            timestamp=_require_str(payload, "timestamp"),
            claim=Claim.from_dict(_mapping_value(payload, "claim")),
        )


@dataclass(frozen=True)
class ClaimRetracted(Op):
    target_claim_id: str | None = None
    target_claim_ref: str | None = None
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    reason: str = ""
    supersedes_claim_id: str | None = None
    supersedes_claim_ref: str | None = None
    signing_key_id: str | None = None
    signature: str | None = None
    op_type: ClassVar[str] = "ClaimRetracted"

    def __post_init__(self) -> None:
        if self.target_claim_id is None and self.target_claim_ref is None:
            raise ValueError("ClaimRetracted requires target_claim_id or target_claim_ref.")
        if (self.signing_key_id is None) != (self.signature is None):
            raise ValueError("signing_key_id and signature must be provided together.")
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))

    def _payload_to_dict(self) -> dict[str, Any]:
        return {
            "target_claim_id": self.target_claim_id,
            "target_claim_ref": self.target_claim_ref,
            "evidence_ids": list(self.evidence_ids),
            "reason": self.reason,
            "supersedes_claim_id": self.supersedes_claim_id,
            "supersedes_claim_ref": self.supersedes_claim_ref,
            "signing_key_id": self.signing_key_id,
            "signature": self.signature,
        }

    @classmethod
    def _from_dict(cls, payload: Mapping[str, Any]) -> "ClaimRetracted":
        return cls(
            op_id=_require_str(payload, "op_id"),
            replica_id=_require_str(payload, "replica_id"),
            timestamp=_require_str(payload, "timestamp"),
            target_claim_id=_optional_str(payload, "target_claim_id"),
            target_claim_ref=_optional_str(payload, "target_claim_ref"),
            evidence_ids=_list_of_str(payload, "evidence_ids"),
            reason=_require_str(payload, "reason"),
            supersedes_claim_id=_optional_str(payload, "supersedes_claim_id"),
            supersedes_claim_ref=_optional_str(payload, "supersedes_claim_ref"),
            signing_key_id=_optional_str(payload, "signing_key_id"),
            signature=_optional_str(payload, "signature"),
        )


@dataclass(frozen=True)
class DecisionAdded(Op):
    decision: Decision
    op_type: ClassVar[str] = "DecisionAdded"

    def _payload_to_dict(self) -> dict[str, Any]:
        return {"decision": self.decision.to_dict()}

    @classmethod
    def _from_dict(cls, payload: Mapping[str, Any]) -> "DecisionAdded":
        return cls(
            op_id=_require_str(payload, "op_id"),
            replica_id=_require_str(payload, "replica_id"),
            timestamp=_require_str(payload, "timestamp"),
            decision=Decision.from_dict(_mapping_value(payload, "decision")),
        )


AnyOp = EvidenceAdded | ClaimAdded | ClaimRetracted | DecisionAdded

_OpType = TypeVar("_OpType", bound=Op)

_OP_TYPES: dict[str, type[_OpType]] = {
    EvidenceAdded.op_type: cast(type[_OpType], EvidenceAdded),
    ClaimAdded.op_type: cast(type[_OpType], ClaimAdded),
    ClaimRetracted.op_type: cast(type[_OpType], ClaimRetracted),
    DecisionAdded.op_type: cast(type[_OpType], DecisionAdded),
}
