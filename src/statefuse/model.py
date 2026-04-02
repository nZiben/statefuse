from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .utils import digest_json_value


JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

_PROVENANCE_AUTH_FIELDS = frozenset({"signature", "signing_key_id"})


@dataclass(frozen=True, order=True)
class ClaimKey:
    namespace: str
    subject: str
    predicate: str

    def to_dict(self) -> dict[str, str]:
        return {
            "namespace": self.namespace,
            "subject": self.subject,
            "predicate": self.predicate,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ClaimKey":
        return cls(
            namespace=str(value["namespace"]),
            subject=str(value["subject"]),
            predicate=str(value["predicate"]),
        )


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    pointer: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "pointer": self.pointer,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Evidence":
        metadata = value.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Evidence.metadata must be a mapping.")
        return cls(
            evidence_id=str(value["evidence_id"]),
            pointer=str(value["pointer"]),
            metadata=dict(metadata),
        )


def claim_provenance_body(provenance: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in provenance.items() if key not in _PROVENANCE_AUTH_FIELDS}


def claim_ref_payload(
    *,
    key: ClaimKey,
    value: JSONValue,
    confidence: float,
    timestamp: str,
    evidence_ids: tuple[str, ...],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "key": key.to_dict(),
        "value": value,
    }


def claim_ref_from_payload(payload: dict[str, Any]) -> str:
    return f"sha256:{digest_json_value(payload)}"


def derive_claim_ref(
    *,
    key: ClaimKey,
    value: JSONValue,
    confidence: float,
    timestamp: str,
    evidence_ids: tuple[str, ...],
    provenance: dict[str, Any],
) -> str:
    return claim_ref_from_payload(
        claim_ref_payload(
            key=key,
            value=value,
            confidence=confidence,
            timestamp=timestamp,
            evidence_ids=evidence_ids,
            provenance=provenance,
        )
    )


@dataclass(frozen=True)
class Claim:
    claim_id: str
    key: ClaimKey
    value: JSONValue
    confidence: float
    timestamp: str
    claim_ref: str = ""
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Claim.confidence must be between 0 and 1.")
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(self, "provenance", dict(self.provenance))
        if not self.claim_ref:
            object.__setattr__(
                self,
                "claim_ref",
                derive_claim_ref(
                    key=self.key,
                    value=self.value,
                    confidence=self.confidence,
                    timestamp=self.timestamp,
                    evidence_ids=self.evidence_ids,
                    provenance=self.provenance,
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_ref": self.claim_ref,
            "key": self.key.to_dict(),
            "value": self.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "evidence_ids": list(self.evidence_ids),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Claim":
        evidence_ids = value.get("evidence_ids", [])
        provenance = value.get("provenance", {})
        if not isinstance(evidence_ids, list):
            raise ValueError("Claim.evidence_ids must be a list.")
        if not isinstance(provenance, dict):
            raise ValueError("Claim.provenance must be a mapping.")
        return cls(
            claim_id=str(value["claim_id"]),
            claim_ref=str(value.get("claim_ref", "")),
            key=ClaimKey.from_dict(dict(value["key"])),
            value=value["value"],
            confidence=float(value["confidence"]),
            timestamp=str(value["timestamp"]),
            evidence_ids=tuple(str(item) for item in evidence_ids),
            provenance=dict(provenance),
        )


@dataclass(frozen=True)
class Decision:
    decision_id: str
    scope: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", dict(self.payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "scope": self.scope,
            "payload": dict(self.payload),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Decision":
        payload = value.get("payload", {})
        if not isinstance(payload, dict):
            raise ValueError("Decision.payload must be a mapping.")
        return cls(
            decision_id=str(value["decision_id"]),
            scope=str(value["scope"]),
            payload=dict(payload),
            timestamp=str(value.get("timestamp", "")),
        )
