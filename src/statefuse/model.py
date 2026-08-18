from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .utils import digest_json_value, parse_utc_iso

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

_PROVENANCE_AUTH_FIELDS = frozenset({"signature", "signing_key_id"})


def _required_string(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        raise ValueError(f"{key} must be a string.")
    return item


def _optional_string(value: dict[str, Any], key: str) -> str | None:
    item = value.get(key)
    if item is not None and not isinstance(item, str):
        raise ValueError(f"{key} must be a string or null.")
    return item


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
    def from_dict(cls, value: dict[str, Any]) -> ClaimKey:
        return cls(
            namespace=str(value["namespace"]),
            subject=str(value["subject"]),
            predicate=str(value["predicate"]),
        )


@dataclass(frozen=True)
class Source:
    source_id: str
    source_type: str
    uri: str | None = None
    system: str | None = None
    actor_id: str | None = None
    session_id: str | None = None
    message_id: str | None = None
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("Source.source_id must be present.")
        if not isinstance(self.source_type, str) or not self.source_type:
            raise ValueError("Source.source_type must be present.")
        optional_strings = (
            self.uri,
            self.system,
            self.actor_id,
            self.session_id,
            self.message_id,
            self.timestamp,
        )
        if any(item is not None and not isinstance(item, str) for item in optional_strings):
            raise ValueError("Source optional identifiers and timestamp must be strings or null.")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "uri": self.uri,
            "system": self.system,
            "actor_id": self.actor_id,
            "session_id": self.session_id,
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Source:
        metadata = value.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Source.metadata must be a mapping.")
        return cls(
            source_id=_required_string(value, "source_id"),
            source_type=_required_string(value, "source_type"),
            uri=_optional_string(value, "uri"),
            system=_optional_string(value, "system"),
            actor_id=_optional_string(value, "actor_id"),
            session_id=_optional_string(value, "session_id"),
            message_id=_optional_string(value, "message_id"),
            timestamp=_optional_string(value, "timestamp"),
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    pointer: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_id: str | None = None
    content_digest: str | None = None

    def __post_init__(self) -> None:
        if self.source_id is not None and not isinstance(self.source_id, str):
            raise ValueError("Evidence.source_id must be a string or null.")
        if self.content_digest is not None and not isinstance(self.content_digest, str):
            raise ValueError("Evidence.content_digest must be a string or null.")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "evidence_id": self.evidence_id,
            "pointer": self.pointer,
            "metadata": dict(self.metadata),
        }
        if self.source_id is not None:
            payload["source_id"] = self.source_id
        if self.content_digest is not None:
            payload["content_digest"] = self.content_digest
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Evidence:
        metadata = value.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Evidence.metadata must be a mapping.")
        return cls(
            evidence_id=str(value["evidence_id"]),
            pointer=None if value.get("pointer") is None else str(value["pointer"]),
            metadata=dict(metadata),
            source_id=_optional_string(value, "source_id"),
            content_digest=_optional_string(value, "content_digest"),
        )


@dataclass(frozen=True)
class ValidityInterval:
    valid_from: str | None = None
    valid_until: str | None = None

    def __post_init__(self) -> None:
        if self.valid_from is not None and not isinstance(self.valid_from, str):
            raise ValueError("ValidityInterval.valid_from must be a string or null.")
        if self.valid_until is not None and not isinstance(self.valid_until, str):
            raise ValueError("ValidityInterval.valid_until must be a string or null.")
        if self.valid_from is not None and self.valid_until is not None:
            if parse_utc_iso(self.valid_from) > parse_utc_iso(self.valid_until):
                raise ValueError("ValidityInterval.valid_from must not be after valid_until.")

    def to_dict(self) -> dict[str, str | None]:
        return {"valid_from": self.valid_from, "valid_until": self.valid_until}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ValidityInterval:
        return cls(
            valid_from=_optional_string(value, "valid_from"),
            valid_until=_optional_string(value, "valid_until"),
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
    kind: str = "fact",
    context: dict[str, JSONValue] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "key": key.to_dict(),
        "value": value,
    }
    if kind != "fact":
        payload["kind"] = kind
    if context:
        payload["context"] = dict(context)
    return payload


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
    kind: str = "fact",
    context: dict[str, JSONValue] | None = None,
) -> str:
    return claim_ref_from_payload(
        claim_ref_payload(
            key=key,
            value=value,
            confidence=confidence,
            timestamp=timestamp,
            evidence_ids=evidence_ids,
            provenance=provenance,
            kind=kind,
            context=context,
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
    validity: ValidityInterval | None = None
    derivation_id: str | None = None
    kind: str = "fact"
    context: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Claim.confidence must be between 0 and 1.")
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(self, "provenance", dict(self.provenance))
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("Claim.kind must be a non-empty string.")
        if not isinstance(self.context, dict) or any(
            not isinstance(key, str) or not key for key in self.context
        ):
            raise ValueError("Claim.context must be a mapping with non-empty string keys.")
        object.__setattr__(self, "context", dict(self.context))
        if self.validity is not None and not isinstance(self.validity, ValidityInterval):
            raise ValueError("Claim.validity must be a ValidityInterval or null.")
        if self.derivation_id is not None and not isinstance(self.derivation_id, str):
            raise ValueError("Claim.derivation_id must be a string or null.")
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
                    kind=self.kind,
                    context=self.context,
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "claim_id": self.claim_id,
            "claim_ref": self.claim_ref,
            "key": self.key.to_dict(),
            "value": self.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "evidence_ids": list(self.evidence_ids),
            "provenance": dict(self.provenance),
        }
        if self.validity is not None:
            payload["validity"] = self.validity.to_dict()
        if self.derivation_id is not None:
            payload["derivation_id"] = self.derivation_id
        if self.kind != "fact":
            payload["kind"] = self.kind
        if self.context:
            payload["context"] = dict(self.context)
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Claim:
        evidence_ids = value.get("evidence_ids", [])
        provenance = value.get("provenance", {})
        context = value.get("context", {})
        if not isinstance(evidence_ids, list):
            raise ValueError("Claim.evidence_ids must be a list.")
        if not isinstance(provenance, dict):
            raise ValueError("Claim.provenance must be a mapping.")
        if not isinstance(context, dict):
            raise ValueError("Claim.context must be a mapping.")
        validity = value.get("validity")
        if validity is not None and not isinstance(validity, dict):
            raise ValueError("Claim.validity must be an object or null.")
        return cls(
            claim_id=str(value["claim_id"]),
            claim_ref=str(value.get("claim_ref", "")),
            key=ClaimKey.from_dict(dict(value["key"])),
            value=value["value"],
            confidence=float(value["confidence"]),
            timestamp=str(value["timestamp"]),
            evidence_ids=tuple(str(item) for item in evidence_ids),
            provenance=dict(provenance),
            validity=ValidityInterval.from_dict(validity) if validity is not None else None,
            derivation_id=_optional_string(value, "derivation_id"),
            kind=value.get("kind", "fact"),
            context=dict(context),
        )


@dataclass(frozen=True)
class Derivation:
    derivation_id: str
    rule_id: str
    input_claim_ids: tuple[str, ...]
    output_claim_ids: tuple[str, ...]
    engine: str
    explanation: str
    timestamp: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required = {
            "derivation_id": self.derivation_id,
            "rule_id": self.rule_id,
            "engine": self.engine,
            "timestamp": self.timestamp,
        }
        if any(not isinstance(item, str) or not item for item in required.values()):
            raise ValueError("Derivation identifiers, rule, engine, and timestamp are required.")
        if not isinstance(self.explanation, str):
            raise ValueError("Derivation.explanation must be a string.")
        if not all(isinstance(item, str) for item in self.input_claim_ids):
            raise ValueError("Derivation.input_claim_ids must contain strings.")
        if not all(isinstance(item, str) for item in self.output_claim_ids):
            raise ValueError("Derivation.output_claim_ids must contain strings.")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Derivation.confidence must be between 0 and 1.")
        object.__setattr__(self, "input_claim_ids", tuple(self.input_claim_ids))
        object.__setattr__(self, "output_claim_ids", tuple(self.output_claim_ids))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "derivation_id": self.derivation_id,
            "rule_id": self.rule_id,
            "input_claim_ids": list(self.input_claim_ids),
            "output_claim_ids": list(self.output_claim_ids),
            "engine": self.engine,
            "explanation": self.explanation,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Derivation:
        input_claim_ids = value.get("input_claim_ids")
        output_claim_ids = value.get("output_claim_ids")
        metadata = value.get("metadata", {})
        if not isinstance(input_claim_ids, list) or not all(
            isinstance(item, str) for item in input_claim_ids
        ):
            raise ValueError("Derivation.input_claim_ids must be a list of strings.")
        if not isinstance(output_claim_ids, list) or not all(
            isinstance(item, str) for item in output_claim_ids
        ):
            raise ValueError("Derivation.output_claim_ids must be a list of strings.")
        if not isinstance(metadata, dict):
            raise ValueError("Derivation.metadata must be a mapping.")
        confidence = value.get("confidence")
        return cls(
            derivation_id=_required_string(value, "derivation_id"),
            rule_id=_required_string(value, "rule_id"),
            input_claim_ids=tuple(input_claim_ids),
            output_claim_ids=tuple(output_claim_ids),
            engine=_required_string(value, "engine"),
            explanation=_required_string(value, "explanation"),
            timestamp=_required_string(value, "timestamp"),
            confidence=None if confidence is None else float(confidence),
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class ResolutionRecord:
    resolution_id: str
    conflict_ref: str
    observed_conflict_id: str
    selected_claim_ids: tuple[str, ...]
    rejected_claim_ids: tuple[str, ...]
    retained_claim_ids: tuple[str, ...]
    resolution_type: str
    reason: str
    evidence_ids: tuple[str, ...]
    actor_id: str
    timestamp: str
    scope: str | None = None
    valid_from: str | None = None
    valid_until: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    outcome: str = "select"

    def __post_init__(self) -> None:
        groups = {
            "selected_claim_ids": tuple(self.selected_claim_ids),
            "rejected_claim_ids": tuple(self.rejected_claim_ids),
            "retained_claim_ids": tuple(self.retained_claim_ids),
        }
        for name, claim_ids in groups.items():
            if not all(isinstance(item, str) for item in claim_ids):
                raise ValueError(f"ResolutionRecord.{name} must contain strings.")
            if len(claim_ids) != len(set(claim_ids)):
                raise ValueError(f"ResolutionRecord.{name} must not contain duplicates.")
            object.__setattr__(self, name, tuple(sorted(claim_ids)))
        if not all(isinstance(item, str) for item in self.evidence_ids):
            raise ValueError("ResolutionRecord.evidence_ids must contain strings.")
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(self, "metadata", dict(self.metadata))
        selected = set(self.selected_claim_ids)
        rejected = set(self.rejected_claim_ids)
        retained = set(self.retained_claim_ids)
        if selected & rejected or selected & retained or rejected & retained:
            raise ValueError("ResolutionRecord claim ID sets must not overlap.")
        identifiers = (self.resolution_id, self.conflict_ref, self.observed_conflict_id)
        if any(not isinstance(item, str) or not item for item in identifiers):
            raise ValueError("ResolutionRecord identifiers must be present.")
        authority = (self.actor_id, self.reason, self.resolution_type)
        if any(not isinstance(item, str) or not item for item in authority):
            raise ValueError("ResolutionRecord actor_id, reason, and resolution_type are required.")
        if not isinstance(self.timestamp, str) or not self.timestamp:
            raise ValueError("ResolutionRecord.timestamp is required.")
        if self.scope is not None and not isinstance(self.scope, str):
            raise ValueError("ResolutionRecord.scope must be a string or null.")
        if self.outcome not in {"select", "replace", "preserve", "merge", "abstain"}:
            raise ValueError(f"Unknown resolution outcome: {self.outcome}")
        if self.outcome in {"preserve", "abstain"} and (selected or rejected):
            raise ValueError(
                f"Resolution outcome {self.outcome!r} cannot select or reject claims."
            )
        if self.outcome == "merge" and len(selected) != 1:
            raise ValueError("Resolution outcome 'merge' requires one merged claim.")
        parse_utc_iso(self.timestamp)
        ValidityInterval(self.valid_from, self.valid_until)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "resolution_id": self.resolution_id,
            "conflict_ref": self.conflict_ref,
            "observed_conflict_id": self.observed_conflict_id,
            "selected_claim_ids": list(self.selected_claim_ids),
            "rejected_claim_ids": list(self.rejected_claim_ids),
            "retained_claim_ids": list(self.retained_claim_ids),
            "resolution_type": self.resolution_type,
            "reason": self.reason,
            "evidence_ids": list(self.evidence_ids),
            "actor_id": self.actor_id,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }
        if self.scope is not None:
            payload["scope"] = self.scope
        if self.valid_from is not None:
            payload["valid_from"] = self.valid_from
        if self.valid_until is not None:
            payload["valid_until"] = self.valid_until
        if self.outcome != "select":
            payload["outcome"] = self.outcome
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ResolutionRecord:
        metadata = value.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("ResolutionRecord.metadata must be a mapping.")

        def claim_ids(key: str) -> tuple[str, ...]:
            items = value.get(key)
            if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
                raise ValueError(f"ResolutionRecord.{key} must be a list of strings.")
            return tuple(items)

        return cls(
            resolution_id=_required_string(value, "resolution_id"),
            conflict_ref=_required_string(value, "conflict_ref"),
            observed_conflict_id=_required_string(value, "observed_conflict_id"),
            selected_claim_ids=claim_ids("selected_claim_ids"),
            rejected_claim_ids=claim_ids("rejected_claim_ids"),
            retained_claim_ids=claim_ids("retained_claim_ids"),
            resolution_type=_required_string(value, "resolution_type"),
            reason=_required_string(value, "reason"),
            evidence_ids=claim_ids("evidence_ids"),
            actor_id=_required_string(value, "actor_id"),
            timestamp=_required_string(value, "timestamp"),
            scope=_optional_string(value, "scope"),
            valid_from=_optional_string(value, "valid_from"),
            valid_until=_optional_string(value, "valid_until"),
            metadata=dict(metadata),
            outcome=str(value.get("outcome", "select")),
        )


_LIFECYCLE_STATUSES = frozenset({"open", "deferred", "resolved", "reopened", "invalidated"})


@dataclass(frozen=True)
class ConflictLifecycleEvent:
    event_id: str
    conflict_ref: str
    observed_conflict_id: str
    status: str
    timestamp: str
    reason: str
    actor_id: str | None = None
    scope: str | None = None
    resolution_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        identifiers = (self.event_id, self.conflict_ref, self.observed_conflict_id)
        if any(not isinstance(item, str) or not item for item in identifiers):
            raise ValueError("ConflictLifecycleEvent identifiers must be present.")
        if self.status not in _LIFECYCLE_STATUSES:
            raise ValueError(f"Unknown conflict lifecycle status: {self.status}")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("ConflictLifecycleEvent.reason is required.")
        if not isinstance(self.timestamp, str) or not self.timestamp:
            raise ValueError("ConflictLifecycleEvent timestamp and reason are required.")
        optional_strings = (self.actor_id, self.scope, self.resolution_id)
        if any(item is not None and not isinstance(item, str) for item in optional_strings):
            raise ValueError("ConflictLifecycleEvent optional links must be strings or null.")
        parse_utc_iso(self.timestamp)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "event_id": self.event_id,
            "conflict_ref": self.conflict_ref,
            "observed_conflict_id": self.observed_conflict_id,
            "status": self.status,
            "timestamp": self.timestamp,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }
        if self.actor_id is not None:
            payload["actor_id"] = self.actor_id
        if self.scope is not None:
            payload["scope"] = self.scope
        if self.resolution_id is not None:
            payload["resolution_id"] = self.resolution_id
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ConflictLifecycleEvent:
        metadata = value.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("ConflictLifecycleEvent.metadata must be a mapping.")
        return cls(
            event_id=_required_string(value, "event_id"),
            conflict_ref=_required_string(value, "conflict_ref"),
            observed_conflict_id=_required_string(value, "observed_conflict_id"),
            status=_required_string(value, "status"),
            timestamp=_required_string(value, "timestamp"),
            reason=_required_string(value, "reason"),
            actor_id=_optional_string(value, "actor_id"),
            scope=_optional_string(value, "scope"),
            resolution_id=_optional_string(value, "resolution_id"),
            metadata=dict(metadata),
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
    def from_dict(cls, value: dict[str, Any]) -> Decision:
        payload = value.get("payload", {})
        if not isinstance(payload, dict):
            raise ValueError("Decision.payload must be a mapping.")
        return cls(
            decision_id=str(value["decision_id"]),
            scope=str(value["scope"]),
            payload=dict(payload),
            timestamp=str(value.get("timestamp", "")),
        )
