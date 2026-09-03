from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from .auth import sign_claim
from .conflict import ConflictDetector, ConflictSet, PredicateRegistry
from .materialize import MemoryState, materialize
from .merge import merge
from .model import (
    Claim,
    ClaimKey,
    Decision,
    Derivation,
    Evidence,
    JSONValue,
    ResolutionRecord,
    Source,
    ValidityInterval,
    derive_claim_ref,
)
from .oplog import OpLog
from .ops import (
    AnyOp,
    ClaimAdded,
    ClaimRetracted,
    DecisionAdded,
    DerivationAdded,
    EvidenceAdded,
    ResolutionAdded,
    SourceAdded,
)
from .resolver import HeuristicResolver, Resolver, ViewConstraints
from .store import InMemoryStore, OpStore
from .utils import content_addressed_op_id, digest_content, digest_json_value, new_uuid, utc_now_iso
from .view import Projection
from .view import build_view as build_projection

OpIdMode = Literal["uuid4", "content-addressed"]


class Memory:
    """High-level API for writing and projecting mergeable memory."""

    def __init__(
        self,
        store: OpStore | None = None,
        replica_id: str = "default",
        *,
        op_id_mode: OpIdMode = "uuid4",
        predicate_registry: PredicateRegistry | None = None,
        conflict_detectors: Sequence[ConflictDetector] = (),
    ) -> None:
        self.store = store or InMemoryStore()
        self.replica_id = replica_id
        if op_id_mode not in {"uuid4", "content-addressed"}:
            raise ValueError("op_id_mode must be 'uuid4' or 'content-addressed'.")
        self.op_id_mode = op_id_mode
        self.predicate_registry = predicate_registry or PredicateRegistry()
        self.conflict_detectors = tuple(conflict_detectors)

    def append_op(self, op: AnyOp) -> bool:
        return self.store.append(op)

    def load_oplog(self) -> OpLog:
        return self.store.load_oplog()

    def add_source(
        self,
        *,
        source_type: str,
        uri: str | None = None,
        system: str | None = None,
        actor_id: str | None = None,
        session_id: str | None = None,
        message_id: str | None = None,
        timestamp: str | None = None,
        metadata: dict[str, Any] | None = None,
        source_id: str | None = None,
        op_id: str | None = None,
    ) -> str:
        source_metadata = dict(metadata or {})
        source_body = {
            "source_type": source_type,
            "uri": uri,
            "system": system,
            "actor_id": actor_id,
            "session_id": session_id,
            "message_id": message_id,
            "timestamp": timestamp,
            "metadata": source_metadata,
        }
        source = Source(
            source_id=source_id or f"sha256:{digest_json_value(source_body)}",
            **source_body,
        )
        op_timestamp = utc_now_iso()
        op = SourceAdded(
            op_id=op_id
            or self._new_op_id("SourceAdded", op_timestamp, {"source": source.to_dict()}),
            replica_id=self.replica_id,
            timestamp=op_timestamp,
            source=source,
        )
        self.store.append(op)
        return source.source_id

    def add_evidence(
        self,
        pointer: str | None = None,
        *,
        content: Any = None,
        metadata: dict[str, Any] | None = None,
        source_id: str | None = None,
        content_digest: str | None = None,
        evidence_id: str | None = None,
        op_id: str | None = None,
    ) -> str:
        if evidence_id is None:
            if source_id is None and content_digest is None:
                digest = (
                    digest_content(content)
                    if content is not None
                    else digest_json_value({"pointer": pointer})
                )
            else:
                identity_digest = content_digest
                if identity_digest is None and content is not None:
                    identity_digest = f"sha256:{digest_content(content)}"
                digest = digest_json_value(
                    {
                        "pointer": pointer,
                        "source_id": source_id,
                        "content_digest": identity_digest,
                    }
                )
            evidence_id = f"sha256:{digest}"
        timestamp = utc_now_iso()
        evidence = Evidence(
            evidence_id=evidence_id,
            pointer=pointer,
            metadata=dict(metadata or {}),
            source_id=source_id,
            content_digest=content_digest,
        )
        op = EvidenceAdded(
            op_id=op_id
            or self._new_op_id(
                "EvidenceAdded", timestamp, {"evidence": evidence.to_dict()}
            ),
            replica_id=self.replica_id,
            timestamp=timestamp,
            evidence=evidence,
        )
        self.store.append(op)
        return evidence.evidence_id

    def add_claim(
        self,
        *,
        namespace: str,
        subject: str,
        predicate: str,
        value: JSONValue,
        confidence: float,
        evidence_ids: list[str] | tuple[str, ...],
        provenance: dict[str, Any] | None = None,
        claim_id: str | None = None,
        claim_ref: str | None = None,
        op_id: str | None = None,
        signing_key: str | None = None,
        signing_key_id: str | None = None,
        validity: ValidityInterval | None = None,
        derivation_id: str | None = None,
        kind: str = "fact",
        context: dict[str, JSONValue] | None = None,
    ) -> str:
        if (signing_key is None) != (signing_key_id is None):
            raise ValueError("signing_key and signing_key_id must be provided together.")
        claim_provenance = dict(provenance or {})
        claim_provenance.setdefault("replica_id", self.replica_id)
        timestamp = utc_now_iso()
        normalized_ref_value = self.predicate_registry.claim_ref_value(predicate, value)
        claim = Claim(
            claim_id=claim_id or new_uuid(),
            key=ClaimKey(namespace=namespace, subject=subject, predicate=predicate),
            value=value,
            confidence=confidence,
            timestamp=timestamp,
            claim_ref=claim_ref
            or derive_claim_ref(
                key=ClaimKey(namespace=namespace, subject=subject, predicate=predicate),
                value=normalized_ref_value,
                confidence=confidence,
                timestamp=timestamp,
                evidence_ids=tuple(evidence_ids),
                provenance=claim_provenance,
                kind=kind,
                context=context,
            ),
            evidence_ids=tuple(evidence_ids),
            provenance=claim_provenance,
            validity=validity,
            derivation_id=derivation_id,
            kind=kind,
            context=dict(context or {}),
        )
        if signing_key is not None and signing_key_id is not None:
            claim = sign_claim(claim, secret=signing_key, key_id=signing_key_id)
        op = ClaimAdded(
            op_id=op_id or self._new_op_id("ClaimAdded", timestamp, {"claim": claim.to_dict()}),
            replica_id=self.replica_id,
            timestamp=timestamp,
            claim=claim,
        )
        self.store.append(op)
        return claim.claim_id

    def retract_claim(
        self,
        *,
        target_claim_id: str | None = None,
        target_claim_ref: str | None = None,
        evidence_ids: list[str] | tuple[str, ...],
        reason: str,
        supersedes_claim_id: str | None = None,
        supersedes_claim_ref: str | None = None,
        op_id: str | None = None,
        signing_key: str | None = None,
        signing_key_id: str | None = None,
    ) -> str:
        if target_claim_id is None and target_claim_ref is None:
            raise ValueError("retract_claim requires target_claim_id or target_claim_ref.")
        if (signing_key is None) != (signing_key_id is None):
            raise ValueError("signing_key and signing_key_id must be provided together.")
        timestamp = utc_now_iso()
        op = ClaimRetracted(
            op_id=op_id
            or self._new_op_id(
                "ClaimRetracted",
                timestamp,
                {
                    "target_claim_id": target_claim_id,
                    "target_claim_ref": target_claim_ref,
                    "evidence_ids": list(evidence_ids),
                    "reason": reason,
                    "supersedes_claim_id": supersedes_claim_id,
                    "supersedes_claim_ref": supersedes_claim_ref,
                },
            ),
            replica_id=self.replica_id,
            timestamp=timestamp,
            target_claim_id=target_claim_id,
            target_claim_ref=target_claim_ref,
            evidence_ids=tuple(evidence_ids),
            reason=reason,
            supersedes_claim_id=supersedes_claim_id,
            supersedes_claim_ref=supersedes_claim_ref,
        )
        if signing_key is not None and signing_key_id is not None:
            from .auth import sign_retraction

            op = sign_retraction(op, secret=signing_key, key_id=signing_key_id)
        self.store.append(op)
        return op.op_id

    def add_resolution(
        self,
        *,
        conflict_ref: str,
        observed_conflict_id: str,
        selected_claim_ids: list[str] | tuple[str, ...],
        resolution_type: str,
        reason: str,
        actor_id: str,
        rejected_claim_ids: list[str] | tuple[str, ...] = (),
        retained_claim_ids: list[str] | tuple[str, ...] = (),
        evidence_ids: list[str] | tuple[str, ...] = (),
        scope: str | None = None,
        valid_from: str | None = None,
        valid_until: str | None = None,
        metadata: dict[str, Any] | None = None,
        outcome: str = "select",
        resolution_id: str | None = None,
        op_id: str | None = None,
    ) -> str:
        timestamp = utc_now_iso()
        resolution = ResolutionRecord(
            resolution_id=resolution_id or new_uuid(),
            conflict_ref=conflict_ref,
            observed_conflict_id=observed_conflict_id,
            selected_claim_ids=tuple(selected_claim_ids),
            rejected_claim_ids=tuple(rejected_claim_ids),
            retained_claim_ids=tuple(retained_claim_ids),
            resolution_type=resolution_type,
            reason=reason,
            evidence_ids=tuple(evidence_ids),
            actor_id=actor_id,
            timestamp=timestamp,
            scope=scope,
            valid_from=valid_from,
            valid_until=valid_until,
            metadata=dict(metadata or {}),
            outcome=outcome,
        )
        op = ResolutionAdded(
            op_id=op_id
            or self._new_op_id(
                "ResolutionAdded", timestamp, {"resolution": resolution.to_dict()}
            ),
            replica_id=self.replica_id,
            timestamp=timestamp,
            resolution=resolution,
        )
        self.store.append(op)
        return resolution.resolution_id

    def add_decision(
        self,
        *,
        scope: str,
        payload: dict[str, Any],
        decision_id: str | None = None,
        op_id: str | None = None,
    ) -> str:
        timestamp = utc_now_iso()
        decision = Decision(
            decision_id=decision_id or new_uuid(),
            scope=scope,
            payload=payload,
            timestamp=timestamp,
        )
        op = DecisionAdded(
            op_id=op_id
            or self._new_op_id(
                "DecisionAdded", timestamp, {"decision": decision.to_dict()}
            ),
            replica_id=self.replica_id,
            timestamp=timestamp,
            decision=decision,
        )
        self.store.append(op)
        return decision.decision_id

    def add_derivation(
        self,
        *,
        rule_id: str,
        input_claim_ids: list[str] | tuple[str, ...],
        output_claim_ids: list[str] | tuple[str, ...],
        engine: str,
        explanation: str,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
        derivation_id: str | None = None,
        op_id: str | None = None,
    ) -> str:
        timestamp = utc_now_iso()
        derivation = Derivation(
            derivation_id=derivation_id or new_uuid(),
            rule_id=rule_id,
            input_claim_ids=tuple(input_claim_ids),
            output_claim_ids=tuple(output_claim_ids),
            engine=engine,
            explanation=explanation,
            timestamp=timestamp,
            confidence=confidence,
            metadata=dict(metadata or {}),
        )
        op = DerivationAdded(
            op_id=op_id
            or self._new_op_id(
                "DerivationAdded", timestamp, {"derivation": derivation.to_dict()}
            ),
            replica_id=self.replica_id,
            timestamp=timestamp,
            derivation=derivation,
        )
        self.store.append(op)
        return derivation.derivation_id

    def materialize(
        self,
        predicate_registry: PredicateRegistry | None = None,
        *,
        conflict_detectors: Sequence[ConflictDetector] | None = None,
        valid_at: str | None = None,
        context: dict[str, JSONValue] | None = None,
    ) -> MemoryState:
        return materialize(
            self.load_oplog(),
            predicate_registry=predicate_registry or self.predicate_registry,
            conflict_detectors=(
                self.conflict_detectors if conflict_detectors is None else conflict_detectors
            ),
            valid_at=valid_at,
            context=context,
        )

    def find_conflicts(
        self,
        *,
        valid_at: str | None = None,
        applicability_context: dict[str, JSONValue] | None = None,
        **filters: Any,
    ) -> tuple[ConflictSet, ...]:
        return self.materialize(
            valid_at=valid_at, context=applicability_context
        ).find_conflicts(**filters)

    def merge_from(self, other_store_or_oplog: OpStore | OpLog) -> OpLog:
        if isinstance(other_store_or_oplog, OpLog):
            other_oplog = other_store_or_oplog
        elif hasattr(other_store_or_oplog, "load_oplog"):
            other_oplog = other_store_or_oplog.load_oplog()  # type: ignore[assignment]
        else:
            raise TypeError("Expected OpLog or OpStore-compatible object.")
        merged = merge(self.load_oplog(), other_oplog)
        for op in merged.iter_ops():
            self.store.append(op)
        return merged

    def build_view(
        self,
        constraints: ViewConstraints,
        resolver: Resolver | None = None,
        predicate_registry: PredicateRegistry | None = None,
    ) -> Projection:
        state = self.materialize(
            predicate_registry=predicate_registry,
            valid_at=constraints.valid_at,
            context=constraints.context,
        )
        return build_projection(
            state=state,
            constraints=constraints,
            resolver=resolver or HeuristicResolver(),
        )

    def claim_ref_for(
        self,
        *,
        namespace: str,
        subject: str,
        predicate: str,
        value: JSONValue,
        confidence: float,
        timestamp: str,
        evidence_ids: list[str] | tuple[str, ...],
        provenance: dict[str, Any] | None = None,
        kind: str = "fact",
        context: dict[str, JSONValue] | None = None,
    ) -> str:
        claim_provenance = dict(provenance or {})
        claim_provenance.setdefault("replica_id", self.replica_id)
        normalized_ref_value = self.predicate_registry.claim_ref_value(predicate, value)
        return derive_claim_ref(
            key=ClaimKey(namespace=namespace, subject=subject, predicate=predicate),
            value=normalized_ref_value,
            confidence=confidence,
            timestamp=timestamp,
            evidence_ids=tuple(evidence_ids),
            provenance=claim_provenance,
            kind=kind,
            context=context,
        )

    def _new_op_id(self, op_type: str, timestamp: str, payload: dict[str, Any]) -> str:
        if self.op_id_mode == "uuid4":
            return new_uuid()
        return content_addressed_op_id(
            op_type=op_type,
            replica_id=self.replica_id,
            timestamp=timestamp,
            payload=payload,
        )
