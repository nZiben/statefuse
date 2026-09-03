from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import fields, replace
from typing import Any, Literal

from .auth import sign_claim
from .conflict import ConflictDetector, ConflictSet, PredicateRegistry
from .materialize import (
    MATERIALIZATION_CHECKPOINT_VERSION,
    MaterializationDelta,
    MemoryState,
    apply_operations_incrementally,
    materialize,
    memory_state_from_checkpoint,
    memory_state_to_checkpoint,
)
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
from .replica import (
    ReplicaDelta,
    ReplicaManifest,
    ReplicaProgress,
    ReplicaSyncReport,
    advance_replica_progress,
)
from .resolver import Resolver, ViewConstraints
from .store import (
    BatchOpStore,
    IncrementalOpStore,
    InMemoryStore,
    MaterializationCheckpoint,
    OpStore,
    ReplicaProgressStore,
    StoreDelta,
)
from .utils import content_addressed_op_id, digest_content, digest_json_value, new_uuid, utc_now_iso
from .view import Projection
from .view import build_view as build_projection

OpIdMode = Literal["uuid4", "content-addressed"]
_CHECKPOINT_INTERVAL_OPS = 100


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
        materialization_config_token: str | None = None,
    ) -> None:
        self.store = store if store is not None else InMemoryStore()
        self.replica_id = replica_id
        if op_id_mode not in {"uuid4", "content-addressed"}:
            raise ValueError("op_id_mode must be 'uuid4' or 'content-addressed'.")
        self.op_id_mode = op_id_mode
        self.predicate_registry = (
            predicate_registry if predicate_registry is not None else PredicateRegistry()
        )
        self.conflict_detectors = tuple(conflict_detectors)
        self._automatic_materialization_config = (
            materialization_config_token is None
            and predicate_registry is None
            and not conflict_detectors
        )
        self._materialization_config_token = (
            materialization_config_token
            if materialization_config_token is not None
            else "default" if self._automatic_materialization_config else None
        )
        self._materialized_state: MemoryState | None = None
        self._materialized_registry: PredicateRegistry | None = None
        self._materialized_registry_revision = -1
        self._materialization_cursor = 0
        self._materialization_history: deque[MaterializationDelta] = deque(maxlen=1024)
        self._ops_since_checkpoint = 0

    def append_op(self, op: AnyOp) -> bool:
        return self.store.append(op)

    def commit_batch(self, ops: Sequence[AnyOp]) -> tuple[bool, ...]:
        """Commit one already-extracted logical event as a store batch."""

        if not isinstance(self.store, BatchOpStore):
            raise TypeError("The configured operation store does not support batch commits.")
        return self.store.append_many(*tuple(ops))

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
            or self._new_op_id("EvidenceAdded", timestamp, {"evidence": evidence.to_dict()}),
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
            or self._new_op_id("ResolutionAdded", timestamp, {"resolution": resolution.to_dict()}),
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
            or self._new_op_id("DecisionAdded", timestamp, {"decision": decision.to_dict()}),
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
            or self._new_op_id("DerivationAdded", timestamp, {"derivation": derivation.to_dict()}),
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
        registry = predicate_registry or self.predicate_registry
        detectors = self.conflict_detectors if conflict_detectors is None else conflict_detectors
        if (
            predicate_registry is None
            and not detectors
            and valid_at is None
            and context is None
            and isinstance(self.store, IncrementalOpStore)
        ):
            return _snapshot_state(self._materialize_canonical_incrementally())
        return materialize(
            self.load_oplog(),
            predicate_registry=registry,
            conflict_detectors=detectors,
            valid_at=valid_at,
            context=context,
        )

    def materialize_with_delta(
        self,
        since_cursor: int | None,
    ) -> tuple[MemoryState, MaterializationDelta]:
        if self.conflict_detectors or not isinstance(self.store, IncrementalOpStore):
            state = self.materialize()
            return state, MaterializationDelta(full_rebuild=True)
        state = _snapshot_state(self._materialize_canonical_incrementally())
        current = self._materialization_cursor
        if since_cursor is None:
            return state, MaterializationDelta(
                cursor_after=current,
                full_rebuild=True,
            )
        if since_cursor == current:
            return state, MaterializationDelta(
                cursor_before=current,
                cursor_after=current,
            )
        deltas: list[MaterializationDelta] = []
        expected = since_cursor
        for delta in self._materialization_history:
            if delta.cursor_after <= since_cursor:
                continue
            if delta.cursor_before != expected:
                return state, MaterializationDelta(
                    cursor_before=since_cursor,
                    cursor_after=current,
                    full_rebuild=True,
                )
            deltas.append(delta)
            expected = delta.cursor_after
        if not deltas or expected != current:
            return state, MaterializationDelta(
                cursor_before=since_cursor,
                cursor_after=current,
                full_rebuild=True,
            )
        return state, self._combine_materialization_deltas(deltas)

    @property
    def materialization_cursor(self) -> int | None:
        if not isinstance(self.store, IncrementalOpStore) or self.conflict_detectors:
            return None
        self._materialize_canonical_incrementally()
        return self._materialization_cursor

    def find_conflicts(
        self,
        *,
        valid_at: str | None = None,
        applicability_context: dict[str, JSONValue] | None = None,
        **filters: Any,
    ) -> tuple[ConflictSet, ...]:
        return self.materialize(valid_at=valid_at, context=applicability_context).find_conflicts(
            **filters
        )

    def merge_from(self, other_store_or_oplog: OpStore | OpLog) -> OpLog:
        if isinstance(other_store_or_oplog, OpLog):
            other_oplog = other_store_or_oplog
        elif hasattr(other_store_or_oplog, "load_oplog"):
            other_oplog = other_store_or_oplog.load_oplog()  # type: ignore[assignment]
        else:
            raise TypeError("Expected OpLog or OpStore-compatible object.")
        merged = merge(self.load_oplog(), other_oplog)
        if isinstance(self.store, BatchOpStore):
            self.store.append_many(*merged.iter_ops())
        else:
            for op in merged.iter_ops():
                self.store.append(op)
        return merged

    def export_replica_manifest(
        self,
        after_cursor: int,
        *,
        max_ops: int | None = None,
    ) -> ReplicaManifest:
        """Describe a stream suffix without transferring operation payloads."""
        store_delta = self._read_replica_store_delta(after_cursor, max_ops=max_ops)
        return ReplicaManifest(
            sender_replica_id=self.replica_id,
            cursor_before=after_cursor,
            cursor_after=store_delta.cursor,
            entries=tuple(
                (op.op_id, digest_content(op.to_json())) for op in store_delta.ops
            ),
            batch_sizes=store_delta.batch_sizes,
        )

    def export_replica_delta(
        self,
        after_cursor: int,
        *,
        max_ops: int | None = None,
    ) -> ReplicaDelta:
        """Return the append-only portion of this replica stream after a cursor."""
        store_delta = self._read_replica_store_delta(after_cursor, max_ops=max_ops)
        return ReplicaDelta(
            sender_replica_id=self.replica_id,
            cursor_before=after_cursor,
            cursor_after=store_delta.cursor,
            ops=store_delta.ops,
            batch_sizes=store_delta.batch_sizes,
        )

    def _export_manifest_delta(
        self, manifest: ReplicaManifest, op_ids: tuple[str, ...]
    ) -> ReplicaDelta:
        if manifest.sender_replica_id != self.replica_id:
            raise ValueError("Replica manifest sender does not match this replica.")
        if not manifest.entries:
            return ReplicaDelta(
                self.replica_id, manifest.cursor_before, manifest.cursor_after
            )
        store_delta = self._read_replica_store_delta(
            manifest.cursor_before, max_ops=len(manifest.entries)
        )
        current_entries = tuple(
            (op.op_id, digest_content(op.to_json())) for op in store_delta.ops
        )
        if store_delta.cursor != manifest.cursor_after or current_entries != manifest.entries:
            raise ValueError("Replica stream changed while exporting a manifest payload.")
        wanted = set(op_ids)
        ops: list[AnyOp] = []
        batch_sizes: list[int] = []
        for batch in store_delta.batches:
            selected = tuple(op for op in batch if op.op_id in wanted)
            if selected:
                ops.extend(selected)
                batch_sizes.append(len(selected))
        return ReplicaDelta(
            sender_replica_id=self.replica_id,
            cursor_before=manifest.cursor_before,
            cursor_after=manifest.cursor_after,
            ops=tuple(ops),
            batch_sizes=tuple(batch_sizes),
        )

    def replica_progress(self, peer_replica_id: str) -> ReplicaProgress:
        """Load durable receive progress for one peer's append-only stream."""
        store = self._replica_progress_store()
        return store.load_replica_progress(peer_replica_id)

    def apply_replica_delta(self, delta: ReplicaDelta) -> ReplicaSyncReport:
        """Apply a peer delta idempotently and advance progress after durable writes."""
        if delta.sender_replica_id == self.replica_id:
            raise ValueError("Cannot apply a replica delta from the local replica_id.")
        progress_store = self._replica_progress_store()
        progress = progress_store.load_replica_progress(delta.sender_replica_id)

        added_items: list[bool] = []
        for batch in delta.batches:
            append_many = getattr(self.store, "append_many", None)
            if callable(append_many):
                added_items.extend(append_many(*batch))
            else:
                added_items.extend(self.store.append(op) for op in batch)
        added = tuple(added_items)

        applied_op_ids = tuple(
            op.op_id for op, was_added in zip(delta.ops, added, strict=True) if was_added
        )
        duplicate_op_ids = tuple(
            op.op_id for op, was_added in zip(delta.ops, added, strict=True) if not was_added
        )

        # Update an existing in-process materialization before recording receipt.
        # If this fails, progress stays behind and a retry safely deduplicates writes.
        if delta.ops:
            self.materialize()

        updated = advance_replica_progress(
            progress,
            cursor_before=delta.cursor_before,
            cursor_after=delta.cursor_after,
        )
        progress_store.save_replica_progress(updated)
        return ReplicaSyncReport(
            peer_replica_id=delta.sender_replica_id,
            cursor_before=progress.cursor,
            cursor_after=updated.cursor,
            transferred_op_ids=tuple(op.op_id for op in delta.ops),
            applied_op_ids=applied_op_ids,
            duplicate_op_ids=duplicate_op_ids,
            pending_ranges=updated.pending_ranges,
        )

    def sync_from(self, peer: Memory, *, max_ops: int | None = None) -> ReplicaSyncReport:
        """Pull only locally unknown operations from a peer's durable cursor."""
        if peer.replica_id == self.replica_id:
            raise ValueError("Replica sync requires distinct replica_id values.")
        progress = self.replica_progress(peer.replica_id)
        manifest = peer.export_replica_manifest(progress.cursor, max_ops=max_ops)
        if manifest.sender_replica_id != peer.replica_id:
            raise ValueError("Replica manifest sender does not match the requested peer.")

        op_ids = tuple(op_id for op_id, _digest in manifest.entries)
        find_existing = getattr(self.store, "find_ops", None)
        if callable(find_existing):
            existing = find_existing(op_ids)
        else:
            local_oplog = self.store.load_oplog()
            existing = {op_id: op for op_id in op_ids if (op := local_oplog.get(op_id)) is not None}
        unknown_op_ids: list[str] = []
        for op_id, expected_digest in manifest.entries:
            known_op = existing.get(op_id)
            if known_op is None:
                unknown_op_ids.append(op_id)
            elif digest_content(known_op.to_json()) != expected_digest:
                raise ValueError(f"op_id collision with different payload: {op_id}")
        return self.apply_replica_delta(
            peer._export_manifest_delta(manifest, tuple(unknown_op_ids))
        )

    def reset_replica_progress(self, peer_replica_id: str) -> None:
        """Forget one stream cursor after its peer log is compacted or replaced."""
        self._replica_progress_store().reset_replica_progress(peer_replica_id)

    def _read_replica_store_delta(
        self, after_cursor: int, *, max_ops: int | None
    ) -> StoreDelta:
        if not isinstance(after_cursor, int) or isinstance(after_cursor, bool) or after_cursor < 0:
            raise ValueError("Replica delta cursor must be a non-negative integer.")
        if max_ops is not None and (
            not isinstance(max_ops, int) or isinstance(max_ops, bool) or max_ops < 1
        ):
            raise ValueError("Replica delta max_ops must be a positive integer.")
        if not isinstance(self.store, IncrementalOpStore):
            raise TypeError("Replica delta export requires an IncrementalOpStore.")
        if max_ops is None:
            return self.store.read_after(after_cursor)
        try:
            return self.store.read_after(after_cursor, limit=max_ops)  # type: ignore[call-arg]
        except TypeError as error:
            raise TypeError(
                "Bounded replica delta export requires a store with limited reads."
            ) from error

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
            resolver=resolver,
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

    def _replica_progress_store(self) -> ReplicaProgressStore:
        if not isinstance(self.store, ReplicaProgressStore):
            raise TypeError("Replica delta sync requires a ReplicaProgressStore.")
        return self.store

    def _materialize_canonical_incrementally(self) -> MemoryState:
        assert isinstance(self.store, IncrementalOpStore)
        if self._automatic_materialization_config and (
            self.predicate_registry.revision != 0
            or (
                self._materialized_registry is not None
                and self.predicate_registry is not self._materialized_registry
            )
        ):
            self._materialization_config_token = None
        if self._materialized_state is not None and (
            self._materialized_registry is not self.predicate_registry
            or self._materialized_registry_revision != self.predicate_registry.revision
        ):
            self._rebuild_materialized_state()
        if self._materialized_state is None:
            self._restore_or_build_materialized_state()
        assert self._materialized_state is not None
        try:
            store_delta = self.store.read_after(self._materialization_cursor)
        except ValueError:
            self._rebuild_materialized_state()
            return self._materialized_state
        if not store_delta.ops:
            return self._materialized_state
        state, delta = apply_operations_incrementally(
            self._materialized_state,
            store_delta.ops,
        )
        delta = replace(
            delta,
            cursor_before=self._materialization_cursor,
            cursor_after=store_delta.cursor,
        )
        self._materialized_state = state
        self._remember_materialization_registry()
        self._materialization_cursor = store_delta.cursor
        self._materialization_history.append(delta)
        self._ops_since_checkpoint += len(store_delta.ops)
        if self._ops_since_checkpoint >= _CHECKPOINT_INTERVAL_OPS:
            self._save_materialization_checkpoint()
        return state

    def _restore_or_build_materialized_state(self) -> None:
        assert isinstance(self.store, IncrementalOpStore)
        checkpoint = self.store.load_materialization_checkpoint()
        if (
            checkpoint is not None
            and self._materialization_config_token is not None
            and checkpoint.version == MATERIALIZATION_CHECKPOINT_VERSION
            and checkpoint.config_token == self._materialization_config_token
        ):
            try:
                state = memory_state_from_checkpoint(
                    checkpoint.payload,
                    predicate_registry=self.predicate_registry,
                )
                store_delta = self.store.read_after(checkpoint.cursor)
                if store_delta.ops:
                    state, delta = apply_operations_incrementally(state, store_delta.ops)
                    self._materialization_history.append(
                        replace(
                            delta,
                            cursor_before=checkpoint.cursor,
                            cursor_after=store_delta.cursor,
                        )
                    )
                self._materialized_state = state
                self._remember_materialization_registry()
                self._materialization_cursor = store_delta.cursor
                self._ops_since_checkpoint = len(store_delta.ops)
                return
            except (KeyError, TypeError, ValueError):
                pass
        self._rebuild_materialized_state()

    def _rebuild_materialized_state(self) -> None:
        assert isinstance(self.store, IncrementalOpStore)
        store_delta = self.store.read_after(0)
        self._materialized_state = materialize(
            self.load_oplog(),
            predicate_registry=self.predicate_registry,
        )
        self._materialization_cursor = store_delta.cursor
        self._remember_materialization_registry()
        self._materialization_history.clear()
        self._save_materialization_checkpoint()

    def _save_materialization_checkpoint(self) -> None:
        if self._materialization_config_token is None or self._materialized_state is None:
            return
        assert isinstance(self.store, IncrementalOpStore)
        self.store.save_materialization_checkpoint(
            MaterializationCheckpoint(
                version=MATERIALIZATION_CHECKPOINT_VERSION,
                config_token=self._materialization_config_token,
                cursor=self._materialization_cursor,
                payload=memory_state_to_checkpoint(self._materialized_state),
            )
        )
        self._ops_since_checkpoint = 0

    def _remember_materialization_registry(self) -> None:
        self._materialized_registry = self.predicate_registry
        self._materialized_registry_revision = self.predicate_registry.revision

    @staticmethod
    def _combine_materialization_deltas(
        deltas: Sequence[MaterializationDelta],
    ) -> MaterializationDelta:
        return MaterializationDelta(
            processed_op_ids=tuple(op_id for delta in deltas for op_id in delta.processed_op_ids),
            affected_keys=tuple(sorted({key for delta in deltas for key in delta.affected_keys})),
            affected_namespaces=tuple(
                sorted({namespace for delta in deltas for namespace in delta.affected_namespaces})
            ),
            affected_claim_ids=tuple(
                sorted({claim_id for delta in deltas for claim_id in delta.affected_claim_ids})
            ),
            previous_conflict_ids=tuple(
                sorted(
                    {conflict_id for delta in deltas for conflict_id in delta.previous_conflict_ids}
                )
            ),
            current_conflict_ids=tuple(
                sorted(
                    {conflict_id for delta in deltas for conflict_id in delta.current_conflict_ids}
                )
            ),
            affected_conflict_refs=tuple(
                sorted(
                    {
                        conflict_ref
                        for delta in deltas
                        for conflict_ref in delta.affected_conflict_refs
                    }
                )
            ),
            affected_resolution_ids=tuple(
                sorted(
                    {
                        resolution_id
                        for delta in deltas
                        for resolution_id in delta.affected_resolution_ids
                    }
                )
            ),
            cursor_before=deltas[0].cursor_before,
            cursor_after=deltas[-1].cursor_after,
            conflict_comparisons=sum(delta.conflict_comparisons for delta in deltas),
        )


def _snapshot_state(state: MemoryState) -> MemoryState:
    updates: dict[str, Any] = {}
    for item in fields(state):
        value = getattr(state, item.name)
        if isinstance(value, dict):
            updates[item.name] = {
                key: list(nested) if isinstance(nested, list) else nested
                for key, nested in value.items()
            }
        elif isinstance(value, list):
            updates[item.name] = list(value)
        elif isinstance(value, set):
            updates[item.name] = set(value)
    return replace(state, **updates)
