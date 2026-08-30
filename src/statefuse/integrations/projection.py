from __future__ import annotations

from collections.abc import Iterable

from ..conflict import ConflictSet
from ..materialize import MemoryState
from ..memory import Memory
from ..model import Claim, ResolutionRecord
from ..utils import canonical_json_dumps, digest_json_value, utc_now_iso
from .base import AsyncMemoryRepositoryAdapter, ExternalReferenceStore, MemoryRepositoryAdapter
from .errors import AdapterProtocolError
from .models import (
    ExternalReference,
    HydratedContext,
    RetrievalRecord,
    SearchHit,
    SearchRequest,
    SyncFailure,
    SyncReport,
)


def project_state(state: MemoryState, namespace: str) -> tuple[RetrievalRecord, ...]:
    conflicts_by_claim: dict[str, tuple[str, ...]] = {}
    for conflict in state.conflicts:
        for claim in conflict.candidates:
            conflicts_by_claim[claim.claim_id] = tuple(
                sorted((*conflicts_by_claim.get(claim.claim_id, ()), conflict.conflict_id))
            )

    records: list[RetrievalRecord] = []
    active_claims = sorted(
        (
            claim
            for claims in state.active_claims_by_key.values()
            for claim in claims
            if claim.key.namespace == namespace
        ),
        key=lambda claim: claim.claim_id,
    )
    records.extend(
        _claim_record(claim, conflicts_by_claim.get(claim.claim_id, ()))
        for claim in active_claims
    )
    records.extend(
        _conflict_record(state, conflict, namespace)
        for conflict in state.conflicts
        if any(key.namespace == namespace for key in conflict.keys)
    )
    records.extend(
        record
        for resolution in state.resolutions_by_id.values()
        if (record := _resolution_record(state, resolution, namespace)) is not None
    )
    return tuple(sorted(records, key=lambda record: record.projection_id))


def hydrate_search_hits(memory: Memory, hits: Iterable[SearchHit]) -> HydratedContext:
    state = memory.materialize()
    unique_hits = tuple(
        {
            (hit.external_id, hit.projection_id): hit
            for hit in hits
        }.values()
    )
    claim_ids = sorted(
        {
            claim_id
            for hit in unique_hits
            for claim_id in hit.claim_ids
            if isinstance(claim_id, str) and claim_id
        }
    )
    conflict_ids = sorted(
        {
            conflict_id
            for hit in unique_hits
            for conflict_id in hit.conflict_ids
            if isinstance(conflict_id, str) and conflict_id
        }
    )
    claims = tuple(
        state.claims_by_id[claim_id]
        for claim_id in claim_ids
        if claim_id in state.claims_by_id
    )
    conflicts = tuple(
        state.conflicts_by_id[conflict_id]
        for conflict_id in conflict_ids
        if conflict_id in state.conflicts_by_id
    )
    return HydratedContext(
        claims=claims,
        conflicts=conflicts,
        missing_claim_ids=tuple(
            claim_id for claim_id in claim_ids if claim_id not in state.claims_by_id
        ),
        missing_conflict_ids=tuple(
            conflict_id for conflict_id in conflict_ids if conflict_id not in state.conflicts_by_id
        ),
        search_hits=unique_hits,
        claim_statuses={
            claim.claim_id: "inactive" if claim.claim_id in state.inactive_claim_ids else "active"
            for claim in claims
        },
        conflict_statuses={
            conflict.conflict_id: state.lifecycle_status_by_conflict_ref_and_scope.get(
                (conflict.conflict_ref, None), "open"
            )
            for conflict in conflicts
        },
    )


class ProjectionService:
    def __init__(
        self,
        memory: Memory,
        adapter: MemoryRepositoryAdapter,
        reference_store: ExternalReferenceStore,
    ) -> None:
        self.memory = memory
        self.adapter = adapter
        self.reference_store = reference_store

    def synchronize(self, namespace: str) -> SyncReport:
        desired = {
            record.projection_id: record
            for record in project_state(self.memory.materialize(), namespace)
        }
        known = {
            reference.projection_id: reference
            for reference in self.reference_store.list(self.adapter.name, namespace)
        }
        created: list[str] = []
        updated: list[str] = []
        deleted: list[str] = []
        unchanged: list[str] = []
        failed: list[SyncFailure] = []

        for projection_id, record in desired.items():
            reference = known.get(projection_id)
            fingerprint = _fingerprint(record)
            if reference is not None and reference.metadata.get("fingerprint") == fingerprint:
                unchanged.append(projection_id)
                continue
            try:
                result = self.adapter.upsert(record)
                if (
                    result.repository != self.adapter.name
                    or result.projection_id != projection_id
                    or not result.external_id
                ):
                    raise AdapterProtocolError(
                        f"Adapter returned an invalid write result for {projection_id}."
                    )
                now = utc_now_iso()
                self.reference_store.upsert(
                    ExternalReference(
                        repository=self.adapter.name,
                        projection_id=projection_id,
                        external_id=result.external_id,
                        namespace=namespace,
                        created_at=reference.created_at if reference else now,
                        updated_at=now,
                        metadata={
                            **result.metadata,
                            "fingerprint": fingerprint,
                            "projection_version": record.projection_version,
                        },
                    )
                )
                (updated if reference else created).append(projection_id)
            except Exception as error:
                if reference is not None:
                    # A connector may implement update as replace. Dropping the disposable
                    # reference forces the next sync to verify/rebuild after a partial failure.
                    self.reference_store.delete(self.adapter.name, namespace, projection_id)
                failed.append(_failure(projection_id, "upsert", error))

        for projection_id in sorted(set(known) - set(desired)):
            try:
                deleted_remotely = self.adapter.delete(projection_id, namespace)
                if not isinstance(deleted_remotely, bool):
                    raise AdapterProtocolError(
                        f"Adapter returned a non-boolean delete result for {projection_id}."
                    )
                self.reference_store.delete(self.adapter.name, namespace, projection_id)
                deleted.append(projection_id)
            except Exception as error:
                failed.append(_failure(projection_id, "delete", error))

        return SyncReport(
            created=tuple(created),
            updated=tuple(updated),
            deleted=tuple(deleted),
            unchanged=tuple(unchanged),
            failed=tuple(failed),
        )

    def search(self, request: SearchRequest) -> HydratedContext:
        return hydrate_search_hits(self.memory, self.adapter.search(request))


class AsyncProjectionService:
    """Async synchronization service for native async repositories such as Graphiti."""

    def __init__(
        self,
        memory: Memory,
        adapter: AsyncMemoryRepositoryAdapter,
        reference_store: ExternalReferenceStore,
    ) -> None:
        self.memory = memory
        self.adapter = adapter
        self.reference_store = reference_store

    async def synchronize(self, namespace: str) -> SyncReport:
        desired = {
            record.projection_id: record
            for record in project_state(self.memory.materialize(), namespace)
        }
        known = {
            reference.projection_id: reference
            for reference in self.reference_store.list(self.adapter.name, namespace)
        }
        created: list[str] = []
        updated: list[str] = []
        deleted: list[str] = []
        unchanged: list[str] = []
        failed: list[SyncFailure] = []

        for projection_id, record in desired.items():
            reference = known.get(projection_id)
            fingerprint = _fingerprint(record)
            if reference is not None and reference.metadata.get("fingerprint") == fingerprint:
                unchanged.append(projection_id)
                continue
            try:
                result = await self.adapter.aupsert(record)
                if (
                    result.repository != self.adapter.name
                    or result.projection_id != projection_id
                    or not result.external_id
                ):
                    raise AdapterProtocolError(
                        f"Adapter returned an invalid write result for {projection_id}."
                    )
                now = utc_now_iso()
                self.reference_store.upsert(
                    ExternalReference(
                        repository=self.adapter.name,
                        projection_id=projection_id,
                        external_id=result.external_id,
                        namespace=namespace,
                        created_at=reference.created_at if reference else now,
                        updated_at=now,
                        metadata={
                            **result.metadata,
                            "fingerprint": fingerprint,
                            "projection_version": record.projection_version,
                        },
                    )
                )
                (updated if reference else created).append(projection_id)
            except Exception as error:
                if reference is not None:
                    self.reference_store.delete(self.adapter.name, namespace, projection_id)
                failed.append(_failure(projection_id, "upsert", error))

        for projection_id in sorted(set(known) - set(desired)):
            try:
                deleted_remotely = await self.adapter.adelete(projection_id, namespace)
                if not isinstance(deleted_remotely, bool):
                    raise AdapterProtocolError(
                        f"Adapter returned a non-boolean delete result for {projection_id}."
                    )
                self.reference_store.delete(self.adapter.name, namespace, projection_id)
                deleted.append(projection_id)
            except Exception as error:
                failed.append(_failure(projection_id, "delete", error))

        return SyncReport(
            created=tuple(created),
            updated=tuple(updated),
            deleted=tuple(deleted),
            unchanged=tuple(unchanged),
            failed=tuple(failed),
        )

    async def search(self, request: SearchRequest) -> HydratedContext:
        return hydrate_search_hits(self.memory, await self.adapter.asearch(request))


def _claim_record(claim: Claim, conflict_ids: tuple[str, ...]) -> RetrievalRecord:
    evidence = ", ".join(claim.evidence_ids) or "none"
    return RetrievalRecord(
        projection_id=f"statefuse:claim:{claim.claim_id}",
        text=(
            f"{claim.key.subject} {claim.key.predicate} is {_display(claim.value)}.\n"
            f"StateFuse claim ID: {claim.claim_id}.\n"
            f"Source evidence: {evidence}."
        ),
        namespace=claim.key.namespace,
        claim_ids=(claim.claim_id,),
        conflict_ids=conflict_ids,
        metadata={
            "kind": "claim",
            "claim_kind": claim.kind,
            "subject": claim.key.subject,
            "predicate": claim.key.predicate,
            "confidence": claim.confidence,
            "timestamp": claim.timestamp,
            "context": dict(claim.context),
            "validity": claim.validity.to_dict() if claim.validity else None,
        },
    )


def _conflict_record(
    state: MemoryState, conflict: ConflictSet, namespace: str
) -> RetrievalRecord:
    candidates = "\n".join(
        f"- {claim.key.subject}.{claim.key.predicate}={_display(claim.value)} "
        f"(claim {claim.claim_id})"
        for claim in conflict.candidates
    )
    status = state.lifecycle_status_by_conflict_ref_and_scope.get(
        (conflict.conflict_ref, None), "open"
    )
    return RetrievalRecord(
        projection_id=f"statefuse:conflict:{conflict.conflict_id}",
        text=(
            f"{conflict.conflict_class}/{conflict.conflict_subclass} conflict:\n"
            f"{candidates}\nReason: {conflict.reason}\nConflict ID: {conflict.conflict_id}\n"
            f"Status: {'resolved' if status == 'resolved' else 'unresolved'}"
        ),
        namespace=namespace,
        claim_ids=tuple(claim.claim_id for claim in conflict.candidates),
        conflict_ids=(conflict.conflict_id,),
        metadata={
            "kind": "conflict",
            "conflict_ref": conflict.conflict_ref,
            "conflict_type": conflict.conflict_type,
            "conflict_class": conflict.conflict_class,
            "conflict_subclass": conflict.conflict_subclass,
            "detector_id": conflict.detector_id,
            "keys": [key.to_dict() for key in conflict.keys],
            "annotations": dict(conflict.annotations),
            "witness": dict(conflict.witness),
            "status": status,
        },
    )


def _resolution_record(
    state: MemoryState, resolution: ResolutionRecord, namespace: str
) -> RetrievalRecord | None:
    claim_ids = tuple(
        sorted(
            set(resolution.selected_claim_ids)
            | set(resolution.rejected_claim_ids)
            | set(resolution.retained_claim_ids)
        )
    )
    if not any(
        state.claims_by_id.get(claim_id)
        and state.claims_by_id[claim_id].key.namespace == namespace
        for claim_id in claim_ids
    ):
        return None
    selected = ", ".join(resolution.selected_claim_ids) or "no claim"
    status = state.lifecycle_status_by_conflict_ref_and_scope.get(
        (resolution.conflict_ref, resolution.scope), "resolved"
    )
    return RetrievalRecord(
        projection_id=f"statefuse:resolution:{resolution.resolution_id}",
        text=(
            f"Conflict {resolution.observed_conflict_id} resolution outcome: "
            f"{resolution.outcome}; selected: {selected}.\n"
            f"Reason: {resolution.reason}\n"
            "The rejected claim remains available in StateFuse history."
        ),
        namespace=namespace,
        claim_ids=claim_ids,
        conflict_ids=(resolution.observed_conflict_id,),
        metadata={
            "kind": "resolution",
            "conflict_ref": resolution.conflict_ref,
            "status": status,
            "outcome": resolution.outcome,
            "timestamp": resolution.timestamp,
        },
    )


def _display(value: object) -> str:
    return value if isinstance(value, str) else canonical_json_dumps(value)


def _fingerprint(record: RetrievalRecord) -> str:
    return digest_json_value(
        {
            "text": record.text,
            "namespace": record.namespace,
            "claim_ids": list(record.claim_ids),
            "conflict_ids": list(record.conflict_ids),
            "metadata": record.metadata,
            "projection_version": record.projection_version,
        }
    )


def _failure(projection_id: str, operation: str, error: Exception) -> SyncFailure:
    return SyncFailure(
        projection_id=projection_id,
        operation=operation,
        error_type=type(error).__name__,
        message=str(error),
    )
