from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace

from ..conflict import ConflictSet
from ..materialize import MaterializationDelta, MemoryState
from ..memory import Memory
from ..model import Claim, ResolutionRecord
from ..utils import canonical_json_dumps, digest_json_value, utc_now_iso
from .base import (
    AsyncMemoryRepositoryAdapter,
    ExternalReferenceStore,
    LocalProjectionDeltaStore,
    MemoryRepositoryAdapter,
)
from .errors import AdapterProtocolError
from .models import (
    ExternalReference,
    HydratedContext,
    PendingProjectionDelta,
    RetrievalRecord,
    SearchHit,
    SearchRequest,
    SyncFailure,
    SyncReport,
)
from .registry import InMemoryLocalProjectionDeltaStore


@dataclass(frozen=True)
class ProjectionDelta:
    upserts: tuple[RetrievalRecord, ...] = ()
    deletes: tuple[str, ...] = ()


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
        _claim_record(claim, conflicts_by_claim.get(claim.claim_id, ())) for claim in active_claims
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


def project_state_delta(
    state: MemoryState,
    namespace: str,
    delta: MaterializationDelta,
) -> ProjectionDelta:
    if delta.full_rebuild:
        return ProjectionDelta(upserts=project_state(state, namespace))

    upserts: dict[str, RetrievalRecord] = {}
    deletes: set[str] = set()
    for claim_id in delta.affected_claim_ids:
        claim = state.claims_by_id.get(claim_id)
        if claim is None or claim.key.namespace != namespace:
            continue
        projection_id = f"statefuse:claim:{claim_id}"
        if claim_id in state.inactive_claim_ids or claim_id in state.inapplicable_claim_ids:
            deletes.add(projection_id)
            continue
        conflict_ids = tuple(
            sorted(
                conflict.conflict_id
                for conflict in state.conflicts_by_key.get(claim.key, ())
                if any(candidate.claim_id == claim_id for candidate in conflict.candidates)
            )
        )
        upserts[projection_id] = _claim_record(claim, conflict_ids)

    for conflict_id in sorted(set(delta.previous_conflict_ids) | set(delta.current_conflict_ids)):
        projection_id = f"statefuse:conflict:{conflict_id}"
        conflict = state.conflicts_by_id.get(conflict_id)
        if conflict is None:
            deletes.add(projection_id)
            continue
        if any(key.namespace == namespace for key in conflict.keys):
            upserts[projection_id] = _conflict_record(state, conflict, namespace)

    for conflict_ref in delta.affected_conflict_refs:
        conflict = state.conflicts_by_ref.get(conflict_ref)
        if conflict is None or not any(key.namespace == namespace for key in conflict.keys):
            continue
        projection_id = f"statefuse:conflict:{conflict.conflict_id}"
        upserts[projection_id] = _conflict_record(state, conflict, namespace)

    for resolution_id in delta.affected_resolution_ids:
        resolution = state.resolutions_by_id.get(resolution_id)
        if resolution is None:
            continue
        record = _resolution_record(state, resolution, namespace)
        if record is not None:
            upserts[record.projection_id] = record

    deletes.difference_update(upserts)
    return ProjectionDelta(
        upserts=tuple(upserts[key] for key in sorted(upserts)),
        deletes=tuple(sorted(deletes)),
    )


def hydrate_search_hits(
    memory: Memory,
    hits: Iterable[SearchHit],
    *,
    max_related_conflicts: int = 32,
    max_claims: int = 128,
) -> HydratedContext:
    if max_related_conflicts < 1 or max_claims < 1:
        raise ValueError("Hydration bounds must be positive.")
    state = memory.materialize()
    unique_hits = tuple(
        sorted(
            {(hit.external_id, hit.projection_id): hit for hit in hits}.values(),
            key=lambda hit: (hit.projection_id or "", hit.external_id),
        )
    )
    seed_claim_ids = {
        claim_id
        for hit in unique_hits
        for claim_id in hit.claim_ids
        if isinstance(claim_id, str) and claim_id
    }
    hinted_conflict_ids = {
        conflict_id
        for hit in unique_hits
        for conflict_id in hit.conflict_ids
        if isinstance(conflict_id, str) and conflict_id
    }
    current_conflict_ids = {
        conflict_id for conflict_id in hinted_conflict_ids if conflict_id in state.conflicts_by_id
    }
    for claim_id in sorted(seed_claim_ids):
        claim = state.claims_by_id.get(claim_id)
        if claim is None:
            continue
        current_conflict_ids.update(
            conflict.conflict_id
            for conflict in state.conflicts_by_key.get(claim.key, ())
            if any(candidate.claim_id == claim_id for candidate in conflict.candidates)
        )

    ordered_conflict_ids = sorted(current_conflict_ids)
    selected_conflict_ids = ordered_conflict_ids[:max_related_conflicts]
    omitted_conflict_ids = ordered_conflict_ids[max_related_conflicts:]
    all_claim_ids = set(seed_claim_ids)
    for conflict_id in selected_conflict_ids:
        all_claim_ids.update(
            claim.claim_id for claim in state.conflicts_by_id[conflict_id].candidates
        )
    ordered_claim_ids = sorted(all_claim_ids)
    claim_ids = ordered_claim_ids[:max_claims]
    omitted_claim_ids = ordered_claim_ids[max_claims:]
    claims = tuple(
        state.claims_by_id[claim_id] for claim_id in claim_ids if claim_id in state.claims_by_id
    )
    conflicts = tuple(
        state.conflicts_by_id[conflict_id]
        for conflict_id in selected_conflict_ids
        if conflict_id in state.conflicts_by_id
    )
    resolutions: dict[str, ResolutionRecord] = {}
    resolution_statuses: dict[str, str] = {}
    for conflict in conflicts:
        lane = (conflict.conflict_ref, None)
        resolution = state.effective_resolutions_by_conflict_ref_and_scope.get(lane)
        if resolution is None:
            continue
        resolutions[resolution.resolution_id] = resolution
        resolution_statuses[resolution.resolution_id] = (
            state.lifecycle_status_by_conflict_ref_and_scope.get(lane, "open")
        )
    return HydratedContext(
        claims=claims,
        conflicts=conflicts,
        missing_claim_ids=tuple(
            claim_id for claim_id in sorted(seed_claim_ids) if claim_id not in state.claims_by_id
        ),
        missing_conflict_ids=tuple(
            conflict_id
            for conflict_id in sorted(hinted_conflict_ids)
            if conflict_id not in state.conflicts_by_id
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
        resolutions=tuple(resolutions[key] for key in sorted(resolutions)),
        resolution_statuses=resolution_statuses,
        omitted_claim_ids=tuple(omitted_claim_ids),
        omitted_conflict_ids=tuple(omitted_conflict_ids),
    )


def _refresh_pending_deltas(
    *,
    memory: Memory,
    cursor: int | None,
    repository: str,
    namespace: str,
    reference_store: ExternalReferenceStore,
    delta_store: LocalProjectionDeltaStore,
) -> int:
    state, materialization_delta = memory.materialize_with_delta(cursor)
    if materialization_delta.is_empty:
        return materialization_delta.cursor_after
    known = {
        reference.projection_id: reference
        for reference in reference_store.list(repository, namespace)
    }
    pending_ids = {
        delta.projection_id for delta in delta_store.list(repository, namespace)
    }
    if materialization_delta.full_rebuild:
        desired = {record.projection_id: record for record in project_state(state, namespace)}
        upserts = tuple(desired.values())
        deletes = tuple(sorted((set(known) | pending_ids) - set(desired)))
    else:
        projected = project_state_delta(state, namespace, materialization_delta)
        upserts = projected.upserts
        deletes = projected.deletes

    for record in upserts:
        reference = known.get(record.projection_id)
        if reference is not None and reference.metadata.get("fingerprint") == _fingerprint(record):
            delta_store.delete(repository, namespace, record.projection_id)
            continue
        delta_store.upsert(
            PendingProjectionDelta(
                repository=repository,
                namespace=namespace,
                projection_id=record.projection_id,
                operation="upsert",
                cursor=materialization_delta.cursor_after,
                record=record,
            )
        )
    for projection_id in deletes:
        delta_store.upsert(
            PendingProjectionDelta(
                repository=repository,
                namespace=namespace,
                projection_id=projection_id,
                operation="delete",
                cursor=materialization_delta.cursor_after,
            )
        )
    return materialization_delta.cursor_after


def _merge_search_hits(
    *,
    external_hits: Iterable[SearchHit],
    pending: tuple[PendingProjectionDelta, ...],
    request: SearchRequest,
) -> tuple[SearchHit, ...]:
    deleted_projection_ids = {
        delta.projection_id for delta in pending if delta.operation == "delete"
    }
    pending_upserts = {
        delta.projection_id: delta
        for delta in pending
        if delta.operation == "upsert" and delta.record is not None
    }
    pending_claim_ids = {
        claim_id
        for delta in pending
        if delta.record is not None
        for claim_id in delta.record.claim_ids
    }
    deleted_claim_ids = {
        projection_id.removeprefix("statefuse:claim:")
        for projection_id in deleted_projection_ids
        if projection_id.startswith("statefuse:claim:")
    }
    merged: dict[tuple[str, str], SearchHit] = {}
    for hit in external_hits:
        if (
            hit.projection_id in deleted_projection_ids
            or deleted_claim_ids.intersection(hit.claim_ids)
            or (hit.projection_id is None and pending_claim_ids.intersection(hit.claim_ids))
        ):
            continue
        key = (
            ("projection", hit.projection_id)
            if hit.projection_id
            else ("external", hit.external_id)
        )
        current = pending_upserts.get(hit.projection_id or "")
        if current is None:
            merged[key] = hit
            continue
        record = current.record
        assert record is not None
        if record.namespace != request.namespace or any(
            record.metadata.get(name) != value for name, value in request.filters.items()
        ):
            continue
        merged[key] = SearchHit(
            external_id=hit.external_id,
            projection_id=record.projection_id,
            text=record.text,
            score=hit.score,
            claim_ids=record.claim_ids,
            conflict_ids=record.conflict_ids,
            metadata={**record.metadata, "freshness_source": "local_pending"},
        )

    for delta in pending:
        if delta.operation != "upsert" or delta.record is None:
            continue
        hit = _local_search_hit(delta.record, request)
        if hit is None:
            continue
        merged.setdefault(("projection", delta.projection_id), hit)

    return tuple(
        sorted(
            merged.values(),
            key=lambda hit: (
                -(hit.score if hit.score is not None else float("-inf")),
                hit.projection_id or "",
                hit.external_id,
            ),
        )[: request.limit]
    )


def _local_search_hit(record: RetrievalRecord, request: SearchRequest) -> SearchHit | None:
    if record.namespace != request.namespace:
        return None
    if any(record.metadata.get(key) != value for key, value in request.filters.items()):
        return None
    query_tokens = set(re.findall(r"\w+", request.query.casefold()))
    record_tokens = set(re.findall(r"\w+", record.text.casefold()))
    overlap = len(query_tokens & record_tokens)
    if query_tokens and overlap == 0:
        return None
    score = overlap / len(query_tokens) if query_tokens else 1.0
    return SearchHit(
        external_id=f"local:{record.projection_id}",
        projection_id=record.projection_id,
        text=record.text,
        score=score,
        claim_ids=record.claim_ids,
        conflict_ids=record.conflict_ids,
        metadata={**record.metadata, "freshness_source": "local_pending"},
    )


class ProjectionService:
    def __init__(
        self,
        memory: Memory,
        adapter: MemoryRepositoryAdapter,
        reference_store: ExternalReferenceStore,
        delta_store: LocalProjectionDeltaStore | None = None,
    ) -> None:
        self.memory = memory
        self.adapter = adapter
        self.reference_store = reference_store
        self.delta_store = (
            delta_store if delta_store is not None else InMemoryLocalProjectionDeltaStore()
        )
        self._materialization_cursors: dict[str, int] = {}

    def synchronize(self, namespace: str) -> SyncReport:
        self._materialization_cursors[namespace] = _refresh_pending_deltas(
            memory=self.memory,
            cursor=self._materialization_cursors.get(namespace),
            repository=self.adapter.name,
            namespace=namespace,
            reference_store=self.reference_store,
            delta_store=self.delta_store,
        )
        known = {
            reference.projection_id: reference
            for reference in self.reference_store.list(self.adapter.name, namespace)
        }
        pending = self.delta_store.list(self.adapter.name, namespace)
        if not pending:
            return SyncReport(unchanged=tuple(sorted(known)))
        created: list[str] = []
        updated: list[str] = []
        deleted: list[str] = []
        unchanged: list[str] = []
        failed: list[SyncFailure] = []

        for delta in pending:
            projection_id = delta.projection_id
            if delta.operation == "delete":
                try:
                    deleted_remotely = self.adapter.delete(projection_id, namespace)
                    if not isinstance(deleted_remotely, bool):
                        raise AdapterProtocolError(
                            f"Adapter returned a non-boolean delete result for {projection_id}."
                        )
                    self.reference_store.delete(self.adapter.name, namespace, projection_id)
                    self.delta_store.delete(self.adapter.name, namespace, projection_id)
                    deleted.append(projection_id)
                except Exception as error:
                    failed.append(_failure(projection_id, "delete", error))
                continue

            record = delta.record
            assert record is not None
            reference = known.get(projection_id)
            fingerprint = _fingerprint(record)
            if reference is not None and reference.metadata.get("fingerprint") == fingerprint:
                self.delta_store.delete(self.adapter.name, namespace, projection_id)
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
                self.delta_store.delete(self.adapter.name, namespace, projection_id)
            except Exception as error:
                if reference is not None:
                    # A connector may implement update as replace. Dropping the disposable
                    # reference forces the next sync to verify/rebuild after a partial failure.
                    self.reference_store.delete(self.adapter.name, namespace, projection_id)
                failed.append(_failure(projection_id, "upsert", error))
        return SyncReport(
            created=tuple(created),
            updated=tuple(updated),
            deleted=tuple(deleted),
            unchanged=tuple(unchanged),
            failed=tuple(failed),
        )

    def search(self, request: SearchRequest) -> HydratedContext:
        self._materialization_cursors[request.namespace] = _refresh_pending_deltas(
            memory=self.memory,
            cursor=self._materialization_cursors.get(request.namespace),
            repository=self.adapter.name,
            namespace=request.namespace,
            reference_store=self.reference_store,
            delta_store=self.delta_store,
        )
        failure: SyncFailure | None = None
        try:
            external_hits = self.adapter.search(request)
        except Exception as error:
            external_hits = ()
            failure = _failure(self.adapter.name, "search", error)
        hits = _merge_search_hits(
            external_hits=external_hits,
            pending=self.delta_store.list(self.adapter.name, request.namespace),
            request=request,
        )
        context = hydrate_search_hits(self.memory, hits)
        return replace(context, search_failures=(failure,) if failure is not None else ())


class AsyncProjectionService:
    """Async synchronization service for native async repositories such as Graphiti."""

    def __init__(
        self,
        memory: Memory,
        adapter: AsyncMemoryRepositoryAdapter,
        reference_store: ExternalReferenceStore,
        delta_store: LocalProjectionDeltaStore | None = None,
    ) -> None:
        self.memory = memory
        self.adapter = adapter
        self.reference_store = reference_store
        self.delta_store = (
            delta_store if delta_store is not None else InMemoryLocalProjectionDeltaStore()
        )
        self._materialization_cursors: dict[str, int] = {}

    async def synchronize(self, namespace: str) -> SyncReport:
        self._materialization_cursors[namespace] = _refresh_pending_deltas(
            memory=self.memory,
            cursor=self._materialization_cursors.get(namespace),
            repository=self.adapter.name,
            namespace=namespace,
            reference_store=self.reference_store,
            delta_store=self.delta_store,
        )
        known = {
            reference.projection_id: reference
            for reference in self.reference_store.list(self.adapter.name, namespace)
        }
        pending = self.delta_store.list(self.adapter.name, namespace)
        if not pending:
            return SyncReport(unchanged=tuple(sorted(known)))
        created: list[str] = []
        updated: list[str] = []
        deleted: list[str] = []
        unchanged: list[str] = []
        failed: list[SyncFailure] = []

        for delta in pending:
            projection_id = delta.projection_id
            if delta.operation == "delete":
                try:
                    deleted_remotely = await self.adapter.adelete(projection_id, namespace)
                    if not isinstance(deleted_remotely, bool):
                        raise AdapterProtocolError(
                            f"Adapter returned a non-boolean delete result for {projection_id}."
                        )
                    self.reference_store.delete(self.adapter.name, namespace, projection_id)
                    self.delta_store.delete(self.adapter.name, namespace, projection_id)
                    deleted.append(projection_id)
                except Exception as error:
                    failed.append(_failure(projection_id, "delete", error))
                continue

            record = delta.record
            assert record is not None
            reference = known.get(projection_id)
            fingerprint = _fingerprint(record)
            if reference is not None and reference.metadata.get("fingerprint") == fingerprint:
                self.delta_store.delete(self.adapter.name, namespace, projection_id)
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
                self.delta_store.delete(self.adapter.name, namespace, projection_id)
            except Exception as error:
                if reference is not None:
                    self.reference_store.delete(self.adapter.name, namespace, projection_id)
                failed.append(_failure(projection_id, "upsert", error))
        return SyncReport(
            created=tuple(created),
            updated=tuple(updated),
            deleted=tuple(deleted),
            unchanged=tuple(unchanged),
            failed=tuple(failed),
        )

    async def search(self, request: SearchRequest) -> HydratedContext:
        self._materialization_cursors[request.namespace] = _refresh_pending_deltas(
            memory=self.memory,
            cursor=self._materialization_cursors.get(request.namespace),
            repository=self.adapter.name,
            namespace=request.namespace,
            reference_store=self.reference_store,
            delta_store=self.delta_store,
        )
        failure: SyncFailure | None = None
        try:
            external_hits = await self.adapter.asearch(request)
        except Exception as error:
            external_hits = ()
            failure = _failure(self.adapter.name, "search", error)
        hits = _merge_search_hits(
            external_hits=external_hits,
            pending=self.delta_store.list(self.adapter.name, request.namespace),
            request=request,
        )
        context = hydrate_search_hits(self.memory, hits)
        return replace(context, search_failures=(failure,) if failure is not None else ())


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


def _conflict_record(state: MemoryState, conflict: ConflictSet, namespace: str) -> RetrievalRecord:
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
        state.claims_by_id.get(claim_id) and state.claims_by_id[claim_id].key.namespace == namespace
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
