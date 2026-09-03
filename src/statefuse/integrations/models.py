from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..conflict import ConflictSet
from ..model import Claim, Derivation, Evidence, JSONValue, ResolutionRecord, Source


@dataclass(frozen=True)
class RetrievalRecord:
    projection_id: str
    text: str
    namespace: str
    claim_ids: tuple[str, ...] = ()
    conflict_ids: tuple[str, ...] = ()
    metadata: dict[str, JSONValue] = field(default_factory=dict)
    projection_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_ids", tuple(self.claim_ids))
        object.__setattr__(self, "conflict_ids", tuple(self.conflict_ids))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ExternalReference:
    repository: str
    projection_id: str
    external_id: str
    namespace: str
    created_at: str
    updated_at: str
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class SearchRequest:
    query: str
    namespace: str
    limit: int = 10
    filters: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("SearchRequest.limit must be positive.")
        object.__setattr__(self, "filters", dict(self.filters))


@dataclass(frozen=True)
class SearchHit:
    external_id: str
    projection_id: str | None
    text: str
    score: float | None
    claim_ids: tuple[str, ...]
    conflict_ids: tuple[str, ...]
    metadata: dict[str, JSONValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_ids", tuple(self.claim_ids))
        object.__setattr__(self, "conflict_ids", tuple(self.conflict_ids))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ExternalWriteResult:
    repository: str
    projection_id: str
    external_id: str
    created: bool
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class SyncFailure:
    projection_id: str
    operation: str
    error_type: str
    message: str


@dataclass(frozen=True)
class SyncReport:
    created: tuple[str, ...] = ()
    updated: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()
    unchanged: tuple[str, ...] = ()
    failed: tuple[SyncFailure, ...] = ()


@dataclass(frozen=True)
class PendingProjectionDelta:
    repository: str
    namespace: str
    projection_id: str
    operation: Literal["upsert", "delete"]
    cursor: int
    record: RetrievalRecord | None = None

    def __post_init__(self) -> None:
        if self.operation == "upsert" and self.record is None:
            raise ValueError("A pending upsert requires a retrieval record.")
        if self.operation == "delete" and self.record is not None:
            raise ValueError("A pending delete cannot contain a retrieval record.")


@dataclass(frozen=True)
class HydratedContext:
    claims: tuple[Claim, ...]
    conflicts: tuple[ConflictSet, ...]
    missing_claim_ids: tuple[str, ...]
    missing_conflict_ids: tuple[str, ...]
    search_hits: tuple[SearchHit, ...]
    claim_statuses: dict[str, str] = field(default_factory=dict)
    conflict_statuses: dict[str, str] = field(default_factory=dict)
    resolutions: tuple[ResolutionRecord, ...] = ()
    resolution_history: tuple[ResolutionRecord, ...] = ()
    stale_resolutions: tuple[ResolutionRecord, ...] = ()
    resolution_statuses: dict[str, str] = field(default_factory=dict)
    evidence: tuple[Evidence, ...] = ()
    sources: tuple[Source, ...] = ()
    derivations: tuple[Derivation, ...] = ()
    missing_evidence_ids: tuple[str, ...] = ()
    missing_source_ids: tuple[str, ...] = ()
    missing_derivation_ids: tuple[str, ...] = ()
    omitted_claim_ids: tuple[str, ...] = ()
    omitted_conflict_ids: tuple[str, ...] = ()
    omitted_evidence_ids: tuple[str, ...] = ()
    omitted_source_ids: tuple[str, ...] = ()
    omitted_derivation_ids: tuple[str, ...] = ()
    omitted_resolution_ids: tuple[str, ...] = ()
    search_failures: tuple[SyncFailure, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_statuses", dict(self.claim_statuses))
        object.__setattr__(self, "conflict_statuses", dict(self.conflict_statuses))
        object.__setattr__(self, "resolutions", tuple(self.resolutions))
        object.__setattr__(self, "resolution_history", tuple(self.resolution_history))
        object.__setattr__(self, "stale_resolutions", tuple(self.stale_resolutions))
        object.__setattr__(self, "resolution_statuses", dict(self.resolution_statuses))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "derivations", tuple(self.derivations))
        object.__setattr__(self, "missing_evidence_ids", tuple(self.missing_evidence_ids))
        object.__setattr__(self, "missing_source_ids", tuple(self.missing_source_ids))
        object.__setattr__(self, "missing_derivation_ids", tuple(self.missing_derivation_ids))
        object.__setattr__(self, "omitted_claim_ids", tuple(self.omitted_claim_ids))
        object.__setattr__(self, "omitted_conflict_ids", tuple(self.omitted_conflict_ids))
        object.__setattr__(self, "omitted_evidence_ids", tuple(self.omitted_evidence_ids))
        object.__setattr__(self, "omitted_source_ids", tuple(self.omitted_source_ids))
        object.__setattr__(self, "omitted_derivation_ids", tuple(self.omitted_derivation_ids))
        object.__setattr__(self, "omitted_resolution_ids", tuple(self.omitted_resolution_ids))
        object.__setattr__(self, "search_failures", tuple(self.search_failures))

    @property
    def truncated(self) -> bool:
        return any(
            (
                self.omitted_claim_ids,
                self.omitted_conflict_ids,
                self.omitted_evidence_ids,
                self.omitted_source_ids,
                self.omitted_derivation_ids,
                self.omitted_resolution_ids,
            )
        )
