from __future__ import annotations

from dataclasses import dataclass, field

from ..conflict import ConflictSet
from ..model import Claim, JSONValue


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
class HydratedContext:
    claims: tuple[Claim, ...]
    conflicts: tuple[ConflictSet, ...]
    missing_claim_ids: tuple[str, ...]
    missing_conflict_ids: tuple[str, ...]
    search_hits: tuple[SearchHit, ...]
    claim_statuses: dict[str, str] = field(default_factory=dict)
    conflict_statuses: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_statuses", dict(self.claim_statuses))
        object.__setattr__(self, "conflict_statuses", dict(self.conflict_statuses))
