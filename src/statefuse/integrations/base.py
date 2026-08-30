from __future__ import annotations

from typing import Protocol

from .models import (
    ExternalReference,
    ExternalWriteResult,
    RetrievalRecord,
    SearchHit,
    SearchRequest,
)


class MemoryRepositoryAdapter(Protocol):
    name: str

    def upsert(self, record: RetrievalRecord) -> ExternalWriteResult:
        ...

    def search(self, request: SearchRequest) -> list[SearchHit]:
        ...

    def delete(self, projection_id: str, namespace: str) -> bool:
        ...

    def healthcheck(self) -> bool:
        ...


class AsyncMemoryRepositoryAdapter(Protocol):
    name: str

    async def aupsert(self, record: RetrievalRecord) -> ExternalWriteResult:
        ...

    async def asearch(self, request: SearchRequest) -> list[SearchHit]:
        ...

    async def adelete(self, projection_id: str, namespace: str) -> bool:
        ...

    async def ahealthcheck(self) -> bool:
        ...


class ExternalReferenceStore(Protocol):
    def get(
        self, repository: str, namespace: str, projection_id: str
    ) -> ExternalReference | None:
        ...

    def get_by_external_id(
        self, repository: str, namespace: str, external_id: str
    ) -> ExternalReference | None:
        ...

    def upsert(self, reference: ExternalReference) -> None:
        ...

    def delete(self, repository: str, namespace: str, projection_id: str) -> bool:
        ...

    def list(self, repository: str, namespace: str) -> tuple[ExternalReference, ...]:
        ...
