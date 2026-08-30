from __future__ import annotations

import re

from ..utils import digest_json_value
from .errors import AdapterUnavailableError
from .models import ExternalWriteResult, RetrievalRecord, SearchHit, SearchRequest


class FakeMemoryRepositoryAdapter:
    """Deterministic in-memory implementation of the adapter contract."""

    name = "fake"

    def __init__(self) -> None:
        self.available = True
        self._records: dict[tuple[str, str], RetrievalRecord] = {}
        self._failures: dict[str, list[Exception]] = {}
        self.write_count = 0
        self.duplicate_write_count = 0

    @property
    def record_count(self) -> int:
        return len(self._records)

    def inject_failure(self, operation: str, error: Exception | None = None) -> None:
        self._failures.setdefault(operation, []).append(
            error or AdapterUnavailableError(f"Injected {operation} failure.")
        )

    def upsert(self, record: RetrievalRecord) -> ExternalWriteResult:
        self._check("upsert")
        key = (record.namespace, record.projection_id)
        previous = self._records.get(key)
        if previous == record:
            self.duplicate_write_count += 1
        else:
            self._records[key] = record
            self.write_count += 1
        return ExternalWriteResult(
            repository=self.name,
            projection_id=record.projection_id,
            external_id=self._external_id(*key),
            created=previous is None,
            metadata=dict(record.metadata),
        )

    def search(self, request: SearchRequest) -> list[SearchHit]:
        self._check("search")
        query_terms = self._terms(request.query)
        hits: list[SearchHit] = []
        for (namespace, projection_id), record in self._records.items():
            if namespace != request.namespace or any(
                record.metadata.get(key) != value for key, value in request.filters.items()
            ):
                continue
            score = len(query_terms & self._terms(record.text)) / len(query_terms or {""})
            if query_terms and score == 0:
                continue
            hits.append(
                SearchHit(
                    external_id=self._external_id(namespace, projection_id),
                    projection_id=projection_id,
                    text=record.text,
                    score=score,
                    claim_ids=record.claim_ids,
                    conflict_ids=record.conflict_ids,
                    metadata=dict(record.metadata),
                )
            )
        hits.sort(key=lambda hit: (-(hit.score or 0.0), hit.external_id))
        return hits[: request.limit]

    def delete(self, projection_id: str, namespace: str) -> bool:
        self._check("delete")
        return self._records.pop((namespace, projection_id), None) is not None

    def healthcheck(self) -> bool:
        self._raise_injected("healthcheck")
        return self.available

    def _check(self, operation: str) -> None:
        self._raise_injected(operation)
        if not self.available:
            raise AdapterUnavailableError("Fake memory repository is unavailable.")

    def _raise_injected(self, operation: str) -> None:
        failures = self._failures.get(operation, [])
        if failures:
            raise failures.pop(0)

    @staticmethod
    def _external_id(namespace: str, projection_id: str) -> str:
        return f"fake:{digest_json_value({'namespace': namespace, 'projection_id': projection_id})}"

    @staticmethod
    def _terms(text: str) -> set[str]:
        # ponytail: lexical matching defines the fake contract; real adapters provide semantics.
        return set(re.findall(r"\w+", text.casefold()))
