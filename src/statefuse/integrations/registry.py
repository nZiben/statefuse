from __future__ import annotations

from .models import ExternalReference


class InMemoryExternalReferenceStore:
    def __init__(self) -> None:
        self._references: dict[tuple[str, str, str], ExternalReference] = {}

    def get(
        self, repository: str, namespace: str, projection_id: str
    ) -> ExternalReference | None:
        return self._references.get((repository, namespace, projection_id))

    def get_by_external_id(
        self, repository: str, namespace: str, external_id: str
    ) -> ExternalReference | None:
        # ponytail: linear scan is enough in memory; a database store should index external_id.
        return next(
            (
                reference
                for reference in self.list(repository, namespace)
                if reference.external_id == external_id
            ),
            None,
        )

    def upsert(self, reference: ExternalReference) -> None:
        self._references[
            (reference.repository, reference.namespace, reference.projection_id)
        ] = reference

    def delete(self, repository: str, namespace: str, projection_id: str) -> bool:
        return self._references.pop((repository, namespace, projection_id), None) is not None

    def list(self, repository: str, namespace: str) -> tuple[ExternalReference, ...]:
        return tuple(
            sorted(
                (
                    reference
                    for (name, item_namespace, _), reference in self._references.items()
                    if name == repository and item_namespace == namespace
                ),
                key=lambda reference: reference.projection_id,
            )
        )
