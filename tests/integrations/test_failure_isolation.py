from __future__ import annotations

from statefuse import Memory
from statefuse.integrations import (
    FakeMemoryRepositoryAdapter,
    InMemoryExternalReferenceStore,
    ProjectionService,
    SearchHit,
    hydrate_search_hits,
)


def test_adapter_outage_does_not_remove_canonical_operations() -> None:
    memory = _memory()
    adapter = FakeMemoryRepositoryAdapter()
    adapter.available = False
    service = ProjectionService(memory, adapter, InMemoryExternalReferenceStore())
    operation_ids = memory.load_oplog().op_ids()

    failed = service.synchronize("project")

    assert failed.created == ()
    assert failed.failed[0].error_type == "AdapterUnavailableError"
    assert memory.load_oplog().op_ids() == operation_ids
    assert memory.materialize().claims_by_id["cl_1"].value == "May 12"

    adapter.available = True
    assert service.synchronize("project").created == ("statefuse:claim:cl_1",)


def test_failed_update_drops_only_disposable_reference_and_retries() -> None:
    memory = _memory()
    adapter = FakeMemoryRepositoryAdapter()
    references = InMemoryExternalReferenceStore()
    service = ProjectionService(memory, adapter, references)
    service.synchronize("project")
    memory.add_claim(
        namespace="project",
        subject="Project Alpha submission deadline",
        predicate="date",
        value="May 15",
        confidence=0.9,
        evidence_ids=(),
        claim_id="cl_2",
    )
    adapter.inject_failure("upsert")

    failed = service.synchronize("project")

    assert failed.failed
    assert references.get("fake", "project", "statefuse:claim:cl_1") is None
    recovered = service.synchronize("project")
    assert "statefuse:claim:cl_1" in recovered.created
    assert memory.materialize().claims_by_id.keys() >= {"cl_1", "cl_2"}


def test_stale_external_hit_marks_retracted_claim_inactive() -> None:
    memory = _memory()
    stale_hit = _hit(claim_ids=("cl_1",))
    memory.retract_claim(target_claim_id="cl_1", evidence_ids=(), reason="Withdrawn.")

    context = hydrate_search_hits(memory, [stale_hit])

    assert [claim.claim_id for claim in context.claims] == ["cl_1"]
    assert context.claim_statuses == {"cl_1": "inactive"}
    assert context.missing_claim_ids == ()


def test_duplicate_unknown_and_invalid_statefuse_references_are_safe() -> None:
    memory = _memory()
    hit = _hit(
        external_id="unknown-external-id",
        claim_ids=("cl_1", "missing", "", 3),  # type: ignore[arg-type]
        conflict_ids=("missing-conflict",),
        metadata={},
    )

    context = hydrate_search_hits(memory, [hit, hit])

    assert len(context.search_hits) == 1
    assert [claim.claim_id for claim in context.claims] == ["cl_1"]
    assert context.missing_claim_ids == ("missing",)
    assert context.missing_conflict_ids == ("missing-conflict",)


def _memory() -> Memory:
    memory = Memory(replica_id="test")
    memory.add_claim(
        namespace="project",
        subject="Project Alpha submission deadline",
        predicate="date",
        value="May 12",
        confidence=0.8,
        evidence_ids=(),
        claim_id="cl_1",
    )
    return memory


def _hit(
    *,
    external_id: str = "external-1",
    claim_ids: tuple[str, ...] = (),
    conflict_ids: tuple[str, ...] = (),
    metadata: dict | None = None,  # type: ignore[type-arg]
) -> SearchHit:
    return SearchHit(
        external_id=external_id,
        projection_id=None,
        text="stale external text",
        score=None,
        claim_ids=claim_ids,
        conflict_ids=conflict_ids,
        metadata=metadata or {},
    )
