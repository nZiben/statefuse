from statefuse import Memory
from statefuse.integrations import (
    FakeMemoryRepositoryAdapter,
    InMemoryExternalReferenceStore,
    ProjectionService,
    SearchRequest,
)

memory = Memory(replica_id="demo")
memory.add_claim(
    namespace="project",
    subject="Project Alpha submission deadline",
    predicate="date",
    value="May 15, 2026",
    confidence=0.9,
    evidence_ids=("ev_21",),
    claim_id="cl_123",
)

service = ProjectionService(
    memory,
    FakeMemoryRepositoryAdapter(),
    InMemoryExternalReferenceStore(),
)
print(service.synchronize("project"))
print(service.search(SearchRequest("submission deadline", "project")))
