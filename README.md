# StateFuse

Deterministic, conflict-preserving memory for AI agents.

StateFuse stores memory as an immutable operation log that replicas can merge without silently
overwriting competing claims. Applications get a deterministic view of active claims, provenance,
conflicts, retractions, and committed resolutions.

## Why StateFuse

- Merge memory across agents and replicas deterministically.
- Preserve sources, evidence, derivations, corrections, and history.
- Surface conflicting claims instead of choosing one silently.
- Resolve conflicts explicitly and keep stale resolutions from applying to new candidates.
- Use external memory systems for retrieval without making them the source of truth.

## Installation

StateFuse requires Python 3.10 or later.

```bash
python3 -m pip install -e .
```

Install optional LLM-backed resolution support with:

```bash
python3 -m pip install -e ".[llm]"
```

## Quick start

```python
from statefuse import Memory

memory = Memory(replica_id="agent-a")

source_id = memory.add_source(
    source_type="user_message",
    actor_id="user-1",
    message_id="message-1",
)
evidence_id = memory.add_evidence(
    pointer="message://message-1",
    content="The launch deadline is 2026-04-10.",
    source_id=source_id,
)
memory.add_claim(
    namespace="project",
    subject="launch",
    predicate="deadline",
    value="2026-04-10",
    confidence=0.8,
    evidence_ids=[evidence_id],
)
```

Operations extracted from one logical event can be committed together with
`memory.commit_batch(ops)`. `InMemoryStore` and `SQLiteStore` make the batch atomically visible.
`JsonlStore` stores each batch in one envelope line and ignores an incomplete final line; retrying
appends the complete batch. JSONL is intended for one writer—use SQLite when writers are concurrent.
Operation-ID idempotency and collision checks are identical to `append()`.

See [`examples/`](examples/) for branching, merging, and conflict-resolution flows.

For context/validity-aware detection, taxonomy annotations, multi-key domain detectors, and
preserve/abstain outcomes, see [Taxonomy-aware conflicts](docs/conflict-taxonomy.md).

## Incremental replica sync

Replicas can pull only the unseen suffix of a peer's append-only log. Receive cursors are stored
per peer and survive restarts with `JsonlStore` and `SQLiteStore`.

```python
from statefuse import JsonlStore, Memory

sender = Memory(JsonlStore("sender.jsonl"), replica_id="agent-a")
receiver = Memory(JsonlStore("receiver.jsonl"), replica_id="agent-b")

report = receiver.sync_from(sender, max_ops=500)
```

See [Replica delta sync](docs/replica-sync.md) for retry, reordering, and compaction behavior.

## Adapters

StateFuse can project canonical memory into external retrieval systems. These systems remain
disposable indexes: search results are hydrated against current StateFuse state before use, so
stale external text cannot reactivate a retracted claim or hide a conflict.

| Adapter | Install extra | Interface |
| --- | --- | --- |
| Mem0 | `statefuse[mem0]` | sync and async |
| LangMem / LangGraph Store | `statefuse[langmem]` | sync and async |
| Letta archive passages | `statefuse[letta]` | sync |
| Graphiti | `statefuse[graphiti]` | async |

```python
from mem0 import Memory as Mem0Memory
from statefuse.integrations import (
    InMemoryExternalReferenceStore,
    Mem0Adapter,
    ProjectionService,
    SearchRequest,
)

service = ProjectionService(
    memory,
    Mem0Adapter(Mem0Memory()),
    InMemoryExternalReferenceStore(),
)

report = service.synchronize("project")
context = service.search(SearchRequest("launch deadline", "project"))
```

Synchronize only after the StateFuse operation commits. Adapter failures are returned in
`report.failed`; they do not roll back canonical memory.

- [Adapter architecture](docs/integrations/architecture.md)
- [Included adapters](docs/integrations/connectors.md)
- [Build a custom adapter](docs/integrations/custom-adapter.md)
- [Test an adapter](docs/integrations/testing.md)

## Incremental materialization

Built-in stores expose cursors so repeated `Memory.materialize()` calls process only newly
appended operations. JSONL and SQLite stores also save versioned checkpoints every 100 applied
operations; incompatible checkpoint versions fall back to a full rebuild.

Custom conflict detectors always use full materialization. A custom `PredicateRegistry` is only
checkpointed when `materialization_config_token` is supplied; change that token whenever predicate
semantics change. Runtime registry changes invalidate the in-memory cache automatically.

## Development

```bash
python3 -m pip install -e . pytest ruff build
ruff check .
python3 -m pytest -q
python3 -m build
```
