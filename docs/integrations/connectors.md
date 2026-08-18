# Included connectors

All examples synchronize after the canonical StateFuse operation has committed. Keep the returned
`SyncReport`; a failure means the external index needs retry, not that the StateFuse operation
failed.

## Mem0

```python
from mem0 import Memory as Mem0Memory
from statefuse.integrations import Mem0Adapter, ProjectionService

adapter = Mem0Adapter(Mem0Memory())
service = ProjectionService(memory, adapter, reference_store)
report = service.synchronize("project")
```

`Mem0Adapter` stores projection text with direct import (`infer=False`) and scopes the record by the
StateFuse namespace. `AsyncMem0Adapter` accepts Mem0's `AsyncMemory` and works with
`AsyncProjectionService`.

## LangMem / LangGraph Store

```python
from langgraph.store.memory import InMemoryStore
from statefuse.integrations import LangMemAdapter, ProjectionService

adapter = LangMemAdapter(InMemoryStore())
service = ProjectionService(memory, adapter, reference_store)
report = service.synchronize("project")
```

Pass a production `BaseStore` implementation for persistence. The adapter uses deterministic
external keys and indexes the `text` field. `AsyncLangMemAdapter` calls the store's native `aget`,
`aput`, `asearch`, and `adelete` methods.

## Letta

```python
from letta_client import Letta
from statefuse.integrations import LettaAdapter, ProjectionService

adapter = LettaAdapter("archive-your-id", Letta())
service = ProjectionService(memory, adapter, reference_store)
report = service.synchronize("project")
```

StateFuse projections become Letta archive passages. Deterministic tags isolate namespaces,
logical projection IDs, and metadata filters. Passage IDs remain external references only.

Use a separate Letta archive for each tenant-level isolation boundary. Letta server 0.16.x applies
tag filtering after its 100-result candidate cap; the adapter over-fetches that maximum and keeps a
write-through projection mapping so repeated writes remain idempotent, but a single very large
shared archive can still starve a namespace from semantic search. StateFuseBench therefore uses one
archive per paired episode and deletes it after measurement.

## Graphiti

```python
import os

from graphiti_core import Graphiti
from statefuse.integrations import AsyncGraphitiAdapter, AsyncProjectionService

graphiti = Graphiti(
    os.environ["NEO4J_URI"],
    os.environ["NEO4J_USER"],
    os.environ["NEO4J_PASSWORD"],
)
adapter = AsyncGraphitiAdapter(graphiti)
service = AsyncProjectionService(memory, adapter, reference_store)
report = await service.synchronize("project")
```

Graphiti projections are JSON episodes in deterministic namespace groups. Search uses Graphiti's
episode search and returns normalized `SearchHit` values, which must still be hydrated against
current StateFuse state.

By default, Graphiti processes each projection episode and extracts entities and edges. If the
repository is used only for StateFuse's already-structured episode BM25 projection, pass
`process_episodes=False` to store it through Graphiti's native `EpisodicNode` CRUD without LLM
extraction. This does not change StateFuse projection or hydration semantics.
