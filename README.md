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

See [`examples/`](examples/) for branching, merging, and conflict-resolution flows.

For context/validity-aware detection, taxonomy annotations, multi-key domain detectors, and
preserve/abstain outcomes, see [Taxonomy-aware conflicts](docs/conflict-taxonomy.md).

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

## Development

```bash
python3 -m pip install -e . pytest ruff build
ruff check .
python3 -m pytest -q
python3 -m build
```

## Citation

```bibtex
@misc{volkov2026statefuse,
  title={StateFuse: Deterministic Conflict-Preserving Memory for Multi-Agent Systems},
  author={Volkov, Sergey and Li, Yang and Luo, Ye},
  year={2026},
  eprint={2607.05844},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  doi={10.48550/arXiv.2607.05844},
  url={https://arxiv.org/abs/2607.05844}
}
```
