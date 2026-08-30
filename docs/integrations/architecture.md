# Integration architecture

StateFuse is the authoritative record. External memory repositories are retrieval indexes or
projections: they may decide what looks relevant, but they do not decide what was claimed,
whether a claim is active, whether claims conflict, or how a conflict was resolved.

The integration flow is deliberately one-way:

```text
commit immutable StateFuse operation
    -> materialize canonical StateFuse state
    -> build deterministic retrieval projections
    -> synchronize an external repository
```

Projection failure never rolls back or edits the committed operation. `ProjectionService`
returns a `SyncReport` with each failed upsert or deletion so callers can retry or alert. It
does not silently turn a failed external write into success.

## Projections

`project_state()` emits `claim`, `conflict`, and `resolution` records. Their logical IDs are:

```text
statefuse:claim:<claim_id>
statefuse:conflict:<conflict_id>
statefuse:resolution:<resolution_id>
```

IDs stay stable across rewrites of retrieval text. `projection_version` is tracked separately.
An external ID is kept in `ExternalReferenceStore` under the repository, namespace, and logical
projection ID; it is never embedded in a StateFuse claim or operation ID.

Only active claims and current conflicts are projected. When they cease to be searchable,
synchronization deletes their disposable projections. Resolution records and canonical history
remain in StateFuse. Clearing the external repository and reference store is safe because every
projection can be rebuilt from canonical state.

## Retrieval and hydration

Search produces normalized `SearchHit` values, not authoritative claims. `hydrate_search_hits()`
loads current StateFuse state, removes nonexistent entity references, deduplicates hits and entity
IDs, and reports current claim and conflict statuses. A hit for a retracted claim is returned with
`claim_statuses[claim_id] == "inactive"`; stale external text cannot reactivate it.

## Included connectors

The connector layer implements four repository mappings:

| Connector | External representation | Interface |
| --- | --- | --- |
| `Mem0Adapter` / `AsyncMem0Adapter` | direct-import memories, scoped by `user_id` | sync / async |
| `LangMemAdapter` / `AsyncLangMemAdapter` | LangGraph Store items with deterministic keys | sync / async |
| `LettaAdapter` | archive passages with deterministic namespace/projection tags | sync |
| `AsyncGraphitiAdapter` | JSON episodes with deterministic UUIDs and group IDs | async |

Mem0 uses `infer=False`; an external LLM never rewrites projection text during ingestion. Letta
creates a replacement passage before deleting stale passages so a retry can reconcile a partial
replace. Graphiti uses its native episode and episode-search APIs. LangMem integration targets the
LangGraph Store contract used by LangMem rather than making an LLM memory manager canonical.

Sync adapters run through `ProjectionService`; native async adapters run through
`AsyncProjectionService`. Both services preserve the same reference and failure semantics.

## Dependency boundary

Core imports never load external SDKs. Connector modules import their SDK lazily and raise an
installation message for the connector extra, for example:

```text
Install the Mem0 integration with:
pip install "statefuse[mem0]"
```

Supplying an already-constructed client/store is supported for dependency injection and testing.
Third-party response objects are normalized inside connector modules and never appear in
`RetrievalRecord`, `SearchHit`, `ExternalReference`, or canonical StateFuse state.
