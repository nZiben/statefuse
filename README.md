# StateFuse

StateFuse is a lightweight Python library for deterministic, conflict-preserving agent memory built on standard OpSet/CRDT principles.

## Overview

StateFuse gives agent systems a mergeable memory substrate with:

- immutable operation history
- canonical source, evidence, and derivation records
- temporal claim validity without changing assertion timestamps
- explicit surfaced conflicts instead of silent overwrite
- append-only conflict lifecycle and committed, scope-aware resolutions
- exact and semantic correction handles via `claim_id` and `claim_ref`
- deterministic materialization and projection
- optional authenticated merge checks and bounded compaction helpers

The project focuses on the memory contract exposed to applications rather than on inventing a new CRDT join.

## Install

```bash
python3 -m pip install -e .
```

For LLM-backed resolution support:

```bash
python3 -m pip install -e ".[llm]"
```

## Quick Start

```python
from statefuse import Memory

mem = Memory(replica_id="agent-a")
source_id = mem.add_source(
    source_type="user_message",
    actor_id="user-1",
    message_id="message-1",
)
evidence_id = mem.add_evidence(
    pointer="message://message-1",
    content="The launch deadline is 2026-04-10.",
    source_id=source_id,
)
mem.add_claim(
    namespace="project",
    subject="launch",
    predicate="deadline",
    value="2026-04-10",
    confidence=0.8,
    evidence_ids=[evidence_id],
)
```

For deterministic or content-addressed deployments:

```python
from statefuse import Memory

mem = Memory(replica_id="agent-a", op_id_mode="content-addressed")
```

Strict `merge()` keeps fail-fast behavior on invalid `op_id` payload collisions. Use `merge_checked()` when a sync pipeline should quarantine bad collisions instead of failing the whole merge.

## Conflict lifecycle

A derived conflict has two identities: `conflict_id` names the exact candidate snapshot, while `conflict_ref` remains stable as candidates arrive or are retracted. A `ResolutionRecord` is an explicit application or human commit; the existing resolver result remains an ephemeral projection-time recommendation.

`build_view()` applies a committed resolution only in its matching scope and only while every current candidate is covered. The selected claim changes in that view, but the conflict remains in `surfaced_conflicts`. An uncovered candidate makes the resolution stale/reopened instead of inheriting it silently. Sources, evidence, claim validity, derivations, resolutions, and lifecycle events all remain immutable operations in the canonical log.

Until StateFuse has a signed operation envelope, `merge_checked_authenticated(..., require_signed=True)` quarantines resolution and lifecycle operations rather than trusting unsigned authority changes.

## Development

Install dev dependencies:

```bash
python3 -m pip install -e . pytest ruff build
```

Run local checks:

```bash
ruff check .
python3 -m pytest -q
python3 -m build
```

## LLM Smoke Check

If you are using an OpenAI-compatible endpoint, you can validate the client configuration before running demos:

```bash
cp .env.test.example .env.test
python3 scripts/llm_endpoint_smoke.py --env-file .env.test
```

## Examples

- `examples/01_basic_ops.py`
- `examples/02_branch_diverge_merge.py`
- `examples/03_llm_resolver_demo.py`
- `examples/env_test.py`

## Citation

If you use StateFuse, please cite the preprint:

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
