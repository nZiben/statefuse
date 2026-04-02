# StateFuse

StateFuse is a lightweight Python library for deterministic, conflict-preserving agent memory built on standard OpSet/CRDT principles.

## Overview

StateFuse gives agent systems a mergeable memory substrate with:

- immutable operation history
- explicit surfaced conflicts instead of silent overwrite
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
mem.add_claim(
    namespace="project",
    subject="launch",
    predicate="deadline",
    value="2026-04-10",
    confidence=0.8,
)
```

For deterministic or content-addressed deployments:

```python
from statefuse import Memory

mem = Memory(replica_id="agent-a", op_id_mode="content-addressed")
```

Strict `merge()` keeps fail-fast behavior on invalid `op_id` payload collisions. Use `merge_checked()` when a sync pipeline should quarantine bad collisions instead of failing the whole merge.

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

## CI

CI is configured in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) and runs lint, tests, and package build checks.
