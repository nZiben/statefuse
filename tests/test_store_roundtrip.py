from __future__ import annotations

from pathlib import Path

from statefuse.merge import merge
from statefuse.model import Claim, ClaimKey, Evidence
from statefuse.oplog import OpLog
from statefuse.ops import ClaimAdded, EvidenceAdded
from statefuse.store import InMemoryStore, JsonlStore, SQLiteStore


def _sample_ops() -> list[EvidenceAdded | ClaimAdded]:
    evidence = EvidenceAdded(
        op_id="op-1",
        replica_id="a",
        timestamp="2026-03-01T00:00:00.000000Z",
        evidence=Evidence(evidence_id="sha256:1", pointer="doc://one", metadata={}),
    )
    claim = ClaimAdded(
        op_id="op-2",
        replica_id="a",
        timestamp="2026-03-01T00:00:01.000000Z",
        claim=Claim(
            claim_id="claim-1",
            key=ClaimKey(namespace="proj", subject="deadline", predicate="date"),
            value="2026-03-25",
            confidence=0.8,
            timestamp="2026-03-01T00:00:01.000000Z",
            evidence_ids=("sha256:1",),
            provenance={"replica_id": "a"},
        ),
    )
    return [evidence, claim]


def _assert_roundtrip(store) -> None:  # type: ignore[no-untyped-def]
    for op in _sample_ops():
        store.append(op)
    loaded = store.load_oplog()
    assert loaded.op_ids() == ("op-1", "op-2")


def test_jsonl_store_roundtrip(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path / "ops.jsonl")
    _assert_roundtrip(store)


def test_sqlite_store_roundtrip(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "ops.db")
    _assert_roundtrip(store)


def test_mixed_append_load_merge_invariants(tmp_path: Path) -> None:
    left = InMemoryStore()
    right = JsonlStore(tmp_path / "right.jsonl")
    for op in _sample_ops():
        left.append(op)
    right.append(
        EvidenceAdded(
            op_id="op-3",
            replica_id="b",
            timestamp="2026-03-01T00:00:02.000000Z",
            evidence=Evidence(evidence_id="sha256:3", pointer="doc://three", metadata={}),
        )
    )

    merged = merge(left.load_oplog(), right.load_oplog())
    assert merged.op_ids() == ("op-1", "op-2", "op-3")
    assert merged == OpLog(merged.iter_ops())
