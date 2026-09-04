from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from statefuse import (
    Claim,
    ClaimAdded,
    ClaimKey,
    Evidence,
    EvidenceAdded,
    InMemoryStore,
    JsonlStore,
    Memory,
    Source,
    SourceAdded,
    SQLiteStore,
)
from statefuse.materialize import apply_operations_incrementally


def _batch() -> tuple[SourceAdded, EvidenceAdded, ClaimAdded]:
    source = SourceAdded(
        op_id="op-source",
        replica_id="test",
        timestamp="2026-03-01T00:00:00.000000Z",
        source=Source(source_id="source-1", source_type="message"),
    )
    evidence = EvidenceAdded(
        op_id="op-evidence",
        replica_id="test",
        timestamp="2026-03-01T00:00:01.000000Z",
        evidence=Evidence(
            evidence_id="evidence-1",
            pointer="message://1",
            source_id="source-1",
        ),
    )
    claim = ClaimAdded(
        op_id="op-claim",
        replica_id="test",
        timestamp="2026-03-01T00:00:02.000000Z",
        claim=Claim(
            claim_id="claim-1",
            key=ClaimKey("project", "deadline", "date"),
            value="May 12",
            confidence=0.8,
            timestamp="2026-03-01T00:00:02.000000Z",
            evidence_ids=("evidence-1",),
            provenance={"replica_id": "test"},
        ),
    )
    return source, evidence, claim


@pytest.mark.parametrize("store_type", [InMemoryStore, JsonlStore, SQLiteStore])
def test_store_batch_commits_source_evidence_and_claim_once(tmp_path, store_type) -> None:  # type: ignore[no-untyped-def]
    store = (
        store_type()
        if store_type is InMemoryStore
        else store_type(tmp_path / ("ops.jsonl" if store_type is JsonlStore else "ops.sqlite"))
    )
    ops = _batch()

    assert store.append_many(*ops) == (True, True, True)
    assert store.append_many(*ops) == (False, False, False)
    assert store.load_oplog().op_ids() == tuple(sorted(op.op_id for op in ops))
    assert Memory(store=store, replica_id="test").materialize().claims_by_id == {
        "claim-1": ops[-1].claim
    }


def test_memory_materializes_one_committed_batch_as_one_unit() -> None:
    memory = Memory(store=InMemoryStore(), replica_id="test")
    memory.materialize()
    ops = _batch()

    with patch(
        "statefuse.memory.apply_operations_incrementally",
        wraps=apply_operations_incrementally,
    ) as apply:
        result = memory.commit_batch(ops)
        state = memory.materialize()

    assert result == (True, True, True)
    assert apply.call_count == 1
    assert apply.call_args.args[1] == ops
    assert state.claims_by_id["claim-1"] == ops[-1].claim


@pytest.mark.parametrize("store_type", [InMemoryStore, JsonlStore, SQLiteStore])
def test_collision_validation_leaves_the_batch_uncommitted(tmp_path, store_type) -> None:  # type: ignore[no-untyped-def]
    store = (
        store_type()
        if store_type is InMemoryStore
        else store_type(tmp_path / ("ops.jsonl" if store_type is JsonlStore else "ops.sqlite"))
    )
    original = SourceAdded(
        op_id="collision",
        replica_id="test",
        timestamp="2026-03-01T00:00:00.000000Z",
        source=Source(source_id="original", source_type="message"),
    )
    conflicting = SourceAdded(
        op_id="collision",
        replica_id="test",
        timestamp="2026-03-01T00:00:00.000000Z",
        source=Source(source_id="different", source_type="message"),
    )
    store.append(original)

    with pytest.raises(ValueError, match="op_id collision"):
        store.append_many(_batch()[1], conflicting, _batch()[2])

    assert store.load_oplog().op_ids() == ("collision",)


def test_sqlite_rolls_back_an_injected_mid_batch_failure(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "ops.sqlite")
    with store._connect() as connection:
        connection.execute(
            """
            CREATE TRIGGER inject_batch_failure
            BEFORE INSERT ON ops
            WHEN NEW.op_id = 'op-evidence'
            BEGIN
                SELECT RAISE(ABORT, 'injected batch failure');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="injected batch failure"):
        store.append_many(*_batch())

    assert store.load_oplog().op_ids() == ()


def test_jsonl_mid_batch_failure_is_prefix_visible_and_retryable(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    store = JsonlStore(tmp_path / "ops.jsonl")
    real_open = Path.open
    writes = 0

    class FailingWriter:
        def __init__(self, wrapped):  # type: ignore[no-untyped-def]
            self.wrapped = wrapped

        def __enter__(self):  # type: ignore[no-untyped-def]
            self.wrapped.__enter__()
            return self

        def __exit__(self, *args):  # type: ignore[no-untyped-def]
            return self.wrapped.__exit__(*args)

        def write(self, value):  # type: ignore[no-untyped-def]
            nonlocal writes
            writes += 1
            if writes == 1:
                self.wrapped.write(value[: len(value) // 2])
                self.wrapped.flush()
                raise OSError("injected write failure")
            return self.wrapped.write(value)

    def failing_open(path, mode="r", *args, **kwargs):  # type: ignore[no-untyped-def]
        opened = real_open(path, mode, *args, **kwargs)
        if path == store.path and mode == "a":
            return FailingWriter(opened)
        return opened

    with monkeypatch.context() as context:
        context.setattr(Path, "open", failing_open)
        with pytest.raises(OSError, match="injected write failure"):
            store.append_many(*_batch())

    assert store.load_oplog().op_ids() == ()
    assert store.append_many(*_batch()) == (True, True, True)
    assert store.load_oplog().op_ids() == ("op-claim", "op-evidence", "op-source")
    assert store.batch_atomicity == "prefix-visible"


@pytest.mark.parametrize("store_type", [InMemoryStore, JsonlStore, SQLiteStore])
def test_limited_reads_never_split_a_committed_batch(tmp_path, store_type) -> None:  # type: ignore[no-untyped-def]
    store = (
        store_type()
        if store_type is InMemoryStore
        else store_type(tmp_path / ("ops.jsonl" if store_type is JsonlStore else "ops.sqlite"))
    )
    store.append_many(*_batch())

    delta = store.read_after(0, limit=2)

    assert delta.ops == _batch()
    assert delta.batch_sizes == (3,)


def test_jsonl_reads_legacy_single_operation_lines(tmp_path) -> None:
    path = tmp_path / "ops.jsonl"
    path.write_text(_batch()[0].to_json() + "\n", encoding="utf-8")

    delta = JsonlStore(path).read_after(0)

    assert delta.ops == (_batch()[0],)
    assert delta.batch_sizes == (1,)


def test_sqlite_migrates_an_existing_operation_table(tmp_path) -> None:
    path = tmp_path / "ops.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE ops (op_id TEXT PRIMARY KEY, op_type TEXT NOT NULL, "
            "ts TEXT NOT NULL, payload TEXT NOT NULL)"
        )

    store = SQLiteStore(path)
    store.append_many(*_batch())

    assert store.read_after(0, limit=1).batch_sizes == (3,)


def test_memory_rejects_a_store_without_multi_append() -> None:
    class SingleAppendStore:
        def __init__(self) -> None:
            self.inner = InMemoryStore()

        def append(self, op):  # type: ignore[no-untyped-def]
            return self.inner.append(op)

        def iter_ops(self):  # type: ignore[no-untyped-def]
            return self.inner.iter_ops()

        def has(self, op_id):  # type: ignore[no-untyped-def]
            return self.inner.has(op_id)

        def load_oplog(self):  # type: ignore[no-untyped-def]
            return self.inner.load_oplog()

    memory = Memory(store=SingleAppendStore(), replica_id="test")

    with pytest.raises(TypeError, match="does not support batch commits"):
        memory.commit_batch(_batch())
