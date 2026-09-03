from __future__ import annotations

from pathlib import Path

import pytest

from statefuse import (
    InMemoryStore,
    JsonlStore,
    Memory,
    ReplicaDelta,
    ReplicaProgress,
    SQLiteStore,
)
from statefuse.ops import AnyOp


def _populate(memory: Memory, count: int, *, start: int = 0) -> tuple[str, ...]:
    op_ids: list[str] = []
    for index in range(start, start + count):
        evidence_id = f"evidence-{index}"
        evidence_op_id = f"{memory.replica_id}-evidence-{index}"
        memory.add_evidence(
            pointer=f"doc://{index}",
            evidence_id=evidence_id,
            op_id=evidence_op_id,
        )
        memory.add_claim(
            namespace="replica-sync",
            subject=f"item-{index}",
            predicate="status",
            value=f"value-{index}",
            confidence=0.9,
            evidence_ids=[evidence_id],
            claim_id=f"claim-{memory.replica_id}-{index}",
            op_id=f"{memory.replica_id}-claim-{index}",
        )
        op_ids.extend((evidence_op_id, f"{memory.replica_id}-claim-{index}"))
    return tuple(op_ids)


def test_repeated_sync_transfers_no_already_seen_operations() -> None:
    sender = Memory(replica_id="sender")
    receiver = Memory(replica_id="receiver")
    expected_op_ids = _populate(sender, 3)

    first = receiver.sync_from(sender)
    second = receiver.sync_from(sender)

    assert first.transferred_op_ids == expected_op_ids
    assert first.applied_op_ids == expected_op_ids
    assert first.duplicate_op_ids == ()
    assert second.transferred_op_ids == ()
    assert second.applied_op_ids == ()
    assert second.duplicate_op_ids == ()
    assert receiver.load_oplog() == sender.load_oplog()
    assert receiver.materialize() == sender.materialize()


def test_behind_replica_receives_only_new_log_suffix() -> None:
    sender = Memory(replica_id="sender")
    receiver = Memory(replica_id="receiver")
    first_ids = _populate(sender, 2)
    receiver.sync_from(sender)
    next_ids = _populate(sender, 2, start=2)

    report = receiver.sync_from(sender)

    assert report.transferred_op_ids == next_ids
    assert set(report.transferred_op_ids).isdisjoint(first_ids)
    assert receiver.load_oplog() == sender.load_oplog()


def test_sync_filters_operations_already_known_through_another_path() -> None:
    class InspectableMemory(Memory):
        requested_op_ids: tuple[str, ...] = ()

        def _export_manifest_delta(self, manifest, op_ids):  # type: ignore[no-untyped-def]
            self.requested_op_ids = op_ids
            return super()._export_manifest_delta(manifest, op_ids)

    sender = InspectableMemory(replica_id="sender")
    receiver = Memory(replica_id="receiver")
    expected_op_ids = _populate(sender, 2)
    known = sender.load_oplog().get(expected_op_ids[0])
    assert known is not None
    receiver.append_op(known)

    report = receiver.sync_from(sender)

    assert report.transferred_op_ids == expected_op_ids[1:]
    assert sender.requested_op_ids == expected_op_ids[1:]
    assert report.duplicate_op_ids == ()
    assert receiver.load_oplog() == sender.load_oplog()
    assert receiver.replica_progress("sender").cursor == sender.export_replica_delta(0).cursor_after


def test_sync_rejects_known_op_id_with_different_payload() -> None:
    sender = Memory(replica_id="sender")
    receiver = Memory(replica_id="receiver")
    _populate(sender, 1)
    colliding = next(iter(sender.load_oplog()))
    receiver.add_evidence(
        pointer="doc://different",
        evidence_id="different-evidence",
        op_id=colliding.op_id,
    )

    with pytest.raises(ValueError, match="collision with different payload"):
        receiver.sync_from(sender)

    assert receiver.replica_progress("sender").cursor == 0


def test_out_of_order_and_duplicate_deltas_converge() -> None:
    sender = Memory(replica_id="sender")
    receiver = Memory(replica_id="receiver")
    _populate(sender, 3)
    first = sender.export_replica_delta(0, max_ops=2)
    second = sender.export_replica_delta(first.cursor_after)

    out_of_order = receiver.apply_replica_delta(second)
    repeated = receiver.apply_replica_delta(second)
    completed = receiver.apply_replica_delta(first)

    assert out_of_order.cursor_after == 0
    assert out_of_order.pending_ranges == ((first.cursor_after, second.cursor_after),)
    assert repeated.applied_op_ids == ()
    assert repeated.duplicate_op_ids == tuple(op.op_id for op in second.ops)
    assert completed.cursor_after == second.cursor_after
    assert completed.pending_ranges == ()
    assert receiver.load_oplog() == sender.load_oplog()
    assert receiver.materialize() == sender.materialize()


@pytest.mark.parametrize("store_type", [JsonlStore, SQLiteStore])
def test_replica_progress_survives_receiver_restart(
    tmp_path: Path,
    store_type: type[JsonlStore] | type[SQLiteStore],
) -> None:
    sender = Memory(replica_id="sender")
    expected_op_ids = _populate(sender, 3)
    path = tmp_path / ("receiver.jsonl" if store_type is JsonlStore else "receiver.sqlite")

    first_receiver = Memory(store=store_type(path), replica_id="receiver")
    first = first_receiver.sync_from(sender, max_ops=2)
    saved_cursor = first_receiver.replica_progress("sender").cursor

    restarted_receiver = Memory(store=store_type(path), replica_id="receiver")
    assert restarted_receiver.replica_progress("sender").cursor == saved_cursor
    rest = restarted_receiver.sync_from(sender)

    assert first.transferred_op_ids == expected_op_ids[:2]
    assert rest.transferred_op_ids == expected_op_ids[2:]
    assert restarted_receiver.load_oplog() == sender.load_oplog()


def test_progress_is_independent_for_each_peer() -> None:
    left = Memory(replica_id="left")
    right = Memory(replica_id="right")
    receiver = Memory(replica_id="receiver")
    _populate(left, 1)
    _populate(right, 2)

    receiver.sync_from(left)
    left_cursor = receiver.replica_progress("left").cursor
    receiver.sync_from(right, max_ops=1)

    assert receiver.replica_progress("left").cursor == left_cursor
    assert receiver.replica_progress("right").cursor == 1


def test_full_merge_remains_a_bootstrap_and_cursor_reset_path() -> None:
    sender = Memory(replica_id="sender")
    receiver = Memory(replica_id="receiver")
    _populate(sender, 2)
    receiver.merge_from(sender.store)

    bootstrap = receiver.sync_from(sender)
    receiver.reset_replica_progress("sender")
    replay = receiver.sync_from(sender)

    assert bootstrap.transferred_op_ids == ()
    assert replay.transferred_op_ids == ()
    assert receiver.replica_progress("sender").cursor > 0
    assert receiver.load_oplog() == sender.load_oplog()


class _OneShotFaultStore(InMemoryStore):
    batch_atomicity = "prefix-visible"

    def __init__(self, fail_op_id: str) -> None:
        super().__init__()
        self.fail_op_id = fail_op_id
        self.failed = False

    def append(self, op: AnyOp) -> bool:
        if op.op_id == self.fail_op_id and not self.failed:
            self.failed = True
            raise RuntimeError("injected replica write failure")
        return super().append(op)

    def append_many(self, *ops: AnyOp) -> tuple[bool, ...]:
        return tuple(self.append(op) for op in ops)


def test_partial_receive_failure_retries_without_advancing_progress() -> None:
    sender = Memory(replica_id="sender")
    expected_op_ids = _populate(sender, 2)
    receiver = Memory(
        store=_OneShotFaultStore(fail_op_id=expected_op_ids[1]),
        replica_id="receiver",
    )
    delta = sender.export_replica_delta(0)

    with pytest.raises(RuntimeError, match="injected replica write failure"):
        receiver.apply_replica_delta(delta)
    assert receiver.replica_progress("sender") == ReplicaProgress("sender")

    report = receiver.apply_replica_delta(delta)

    assert report.duplicate_op_ids == (expected_op_ids[0],)
    assert report.applied_op_ids == expected_op_ids[1:]
    assert receiver.load_oplog() == sender.load_oplog()


def test_replica_delta_validation_and_invalid_sync_requests() -> None:
    sender = Memory(replica_id="sender")
    _populate(sender, 1)

    with pytest.raises(ValueError, match="positive"):
        sender.export_replica_delta(0, max_ops=0)
    with pytest.raises(ValueError, match="non-negative integer"):
        sender.export_replica_delta(-1)
    with pytest.raises(ValueError, match="positive integer"):
        sender.export_replica_delta(0, max_ops=True)
    with pytest.raises(ValueError, match="distinct"):
        Memory(replica_id="sender").sync_from(sender)
    with pytest.raises(ValueError, match="zero-width"):
        ReplicaDelta(
            sender_replica_id="sender",
            cursor_before=1,
            cursor_after=1,
            ops=sender.export_replica_delta(0, max_ops=1).ops,
        )


@pytest.mark.parametrize("store_type", [InMemoryStore, JsonlStore, SQLiteStore])
def test_paginated_sync_keeps_a_logical_batch_atomic(tmp_path, store_type) -> None:  # type: ignore[no-untyped-def]
    sender_store = (
        store_type()
        if store_type is InMemoryStore
        else store_type(
            tmp_path / ("sender.jsonl" if store_type is JsonlStore else "sender.sqlite")
        )
    )
    sender = Memory(store=sender_store, replica_id="sender")
    receiver = Memory(replica_id="receiver")
    from tests.test_atomic_batches import _batch

    sender.commit_batch(_batch())
    report = receiver.sync_from(sender, max_ops=2)

    assert report.transferred_op_ids == tuple(op.op_id for op in _batch())
    assert "claim-1" in receiver.materialize().claims_by_id
