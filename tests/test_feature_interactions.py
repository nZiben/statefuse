from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from statefuse import Claim, ClaimAdded, ClaimKey, Memory, SQLiteStore, materialize
from statefuse.materialize import apply_operations_incrementally

KEY = ClaimKey("interaction", "shared-item", "status")


def _claim_op(index: int, value: str) -> ClaimAdded:
    timestamp = f"2026-03-01T00:00:0{index}.000000Z"
    return ClaimAdded(
        op_id=f"op-{index}",
        replica_id="sender",
        timestamp=timestamp,
        claim=Claim(
            claim_id=f"claim-{index}",
            key=KEY,
            value=value,
            confidence=0.8,
            timestamp=timestamp,
            evidence_ids=(),
            provenance={"replica_id": "sender"},
        ),
    )


def test_atomic_replica_batch_updates_conflict_index_once() -> None:
    sender = Memory(replica_id="sender")
    receiver = Memory(replica_id="receiver")
    ops = (_claim_op(1, "one"), _claim_op(2, "two"), _claim_op(3, "three"))
    sender.commit_batch(ops)
    receiver.materialize()
    cursor = receiver.materialization_cursor
    assert cursor == 0

    with patch(
        "statefuse.memory.apply_operations_incrementally",
        wraps=apply_operations_incrementally,
    ) as apply:
        report = receiver.sync_from(sender)

    state, delta = receiver.materialize_with_delta(cursor)
    assert apply.call_count == 1
    assert apply.call_args.args[1] == ops
    assert report.applied_op_ids == tuple(op.op_id for op in ops)
    assert delta.conflict_comparisons == 3
    assert state == materialize(receiver.load_oplog())
    assert len(state.conflicts) == 1


def test_synced_conflict_index_and_progress_restore_together(tmp_path: Path) -> None:
    sender = Memory(replica_id="sender")
    path = tmp_path / "receiver.sqlite"
    receiver = Memory(SQLiteStore(path), replica_id="receiver")
    sender.commit_batch((_claim_op(1, "one"), _claim_op(2, "two")))
    receiver.sync_from(sender)

    restarted = Memory(SQLiteStore(path), replica_id="receiver")
    restored = restarted.materialize()
    assert restored == materialize(restarted.load_oplog())
    assert restarted.replica_progress("sender").cursor == 2

    sender.commit_batch((_claim_op(3, "three"),))
    cursor = restarted.materialization_cursor
    restarted.sync_from(sender)
    updated, delta = restarted.materialize_with_delta(cursor)

    assert updated == materialize(restarted.load_oplog())
    assert delta.conflict_comparisons == 2
    assert restarted.replica_progress("sender").cursor == 3


def test_direct_replica_delta_uses_sqlite_batch_rollback(tmp_path: Path) -> None:
    sender = Memory(replica_id="sender")
    first = _claim_op(1, "one")
    second = _claim_op(2, "two")
    sender.commit_batch((first, second))
    receiver = Memory(SQLiteStore(tmp_path / "receiver.sqlite"), replica_id="receiver")
    receiver.append_op(
        ClaimAdded(
            op_id=second.op_id,
            replica_id="other",
            timestamp=second.timestamp,
            claim=Claim(
                claim_id="colliding-claim",
                key=KEY,
                value="different",
                confidence=0.8,
                timestamp=second.timestamp,
                evidence_ids=(),
                provenance={"replica_id": "other"},
            ),
        )
    )

    with pytest.raises(ValueError, match="collision with different payload"):
        receiver.apply_replica_delta(sender.export_replica_delta(0))

    assert receiver.load_oplog().op_ids() == (second.op_id,)
    assert receiver.replica_progress("sender").cursor == 0
