# Replica delta sync

`Memory.sync_from(peer)` first exchanges operation IDs and content digests after the receiver's
durable cursor, then pulls only missing payloads. It applies those operations idempotently, updates
materialized state, and only then saves receive progress.

```python
report = receiver.sync_from(sender, max_ops=500)

print(report.transferred_op_ids)
print(report.cursor_after)
```

`max_ops` is a soft page size: a committed logical batch is never split, so one page may exceed the
limit. Call `sync_from` again while the report advances the cursor to drain a larger suffix. A
repeated call at the end of the stream transfers no operations.

## Delivery and recovery

- Applying the same delta more than once is safe. Identical operation IDs are duplicates; the same
  ID with a different payload remains an error.
- Out-of-order deltas are accepted. The receiver stores pending cursor ranges and advances its
  contiguous cursor when the gap arrives.
- A write failure cannot advance receive progress. Some operations may already be durable, but a
  retry recognizes that prefix as duplicates and completes the delta.
- Batch boundaries survive export and receive. Atomic stores never expose part of a committed
  logical event merely because a sync page boundary falls inside it.
- Progress is independent for each peer. `JsonlStore` uses an atomic sidecar file and `SQLiteStore`
  uses a dedicated table; `InMemoryStore` keeps progress for the process lifetime.

The peer's `replica_id` identifies one append-only stream. Do not reuse it for an unrelated log.
Use SQLite for concurrent writers; JSONL is a single-writer store.

## Compaction or log replacement

Byte and row cursors are stable only while the sender log remains append-only. If a peer compacts,
truncates, or replaces its log, call `reset_replica_progress(peer_id)` before delta sync. Use
`merge_from(peer.store)` as the full-log bootstrap and repair path when cursor continuity is not
known. The existing full CRDT merge API remains unchanged.
