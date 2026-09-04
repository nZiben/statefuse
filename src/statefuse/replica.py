from __future__ import annotations

from dataclasses import dataclass

from .ops import AnyOp


@dataclass(frozen=True)
class ReplicaDelta:
    sender_replica_id: str
    cursor_before: int
    cursor_after: int
    ops: tuple[AnyOp, ...] = ()
    batch_sizes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "ops", tuple(self.ops))
        sizes = tuple(self.batch_sizes) or ((len(self.ops),) if self.ops else ())
        _validate_batch_sizes(sizes, len(self.ops))
        object.__setattr__(self, "batch_sizes", sizes)
        if not self.sender_replica_id:
            raise ValueError("ReplicaDelta.sender_replica_id is required.")
        if not _is_cursor(self.cursor_before) or not _is_cursor(self.cursor_after):
            raise ValueError("ReplicaDelta cursors must be integers.")
        if self.cursor_before < 0 or self.cursor_after < self.cursor_before:
            raise ValueError("ReplicaDelta cursors must form a non-negative range.")
        if self.cursor_before == self.cursor_after and self.ops:
            raise ValueError("A zero-width replica delta cannot contain operations.")

    @property
    def batches(self) -> tuple[tuple[AnyOp, ...], ...]:
        result: list[tuple[AnyOp, ...]] = []
        offset = 0
        for size in self.batch_sizes:
            result.append(self.ops[offset : offset + size])
            offset += size
        return tuple(result)


@dataclass(frozen=True)
class ReplicaManifest:
    sender_replica_id: str
    cursor_before: int
    cursor_after: int
    entries: tuple[tuple[str, str], ...] = ()
    batch_sizes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(tuple(item) for item in self.entries))
        sizes = tuple(self.batch_sizes) or ((len(self.entries),) if self.entries else ())
        _validate_batch_sizes(sizes, len(self.entries))
        object.__setattr__(self, "batch_sizes", sizes)
        if not self.sender_replica_id:
            raise ValueError("ReplicaManifest.sender_replica_id is required.")
        if not _is_cursor(self.cursor_before) or not _is_cursor(self.cursor_after):
            raise ValueError("ReplicaManifest cursors must be integers.")
        if self.cursor_before < 0 or self.cursor_after < self.cursor_before:
            raise ValueError("ReplicaManifest cursors must form a non-negative range.")
        if any(len(item) != 2 or not item[0] or not item[1] for item in self.entries):
            raise ValueError("Replica manifest entries require operation IDs and digests.")


@dataclass(frozen=True)
class ReplicaProgress:
    peer_replica_id: str
    cursor: int = 0
    pending_ranges: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.peer_replica_id:
            raise ValueError("ReplicaProgress.peer_replica_id is required.")
        if not _is_cursor(self.cursor):
            raise ValueError("ReplicaProgress.cursor must be an integer.")
        if self.cursor < 0:
            raise ValueError("ReplicaProgress.cursor must be non-negative.")
        ranges = tuple(tuple(item) for item in self.pending_ranges)
        if any(
            len(item) != 2
            or not _is_cursor(item[0])
            or not _is_cursor(item[1])
            or item[0] < 0
            or item[1] <= item[0]
            for item in ranges
        ):
            raise ValueError("Replica progress ranges must be positive cursor intervals.")
        ranges = tuple(sorted(ranges))
        object.__setattr__(self, "pending_ranges", ranges)


@dataclass(frozen=True)
class ReplicaSyncReport:
    peer_replica_id: str
    cursor_before: int
    cursor_after: int
    transferred_op_ids: tuple[str, ...] = ()
    applied_op_ids: tuple[str, ...] = ()
    duplicate_op_ids: tuple[str, ...] = ()
    pending_ranges: tuple[tuple[int, int], ...] = ()


def advance_replica_progress(
    progress: ReplicaProgress,
    *,
    cursor_before: int,
    cursor_after: int,
) -> ReplicaProgress:
    if not _is_cursor(cursor_before) or not _is_cursor(cursor_after):
        raise ValueError("Replica progress cursors must be integers.")
    if cursor_before < 0 or cursor_after < cursor_before:
        raise ValueError("Replica progress update has an invalid cursor range.")
    if cursor_before == cursor_after:
        return progress

    ranges = [*progress.pending_ranges, (cursor_before, cursor_after)]
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    cursor = progress.cursor
    pending: list[tuple[int, int]] = []
    for start, end in merged:
        if start <= cursor:
            cursor = max(cursor, end)
        elif end > cursor:
            pending.append((start, end))
    return ReplicaProgress(
        peer_replica_id=progress.peer_replica_id,
        cursor=cursor,
        pending_ranges=tuple(pending),
    )


def _is_cursor(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_batch_sizes(sizes: tuple[int, ...], op_count: int) -> None:
    if any(not _is_cursor(size) or size < 1 for size in sizes) or sum(sizes) != op_count:
        raise ValueError("Replica batch sizes must cover every operation.")
