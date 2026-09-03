from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import Protocol, runtime_checkable
from uuid import uuid4

from .oplog import OpLog
from .ops import AnyOp, Op
from .replica import ReplicaProgress


class OpStore(Protocol):
    def append(self, op: AnyOp) -> bool: ...

    def iter_ops(self) -> Iterator[AnyOp]: ...

    def has(self, op_id: str) -> bool: ...

    def load_oplog(self) -> OpLog: ...


@dataclass(frozen=True)
class StoreDelta:
    cursor: int
    ops: tuple[AnyOp, ...]
    batch_sizes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "ops", tuple(self.ops))
        sizes = tuple(self.batch_sizes) or (1,) * len(self.ops)
        if any(size < 1 for size in sizes) or sum(sizes) != len(self.ops):
            raise ValueError("Store delta batch sizes must cover every operation.")
        object.__setattr__(self, "batch_sizes", sizes)

    @property
    def batches(self) -> tuple[tuple[AnyOp, ...], ...]:
        result: list[tuple[AnyOp, ...]] = []
        offset = 0
        for size in self.batch_sizes:
            result.append(self.ops[offset : offset + size])
            offset += size
        return tuple(result)


@dataclass(frozen=True)
class MaterializationCheckpoint:
    version: str
    config_token: str
    cursor: int
    payload: str


@runtime_checkable
class IncrementalOpStore(Protocol):
    """Optional local cursor/checkpoint capability for incremental materialization."""

    def read_after(self, cursor: int) -> StoreDelta: ...

    def load_materialization_checkpoint(self) -> MaterializationCheckpoint | None: ...

    def save_materialization_checkpoint(self, checkpoint: MaterializationCheckpoint) -> None: ...


@runtime_checkable
class BatchOpStore(Protocol):
    """Optional multi-append capability.

    Implementations expose ``batch_atomicity`` as ``atomic`` or ``prefix-visible``.
    """

    batch_atomicity: str

    def append_many(self, *ops: AnyOp) -> tuple[bool, ...]: ...


@runtime_checkable
class ReplicaProgressStore(Protocol):
    def load_replica_progress(self, peer_replica_id: str) -> ReplicaProgress: ...

    def save_replica_progress(self, progress: ReplicaProgress) -> None: ...

    def reset_replica_progress(self, peer_replica_id: str) -> None: ...


class InMemoryStore:
    batch_atomicity = "atomic"

    def __init__(self) -> None:
        self._oplog = OpLog()
        self._append_order: list[AnyOp] = []
        self._batch_ends: list[int] = []
        self._checkpoint: MaterializationCheckpoint | None = None
        self._lock = RLock()
        self._replica_progress: dict[str, ReplicaProgress] = {}

    def append(self, op: AnyOp) -> bool:
        with self._lock:
            added = self._oplog.add(op)
            if added:
                self._append_order.append(op)
                self._batch_ends.append(len(self._append_order))
            return added

    def append_many(self, *ops: AnyOp) -> tuple[bool, ...]:
        with self._lock:
            results = self._oplog.add_many(*ops)
            added = [op for op, was_added in zip(ops, results, strict=True) if was_added]
            self._append_order.extend(added)
            if added:
                self._batch_ends.append(len(self._append_order))
            return results

    def iter_ops(self) -> Iterator[AnyOp]:
        with self._lock:
            return iter(self._oplog.iter_ops())

    def has(self, op_id: str) -> bool:
        with self._lock:
            return self._oplog.has(op_id)

    def find_ops(self, op_ids: tuple[str, ...]) -> dict[str, AnyOp]:
        with self._lock:
            return {op_id: op for op_id in op_ids if (op := self._oplog.get(op_id)) is not None}

    def load_oplog(self) -> OpLog:
        with self._lock:
            return self._oplog.copy()

    def read_after(self, cursor: int, *, limit: int | None = None) -> StoreDelta:
        with self._lock:
            if cursor < 0 or cursor > len(self._append_order):
                raise ValueError("Invalid in-memory operation cursor.")
            _validate_limit(limit)
            previous = 0
            sizes: list[int] = []
            cursor_after = cursor
            for end in self._batch_ends:
                if previous < cursor < end:
                    raise ValueError("In-memory operation cursor splits a committed batch.")
                if end > cursor:
                    sizes.append(end - previous)
                    cursor_after = end
                    if limit is not None and sum(sizes) >= limit:
                        break
                previous = end
            return StoreDelta(
                cursor=cursor_after,
                ops=tuple(self._append_order[cursor:cursor_after]),
                batch_sizes=tuple(sizes),
            )

    def load_materialization_checkpoint(self) -> MaterializationCheckpoint | None:
        with self._lock:
            return self._checkpoint

    def save_materialization_checkpoint(self, checkpoint: MaterializationCheckpoint) -> None:
        with self._lock:
            self._checkpoint = checkpoint

    def load_replica_progress(self, peer_replica_id: str) -> ReplicaProgress:
        with self._lock:
            return self._replica_progress.get(
                peer_replica_id,
                ReplicaProgress(peer_replica_id=peer_replica_id),
            )

    def save_replica_progress(self, progress: ReplicaProgress) -> None:
        with self._lock:
            self._replica_progress[progress.peer_replica_id] = progress

    def reset_replica_progress(self, peer_replica_id: str) -> None:
        with self._lock:
            self._replica_progress.pop(peer_replica_id, None)


class JsonlStore:
    """JSONL store whose batches are contiguous but may expose a prefix to other instances."""

    batch_atomicity = "prefix-visible"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def append(self, op: AnyOp) -> bool:
        return self.append_many(op)[0]

    def append_many(self, *ops: AnyOp) -> tuple[bool, ...]:
        with self._lock:
            existing = {op.op_id: op for op in self.iter_ops()}
            results: list[bool] = []
            added: list[AnyOp] = []
            for op in ops:
                current = existing.get(op.op_id)
                if current is not None:
                    if current != op:
                        raise ValueError(f"op_id collision with different payload: {op.op_id}")
                    results.append(False)
                    continue
                existing[op.op_id] = op
                added.append(op)
                results.append(True)
            if added:
                self._ensure_line_boundary()
                with self.path.open("a", encoding="utf-8") as file:
                    file.write(
                        json.dumps(
                            {
                                "statefuse_batch": 1,
                                "ops": [op.to_dict() for op in added],
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
            return tuple(results)

    def iter_ops(self) -> Iterator[AnyOp]:
        with self._lock:
            if not self.path.exists():
                return iter(())
            seen: dict[str, AnyOp] = {}
            with self.path.open("r", encoding="utf-8") as file:
                for raw_line in file:
                    payload = raw_line.strip()
                    if not payload:
                        continue
                    try:
                        batch = _decode_jsonl_batch(payload)
                    except Exception:
                        continue
                    for op in batch:
                        existing = seen.get(op.op_id)
                        if existing is not None:
                            if existing != op:
                                raise ValueError(
                                    f"op_id collision with different payload: {op.op_id}"
                                )
                            continue
                        seen[op.op_id] = op
            return iter(seen[op_id] for op_id in sorted(seen))

    def has(self, op_id: str) -> bool:
        return self._find_op(op_id) is not None

    def find_ops(self, op_ids: tuple[str, ...]) -> dict[str, AnyOp]:
        candidates = set(op_ids)
        if not candidates:
            return {}
        return {op.op_id: op for op in self.iter_ops() if op.op_id in candidates}

    def load_oplog(self) -> OpLog:
        return OpLog(self.iter_ops())

    def read_after(self, cursor: int, *, limit: int | None = None) -> StoreDelta:
        with self._lock:
            if cursor < 0:
                raise ValueError("Invalid JSONL operation cursor.")
            _validate_limit(limit)
            if not self.path.exists():
                if cursor != 0:
                    raise ValueError("JSONL operation cursor is beyond the current file.")
                return StoreDelta(cursor=0, ops=())
            size = self.path.stat().st_size
            if cursor > size:
                raise ValueError("JSONL operation cursor is beyond the current file.")
            ops: list[AnyOp] = []
            batch_sizes: list[int] = []
            with self.path.open("rb") as file:
                file.seek(cursor)
                next_cursor = cursor
                while raw_line := file.readline():
                    if not raw_line.endswith(b"\n"):
                        break
                    next_cursor = file.tell()
                    payload = raw_line.strip()
                    if not payload:
                        continue
                    try:
                        batch = _decode_jsonl_batch(payload.decode("utf-8"))
                    except Exception:
                        continue
                    ops.extend(batch)
                    batch_sizes.append(len(batch))
                    if limit is not None and len(ops) >= limit:
                        break
            return StoreDelta(
                cursor=next_cursor,
                ops=tuple(ops),
                batch_sizes=tuple(batch_sizes),
            )

    def load_materialization_checkpoint(self) -> MaterializationCheckpoint | None:
        checkpoint_path = self._checkpoint_path()
        if not checkpoint_path.exists():
            return None
        try:
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            return MaterializationCheckpoint(**payload)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def save_materialization_checkpoint(self, checkpoint: MaterializationCheckpoint) -> None:
        checkpoint_path = self._checkpoint_path()
        temporary_path = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp")
        temporary_path.write_text(
            json.dumps(asdict(checkpoint), sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary_path.replace(checkpoint_path)

    def load_replica_progress(self, peer_replica_id: str) -> ReplicaProgress:
        with self._lock:
            payload = self._load_replica_progress_payload().get(peer_replica_id)
            if not isinstance(payload, dict):
                return ReplicaProgress(peer_replica_id=peer_replica_id)
            try:
                return ReplicaProgress(
                    peer_replica_id=peer_replica_id,
                    cursor=int(payload.get("cursor", 0)),
                    pending_ranges=tuple(tuple(item) for item in payload.get("pending_ranges", [])),
                )
            except (TypeError, ValueError):
                return ReplicaProgress(peer_replica_id=peer_replica_id)

    def save_replica_progress(self, progress: ReplicaProgress) -> None:
        with self._lock:
            payload = self._load_replica_progress_payload()
            payload[progress.peer_replica_id] = {
                "cursor": progress.cursor,
                "pending_ranges": [list(item) for item in progress.pending_ranges],
            }
            self._save_replica_progress_payload(payload)

    def reset_replica_progress(self, peer_replica_id: str) -> None:
        with self._lock:
            payload = self._load_replica_progress_payload()
            if payload.pop(peer_replica_id, None) is not None:
                self._save_replica_progress_payload(payload)

    def _checkpoint_path(self) -> Path:
        return self.path.with_name(f"{self.path.name}.checkpoint.json")

    def _ensure_line_boundary(self) -> None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        with self.path.open("rb") as file:
            file.seek(-1, 2)
            at_boundary = file.read(1) == b"\n"
        if not at_boundary:
            with self.path.open("ab") as file:
                file.write(b"\n")

    def _replica_progress_path(self) -> Path:
        return self.path.with_name(f"{self.path.name}.replica-progress.json")

    def _load_replica_progress_payload(self) -> dict[str, object]:
        path = self._replica_progress_path()
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _save_replica_progress_payload(self, payload: dict[str, object]) -> None:
        path = self._replica_progress_path()
        temporary_path = path.with_name(f"{path.name}.tmp")
        temporary_path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary_path.replace(path)

    def _find_op(self, op_id: str) -> AnyOp | None:
        for op in self.iter_ops():
            if op.op_id == op_id:
                return op
        return None


class SQLiteStore:
    batch_atomicity = "atomic"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def append(self, op: AnyOp) -> bool:
        return self.append_many(op)[0]

    def append_many(self, *ops: AnyOp) -> tuple[bool, ...]:
        results: list[bool] = []
        batch_id = uuid4().hex
        with self._connect() as conn:
            for position, op in enumerate(ops):
                payload = op.to_json()
                try:
                    conn.execute(
                        "INSERT INTO ops(op_id, op_type, ts, payload, batch_id, batch_position) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            op.op_id,
                            op.op_type,
                            op.timestamp,
                            payload,
                            batch_id,
                            position,
                        ),
                    )
                    results.append(True)
                except sqlite3.IntegrityError:
                    row = conn.execute(
                        "SELECT payload FROM ops WHERE op_id = ?", (op.op_id,)
                    ).fetchone()
                    if row and row[0] == payload:
                        results.append(False)
                        continue
                    if row is None:
                        raise
                    raise ValueError(
                        f"op_id collision with different payload: {op.op_id}"
                    ) from None
        return tuple(results)

    def iter_ops(self) -> Iterator[AnyOp]:
        with self._connect() as conn:
            rows = conn.execute("SELECT payload FROM ops ORDER BY ts, op_id").fetchall()
        return iter(Op.from_json(row[0]) for row in rows)

    def has(self, op_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM ops WHERE op_id = ? LIMIT 1", (op_id,)).fetchone()
        return row is not None

    def find_ops(self, op_ids: tuple[str, ...]) -> dict[str, AnyOp]:
        if not op_ids:
            return {}
        matches: dict[str, AnyOp] = {}
        with self._connect() as conn:
            for start in range(0, len(op_ids), 900):
                chunk = op_ids[start : start + 900]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"SELECT op_id, payload FROM ops WHERE op_id IN ({placeholders})",
                    chunk,
                ).fetchall()
                matches.update({row[0]: Op.from_json(row[1]) for row in rows})
        return matches

    def load_oplog(self) -> OpLog:
        return OpLog(self.iter_ops())

    def read_after(self, cursor: int, *, limit: int | None = None) -> StoreDelta:
        if cursor < 0:
            raise ValueError("Invalid SQLite operation cursor.")
        _validate_limit(limit)
        with self._connect() as conn:
            maximum = conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM ops").fetchone()[0]
            if cursor > maximum:
                raise ValueError("SQLite operation cursor is beyond the current log.")
            rows = conn.execute(
                "SELECT rowid, payload, batch_id FROM ops WHERE rowid > ? ORDER BY rowid",
                (cursor,),
            ).fetchall()
        selected: list[tuple[int, str, str | None]] = []
        batch_sizes: list[int] = []
        previous_batch: str | None = None
        for row in rows:
            batch = row[2] or f"legacy:{row[0]}"
            if batch != previous_batch:
                if limit is not None and len(selected) >= limit:
                    break
                batch_sizes.append(0)
                previous_batch = batch
            selected.append(row)
            batch_sizes[-1] += 1
        return StoreDelta(
            cursor=selected[-1][0] if selected else cursor,
            ops=tuple(Op.from_json(row[1]) for row in selected),
            batch_sizes=tuple(batch_sizes),
        )

    def load_materialization_checkpoint(self) -> MaterializationCheckpoint | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT version, config_token, cursor, payload "
                "FROM materialization_checkpoint WHERE id = 1"
            ).fetchone()
        if row is None:
            return None
        return MaterializationCheckpoint(
            version=row[0], config_token=row[1], cursor=row[2], payload=row[3]
        )

    def save_materialization_checkpoint(self, checkpoint: MaterializationCheckpoint) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO materialization_checkpoint(
                    id, version, config_token, cursor, payload
                ) VALUES (1, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    version = excluded.version,
                    config_token = excluded.config_token,
                    cursor = excluded.cursor,
                    payload = excluded.payload
                """,
                (
                    checkpoint.version,
                    checkpoint.config_token,
                    checkpoint.cursor,
                    checkpoint.payload,
                ),
            )

    def load_replica_progress(self, peer_replica_id: str) -> ReplicaProgress:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT cursor, pending_ranges FROM replica_progress WHERE peer_replica_id = ?",
                (peer_replica_id,),
            ).fetchone()
        if row is None:
            return ReplicaProgress(peer_replica_id=peer_replica_id)
        try:
            ranges = json.loads(row[1])
            return ReplicaProgress(
                peer_replica_id=peer_replica_id,
                cursor=row[0],
                pending_ranges=tuple(tuple(item) for item in ranges),
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return ReplicaProgress(peer_replica_id=peer_replica_id)

    def save_replica_progress(self, progress: ReplicaProgress) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO replica_progress(peer_replica_id, cursor, pending_ranges)
                VALUES (?, ?, ?)
                ON CONFLICT(peer_replica_id) DO UPDATE SET
                    cursor = excluded.cursor,
                    pending_ranges = excluded.pending_ranges
                """,
                (
                    progress.peer_replica_id,
                    progress.cursor,
                    json.dumps(progress.pending_ranges, separators=(",", ":")),
                ),
            )

    def reset_replica_progress(self, peer_replica_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM replica_progress WHERE peer_replica_id = ?",
                (peer_replica_id,),
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ops (
                    op_id TEXT PRIMARY KEY,
                    op_type TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    batch_id TEXT,
                    batch_position INTEGER
                )
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(ops)")}
            if "batch_id" not in columns:
                conn.execute("ALTER TABLE ops ADD COLUMN batch_id TEXT")
            if "batch_position" not in columns:
                conn.execute("ALTER TABLE ops ADD COLUMN batch_position INTEGER")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ops_ts ON ops(ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ops_type ON ops(op_type)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS materialization_checkpoint (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    version TEXT NOT NULL,
                    config_token TEXT NOT NULL,
                    cursor INTEGER NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS replica_progress (
                    peer_replica_id TEXT PRIMARY KEY,
                    cursor INTEGER NOT NULL,
                    pending_ranges TEXT NOT NULL
                )
                """
            )


def _validate_limit(limit: int | None) -> None:
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise ValueError("Operation read limit must be positive.")


def _decode_jsonl_batch(payload: str) -> tuple[AnyOp, ...]:
    value = json.loads(payload)
    if isinstance(value, dict) and value.get("statefuse_batch") == 1:
        raw_ops = value.get("ops")
        if not isinstance(raw_ops, list) or not raw_ops:
            raise ValueError("A JSONL batch must contain operations.")
        return tuple(Op.from_dict(item) for item in raw_ops)
    if not isinstance(value, dict):
        raise ValueError("A JSONL operation must be an object.")
    return (Op.from_dict(value),)
