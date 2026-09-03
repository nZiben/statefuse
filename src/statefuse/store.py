from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from .oplog import OpLog
from .ops import AnyOp, Op


class OpStore(Protocol):
    def append(self, op: AnyOp) -> bool: ...

    def iter_ops(self) -> Iterator[AnyOp]: ...

    def has(self, op_id: str) -> bool: ...

    def load_oplog(self) -> OpLog: ...


@dataclass(frozen=True)
class StoreDelta:
    cursor: int
    ops: tuple[AnyOp, ...]


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


class InMemoryStore:
    def __init__(self) -> None:
        self._oplog = OpLog()
        self._append_order: list[AnyOp] = []
        self._checkpoint: MaterializationCheckpoint | None = None

    def append(self, op: AnyOp) -> bool:
        added = self._oplog.add(op)
        if added:
            self._append_order.append(op)
        return added

    def iter_ops(self) -> Iterator[AnyOp]:
        return iter(self._oplog.iter_ops())

    def has(self, op_id: str) -> bool:
        return self._oplog.has(op_id)

    def load_oplog(self) -> OpLog:
        return self._oplog.copy()

    def read_after(self, cursor: int) -> StoreDelta:
        if cursor < 0 or cursor > len(self._append_order):
            raise ValueError("Invalid in-memory operation cursor.")
        return StoreDelta(
            cursor=len(self._append_order),
            ops=tuple(self._append_order[cursor:]),
        )

    def load_materialization_checkpoint(self) -> MaterializationCheckpoint | None:
        return self._checkpoint

    def save_materialization_checkpoint(self, checkpoint: MaterializationCheckpoint) -> None:
        self._checkpoint = checkpoint


class JsonlStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, op: AnyOp) -> bool:
        existing = self._find_op(op.op_id)
        if existing is not None:
            if existing != op:
                raise ValueError(f"op_id collision with different payload: {op.op_id}")
            return False
        with self.path.open("a", encoding="utf-8") as file:
            file.write(op.to_json())
            file.write("\n")
        return True

    def iter_ops(self) -> Iterator[AnyOp]:
        if not self.path.exists():
            return iter(())
        seen: dict[str, AnyOp] = {}
        with self.path.open("r", encoding="utf-8") as file:
            for raw_line in file:
                payload = raw_line.strip()
                if not payload:
                    continue
                try:
                    op = Op.from_json(payload)
                except Exception:
                    continue
                existing = seen.get(op.op_id)
                if existing is not None:
                    if existing != op:
                        raise ValueError(f"op_id collision with different payload: {op.op_id}")
                    continue
                seen[op.op_id] = op
        return iter(seen[op_id] for op_id in sorted(seen))

    def has(self, op_id: str) -> bool:
        return self._find_op(op_id) is not None

    def load_oplog(self) -> OpLog:
        return OpLog(self.iter_ops())

    def read_after(self, cursor: int) -> StoreDelta:
        if cursor < 0:
            raise ValueError("Invalid JSONL operation cursor.")
        if not self.path.exists():
            if cursor != 0:
                raise ValueError("JSONL operation cursor is beyond the current file.")
            return StoreDelta(cursor=0, ops=())
        size = self.path.stat().st_size
        if cursor > size:
            raise ValueError("JSONL operation cursor is beyond the current file.")
        ops: list[AnyOp] = []
        with self.path.open("rb") as file:
            file.seek(cursor)
            for raw_line in file:
                payload = raw_line.strip()
                if not payload:
                    continue
                try:
                    ops.append(Op.from_json(payload.decode("utf-8")))
                except Exception:
                    continue
            next_cursor = file.tell()
        return StoreDelta(cursor=next_cursor, ops=tuple(ops))

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

    def _checkpoint_path(self) -> Path:
        return self.path.with_name(f"{self.path.name}.checkpoint.json")

    def _find_op(self, op_id: str) -> AnyOp | None:
        for op in self.iter_ops():
            if op.op_id == op_id:
                return op
        return None


class SQLiteStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def append(self, op: AnyOp) -> bool:
        payload = op.to_json()
        with self._connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO ops(op_id, op_type, ts, payload) VALUES (?, ?, ?, ?)",
                    (op.op_id, op.op_type, op.timestamp, payload),
                )
                return True
            except sqlite3.IntegrityError:
                row = conn.execute(
                    "SELECT payload FROM ops WHERE op_id = ?", (op.op_id,)
                ).fetchone()
                if row and row[0] == payload:
                    return False
                raise ValueError(f"op_id collision with different payload: {op.op_id}") from None

    def iter_ops(self) -> Iterator[AnyOp]:
        with self._connect() as conn:
            rows = conn.execute("SELECT payload FROM ops ORDER BY ts, op_id").fetchall()
        return iter(Op.from_json(row[0]) for row in rows)

    def has(self, op_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM ops WHERE op_id = ? LIMIT 1", (op_id,)).fetchone()
        return row is not None

    def load_oplog(self) -> OpLog:
        return OpLog(self.iter_ops())

    def read_after(self, cursor: int) -> StoreDelta:
        if cursor < 0:
            raise ValueError("Invalid SQLite operation cursor.")
        with self._connect() as conn:
            maximum = conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM ops").fetchone()[0]
            if cursor > maximum:
                raise ValueError("SQLite operation cursor is beyond the current log.")
            rows = conn.execute(
                "SELECT rowid, payload FROM ops WHERE rowid > ? ORDER BY rowid",
                (cursor,),
            ).fetchall()
        return StoreDelta(
            cursor=rows[-1][0] if rows else cursor,
            ops=tuple(Op.from_json(row[1]) for row in rows),
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
                    payload TEXT NOT NULL
                )
                """
            )
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
