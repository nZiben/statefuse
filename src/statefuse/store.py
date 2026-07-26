from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from .oplog import OpLog
from .ops import AnyOp, Op


class OpStore(Protocol):
    def append(self, op: AnyOp) -> bool:
        ...

    def iter_ops(self) -> Iterator[AnyOp]:
        ...

    def has(self, op_id: str) -> bool:
        ...

    def load_oplog(self) -> OpLog:
        ...


class InMemoryStore:
    def __init__(self) -> None:
        self._oplog = OpLog()

    def append(self, op: AnyOp) -> bool:
        return self._oplog.add(op)

    def iter_ops(self) -> Iterator[AnyOp]:
        return iter(self._oplog.iter_ops())

    def has(self, op_id: str) -> bool:
        return self._oplog.has(op_id)

    def load_oplog(self) -> OpLog:
        return self._oplog.copy()


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
