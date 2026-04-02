from __future__ import annotations

from typing import Iterable, Iterator

from .ops import AnyOp


class OpLog:
    """Set-like op-log keyed by immutable op_id."""

    def __init__(self, ops: Iterable[AnyOp] | None = None) -> None:
        self._ops: dict[str, AnyOp] = {}
        if ops is None:
            return
        for op in ops:
            self.add(op)

    def add(self, op: AnyOp) -> bool:
        existing = self._ops.get(op.op_id)
        if existing is None:
            self._ops[op.op_id] = op
            return True
        if existing != op:
            raise ValueError(f"op_id collision with different payload: {op.op_id}")
        return False

    def has(self, op_id: str) -> bool:
        return op_id in self._ops

    def get(self, op_id: str) -> AnyOp | None:
        return self._ops.get(op_id)

    def iter_ops(self) -> list[AnyOp]:
        return [self._ops[op_id] for op_id in sorted(self._ops)]

    def op_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._ops))

    def copy(self) -> "OpLog":
        return OpLog(self.iter_ops())

    def __iter__(self) -> Iterator[AnyOp]:
        return iter(self.iter_ops())

    def __len__(self) -> int:
        return len(self._ops)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, OpLog) and self._ops == other._ops

    def __repr__(self) -> str:
        return f"OpLog(ops={len(self._ops)})"
