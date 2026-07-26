from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

from .oplog import OpLog
from .ops import AnyOp, Op


def append_op_to_graph_state(
    graph_state: MutableMapping[str, Any],
    op: AnyOp,
    *,
    key: str = "statefuse_ops",
) -> None:
    raw_ops = list(graph_state.get(key, []))
    raw_ops.append(op.to_dict())
    graph_state[key] = raw_ops


def oplog_from_graph_state(graph_state: Mapping[str, Any], *, key: str = "statefuse_ops") -> OpLog:
    raw_ops = graph_state.get(key, [])
    oplog = OpLog()
    if not isinstance(raw_ops, list):
        return oplog
    for raw in raw_ops:
        if isinstance(raw, str):
            op = Op.from_json(raw)
        elif isinstance(raw, dict):
            op = Op.from_dict(raw)
        else:
            continue
        oplog.add(op)
    return oplog


def merged_graph_state_ops(
    left_state: Mapping[str, Any],
    right_state: Mapping[str, Any],
    *,
    key: str = "statefuse_ops",
) -> list[dict[str, Any]]:
    merged = OpLog()
    for oplog in (
        oplog_from_graph_state(left_state, key=key),
        oplog_from_graph_state(right_state, key=key),
    ):
        for op in oplog.iter_ops():
            merged.add(op)
    return [op.to_dict() for op in merged.iter_ops()]
