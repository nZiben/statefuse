from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from statefuse.integrations import (
    AdapterConfigurationError,
    AsyncGraphitiAdapter,
    LangGraphStoreAdapter,
    LettaAdapter,
    Mem0Adapter,
    RetrievalRecord,
    SearchRequest,
)

from .test_adapter_contract import AdapterContract, AsyncAdapterContract


class _UnavailableClient:
    def __init__(self) -> None:
        self.available = True

    def check(self) -> None:
        if not self.available:
            raise ConnectionError("repository unavailable")


class _Mem0Client(_UnavailableClient):
    def __init__(self) -> None:
        super().__init__()
        self.records: dict[str, dict[str, Any]] = {}
        self.next_id = 1

    def add(self, messages, *, user_id, metadata, infer):  # type: ignore[no-untyped-def]
        self.check()
        assert infer is False
        external_id = f"mem-{self.next_id}"
        self.next_id += 1
        self.records[external_id] = {
            "id": external_id,
            "memory": messages[0]["content"],
            "user_id": user_id,
            "metadata": dict(metadata),
            "score": 1.0,
        }
        return {"results": [{"id": external_id}]}

    def get_all(self, *, filters, top_k):  # type: ignore[no-untyped-def]
        self.check()
        return {"results": self._matching(filters)[:top_k]}

    def search(self, query, *, top_k, filters):  # type: ignore[no-untyped-def]
        self.check()
        terms = _terms(query)
        results = [
            item
            for item in self._matching(filters)
            if not terms or terms & _terms(item["memory"])
        ]
        return {"results": results[:top_k]}

    def update(self, memory_id, *, text, metadata):  # type: ignore[no-untyped-def]
        self.check()
        self.records[memory_id]["memory"] = text
        self.records[memory_id]["metadata"].update(metadata)

    def delete(self, memory_id):  # type: ignore[no-untyped-def]
        self.check()
        del self.records[memory_id]

    def _matching(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            item
            for item in self.records.values()
            if all(_mem0_filter(item, key) == value for key, value in filters.items())
        ]


class TestMem0AdapterContract(AdapterContract):
    def make_adapter(self) -> Mem0Adapter:
        return Mem0Adapter(_Mem0Client())

    def make_unavailable(self, adapter: Mem0Adapter) -> None:
        adapter.client.available = False

    def make_available(self, adapter: Mem0Adapter) -> None:
        adapter.client.available = True


def test_mem0_update_clears_removed_filter_values() -> None:
    adapter = Mem0Adapter(_Mem0Client())
    first = RetrievalRecord("statefuse:claim:cl_1", "Alpha", "project", metadata={"v": 1})
    second = RetrievalRecord("statefuse:claim:cl_1", "Alpha", "project", metadata={"v": 2})
    adapter.upsert(first)
    adapter.upsert(second)

    assert adapter.search(SearchRequest("Alpha", "project", filters={"v": 1})) == []
    assert len(adapter.search(SearchRequest("Alpha", "project", filters={"v": 2}))) == 1


@dataclass
class _StoreItem:
    key: str
    value: dict[str, Any]
    score: float | None = None


class _Store(_UnavailableClient):
    def __init__(self) -> None:
        super().__init__()
        self.records: dict[tuple[tuple[str, ...], str], _StoreItem] = {}

    def get(self, namespace, key):  # type: ignore[no-untyped-def]
        self.check()
        return self.records.get((namespace, key))

    def put(self, namespace, key, value, *, index):  # type: ignore[no-untyped-def]
        self.check()
        assert index == ["text"]
        self.records[(namespace, key)] = _StoreItem(key, dict(value))

    def search(self, namespace, *, query, limit, filter):  # type: ignore[no-untyped-def]
        self.check()
        terms = _terms(query)
        items = []
        for (item_namespace, _), item in self.records.items():
            if item_namespace != namespace or terms and not terms & _terms(item.value["text"]):
                continue
            if filter and any(_dotted(item.value, key) != value for key, value in filter.items()):
                continue
            items.append(_StoreItem(item.key, item.value, 1.0))
        return items[:limit]

    def delete(self, namespace, key):  # type: ignore[no-untyped-def]
        self.check()
        del self.records[(namespace, key)]

    def list_namespaces(self, *, prefix, limit):  # type: ignore[no-untyped-def]
        self.check()
        matching = {
            namespace
            for namespace, _ in self.records
            if namespace[: len(prefix)] == prefix
        }
        return list(matching)[:limit]


class TestLangMemAdapterContract(AdapterContract):
    def make_adapter(self) -> LangGraphStoreAdapter:
        return LangGraphStoreAdapter(_Store())

    def make_unavailable(self, adapter: LangGraphStoreAdapter) -> None:
        adapter.store.available = False

    def make_available(self, adapter: LangGraphStoreAdapter) -> None:
        adapter.store.available = True


class _LettaClient(_UnavailableClient):
    def __init__(self) -> None:
        super().__init__()
        self.records: dict[str, SimpleNamespace] = {}
        self.next_id = 1
        self.archives = SimpleNamespace(
            passages=SimpleNamespace(create=self.create, delete=self.delete)
        )
        self.passages = SimpleNamespace(search=self.search)

    def create(self, archive_id, *, text, metadata, tags):  # type: ignore[no-untyped-def]
        self.check()
        external_id = f"passage-{self.next_id}"
        self.next_id += 1
        passage = SimpleNamespace(
            id=external_id,
            archive_id=archive_id,
            text=text,
            metadata=dict(metadata),
            tags=list(tags),
        )
        self.records[external_id] = passage
        return passage

    def delete(self, passage_id, *, archive_id):  # type: ignore[no-untyped-def]
        self.check()
        assert self.records[passage_id].archive_id == archive_id
        del self.records[passage_id]

    def search(self, *, archive_id, limit, query=None, tags=None, tag_match_mode=None):  # type: ignore[no-untyped-def]
        self.check()
        terms = _terms(query or "")
        items = []
        for passage in self.records.values():
            if passage.archive_id != archive_id or terms and not terms & _terms(passage.text):
                continue
            if tags and not set(tags) <= set(passage.tags):
                continue
            items.append(SimpleNamespace(passage=passage, score=1.0))
        return items[:limit]


class TestLettaAdapterContract(AdapterContract):
    def make_adapter(self) -> LettaAdapter:
        return LettaAdapter("archive-1", _LettaClient())

    def make_unavailable(self, adapter: LettaAdapter) -> None:
        adapter.client.available = False

    def make_available(self, adapter: LettaAdapter) -> None:
        adapter.client.available = True


class _CappedLettaClient(_LettaClient):
    """Reproduce Letta 0.16's limit-before-tag-filter behavior."""

    def search(self, *, archive_id, limit, query=None, tags=None, tag_match_mode=None):  # type: ignore[no-untyped-def]
        self.check()
        candidates = [
            passage
            for passage in self.records.values()
            if passage.archive_id == archive_id
        ][:limit]
        terms = _terms(query or "")
        return [
            SimpleNamespace(passage=passage, score=1.0)
            for passage in candidates
            if (not terms or terms & _terms(passage.text))
            and (not tags or set(tags) <= set(passage.tags))
        ]


def test_letta_upsert_remains_idempotent_after_server_candidate_cap() -> None:
    client = _CappedLettaClient()
    adapter = LettaAdapter("archive-1", client)
    record = RetrievalRecord("statefuse:claim:cl_1", "Alpha", "project")
    for index in range(100):
        client.create(
            "archive-1",
            text=f"noise {index}",
            metadata={},
            tags=[],
        )
    created = adapter.upsert(record)

    repeated = adapter.upsert(record)

    assert repeated.created is False
    assert repeated.external_id == created.external_id


class NodeNotFoundError(Exception):
    pass


class _GraphitiClient(_UnavailableClient):
    def __init__(self) -> None:
        super().__init__()
        self.records: dict[str, SimpleNamespace] = {}
        self.nodes = SimpleNamespace(
            episode=SimpleNamespace(
                get_by_uuid=self.get_by_uuid,
                get_by_group_ids=self.get_by_group_ids,
            )
        )

    async def get_by_uuid(self, external_id):  # type: ignore[no-untyped-def]
        self.check()
        if external_id not in self.records:
            raise NodeNotFoundError(external_id)
        return self.records[external_id]

    async def get_by_group_ids(self, group_ids, *, limit):  # type: ignore[no-untyped-def]
        self.check()
        return [item for item in self.records.values() if item.group_id in group_ids][:limit]

    async def add_episode(self, **values):  # type: ignore[no-untyped-def]
        self.check()
        episode = SimpleNamespace(
            uuid=f"episode-{len(self.records) + 1}",
            name=values["name"],
            content=values["episode_body"],
            group_id=values["group_id"],
        )
        self.records[episode.uuid] = episode
        return SimpleNamespace(episode=episode)

    async def remove_episode(self, external_id):  # type: ignore[no-untyped-def]
        self.check()
        del self.records[external_id]

    async def search_(self, query, *, config, group_ids):  # type: ignore[no-untyped-def]
        self.check()
        terms = _terms(query)
        episodes = [
            item
            for item in self.records.values()
            if item.group_id in group_ids
            and (not terms or terms & _terms(json.loads(item.content)["text"]))
        ][: config.limit]
        return SimpleNamespace(
            episodes=episodes,
            episode_reranker_scores=[1.0] * len(episodes),
        )


class TestGraphitiAdapterContract(AsyncAdapterContract):
    def make_adapter(self) -> AsyncGraphitiAdapter:
        return AsyncGraphitiAdapter(
            _GraphitiClient(),
            episode_type="json",
            search_config_factory=lambda limit: SimpleNamespace(limit=limit),
        )

    def make_unavailable(self, adapter: AsyncGraphitiAdapter) -> None:
        adapter.client.available = False

    def make_available(self, adapter: AsyncGraphitiAdapter) -> None:
        adapter.client.available = True


def _terms(value: str) -> set[str]:
    return set(re.findall(r"\w+", value.casefold()))


def _dotted(value: dict[str, Any], key: str) -> object:
    current: object = value
    for part in key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _mem0_filter(item: dict[str, Any], key: str) -> object:
    if key == "user_id":
        return item["user_id"]
    return _dotted(item["metadata"], key)


@pytest.mark.parametrize(
    ("module", "factory", "extra"),
    [
        ("mem0", lambda: Mem0Adapter(), "mem0"),
        ("langgraph", lambda: LangGraphStoreAdapter(), "langmem"),
        ("letta_client", lambda: LettaAdapter("archive-1"), "letta"),
        ("graphiti_core", lambda: AsyncGraphitiAdapter(uri="bolt://unused"), "graphiti"),
    ],
)
def test_missing_connector_dependency_has_install_message(module, factory, extra) -> None:  # type: ignore[no-untyped-def]
    if importlib.util.find_spec(module) is not None:
        pytest.skip(f"{module} is installed")
    with pytest.raises(AdapterConfigurationError, match=rf"statefuse\[{extra}\]"):
        factory()
