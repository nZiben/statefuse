from __future__ import annotations

import pytest

from statefuse.integrations import (
    AdapterSearchError,
    FakeMemoryRepositoryAdapter,
    RetrievalRecord,
    SearchRequest,
)


def test_lexical_search_is_deterministic_and_supports_filters() -> None:
    adapter = FakeMemoryRepositoryAdapter()
    for projection_id, text, topic in (
        ("statefuse:claim:b", "Alpha deadline May", "schedule"),
        ("statefuse:claim:a", "Alpha launch", "launch"),
    ):
        adapter.upsert(
            RetrievalRecord(
                projection_id=projection_id,
                text=text,
                namespace="project",
                metadata={"topic": topic},
            )
        )

    assert [hit.projection_id for hit in adapter.search(SearchRequest("Alpha", "project"))] == [
        "statefuse:claim:a",
        "statefuse:claim:b",
    ]
    assert adapter.search(
        SearchRequest("Alpha", "project", filters={"topic": "schedule"})
    )[0].projection_id == "statefuse:claim:b"


def test_injected_failure_is_one_shot_and_explicit() -> None:
    adapter = FakeMemoryRepositoryAdapter()
    adapter.inject_failure("search", AdapterSearchError("search failed"))

    with pytest.raises(AdapterSearchError, match="search failed"):
        adapter.search(SearchRequest("anything", "project"))

    assert adapter.search(SearchRequest("anything", "project")) == []


def test_duplicate_write_is_detected_without_creating_another_record() -> None:
    adapter = FakeMemoryRepositoryAdapter()
    record = RetrievalRecord("statefuse:claim:cl_1", "Alpha deadline", "project")

    adapter.upsert(record)
    adapter.upsert(record)

    assert adapter.record_count == 1
    assert adapter.duplicate_write_count == 1
