from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from ..utils import canonical_json_dumps, digest_json_value
from ._common import (
    metadata_hit,
    metadata_matches_filters,
    record_payload,
    translated_error,
)
from .errors import (
    AdapterConfigurationError,
    AdapterProtocolError,
    AdapterSearchError,
    AdapterWriteError,
)
from .models import ExternalWriteResult, RetrievalRecord, SearchHit, SearchRequest

_INSTALL = 'Install the Graphiti integration with:\npip install "statefuse[graphiti]"'


class AsyncGraphitiAdapter:
    """Native async adapter storing disposable projections as Graphiti episodes."""

    name = "graphiti"

    def __init__(
        self,
        client: Any | None = None,
        *,
        episode_type: Any | None = None,
        search_config_factory: Callable[[int], Any] | None = None,
        process_episodes: bool = True,
        **client_options: Any,
    ) -> None:
        if client is None:
            try:
                from graphiti_core import Graphiti
            except ImportError as error:
                raise AdapterConfigurationError(_INSTALL) from error
            try:
                client = Graphiti(**client_options)
            except Exception as error:
                raise AdapterConfigurationError(str(error)) from error
        self.client = client
        self._episode_type = episode_type
        self._search_config_factory = search_config_factory
        self.process_episodes = process_episodes

    async def aupsert(self, record: RetrievalRecord) -> ExternalWriteResult:
        try:
            existing = await self._find(record.projection_id, record.namespace)
            if len(existing) > 1:
                raise AdapterProtocolError(
                    f"Graphiti returned duplicate projections for {record.projection_id}."
                )
            if existing and _episode_payload(existing[0]) == record_payload(record):
                return _write_result(record, _episode_id(existing[0]), created=False)
            if existing:
                await self.client.remove_episode(_episode_id(existing[0]))
            episode = (
                await self._add_processed_episode(record)
                if self.process_episodes
                else await self._add_raw_episode(record)
            )
            external_id = _episode_id(episode)
            return _write_result(record, external_id, created=not existing)
        except Exception as error:
            raise translated_error(error, AdapterWriteError) from error

    async def asearch(self, request: SearchRequest) -> list[SearchHit]:
        try:
            results = await self.client.search_(
                request.query,
                config=self._search_config(max(request.limit * 2, 10)),
                group_ids=[_group_id(request.namespace)],
            )
            episodes = getattr(results, "episodes", None)
            scores = getattr(results, "episode_reranker_scores", ())
            if not isinstance(episodes, list):
                raise AdapterProtocolError("Graphiti search response is missing episodes.")
            hits = []
            for index, episode in enumerate(episodes):
                payload = _episode_payload(episode)
                metadata = payload.get("statefuse_metadata", {})
                if not metadata_matches_filters(metadata, request.filters):
                    continue
                hits.append(
                    metadata_hit(
                        external_id=getattr(episode, "uuid", ""),
                        text=payload.get("text", ""),
                        score=scores[index] if index < len(scores) else None,
                        metadata=payload,
                    )
                )
                if len(hits) == request.limit:
                    break
            return hits
        except Exception as error:
            raise translated_error(error, AdapterSearchError) from error

    async def adelete(self, projection_id: str, namespace: str) -> bool:
        try:
            existing = await self._find(projection_id, namespace)
            for episode in existing:
                await self.client.remove_episode(_episode_id(episode))
            return bool(existing)
        except Exception as error:
            raise translated_error(error, AdapterWriteError) from error

    async def ahealthcheck(self) -> bool:
        try:
            await self.client.nodes.episode.get_by_group_ids(
                [_group_id("statefuse-healthcheck")], limit=1
            )
        except Exception:
            return False
        return True

    async def _find(self, projection_id: str, namespace: str) -> list[Any]:
        episodes = await self.client.nodes.episode.get_by_group_ids(
            [_group_id(namespace)], limit=100
        )
        return [episode for episode in episodes if getattr(episode, "name", None) == projection_id]

    async def _add_processed_episode(self, record: RetrievalRecord) -> object:
        result = await self.client.add_episode(
            name=record.projection_id,
            episode_body=_episode_body(record),
            source_description="StateFuse retrieval projection",
            reference_time=datetime.now(timezone.utc),
            source=self._episode_source(),
            group_id=_group_id(record.namespace),
        )
        return getattr(result, "episode", None)

    async def _add_raw_episode(self, record: RetrievalRecord) -> object:
        try:
            from graphiti_core.nodes import EpisodicNode
        except ImportError as error:
            raise AdapterConfigurationError(_INSTALL) from error
        now = datetime.now(timezone.utc)
        episode = EpisodicNode(
            name=record.projection_id,
            group_id=_group_id(record.namespace),
            source=self._episode_source(),
            source_description="StateFuse retrieval projection",
            content=_episode_body(record),
            created_at=now,
            valid_at=now,
        )
        await episode.save(self.client.driver)
        return episode

    def _episode_source(self) -> Any:
        if self._episode_type is not None:
            return self._episode_type
        try:
            from graphiti_core.nodes import EpisodeType
        except ImportError as error:
            raise AdapterConfigurationError(_INSTALL) from error
        return EpisodeType.json

    def _search_config(self, limit: int) -> Any:
        if self._search_config_factory is not None:
            return self._search_config_factory(limit)
        try:
            from graphiti_core.search.search_config import (
                EpisodeReranker,
                EpisodeSearchConfig,
                EpisodeSearchMethod,
                SearchConfig,
            )
        except ImportError as error:
            raise AdapterConfigurationError(_INSTALL) from error
        return SearchConfig(
            episode_config=EpisodeSearchConfig(
                search_methods=[EpisodeSearchMethod.bm25],
                reranker=EpisodeReranker.rrf,
            ),
            limit=limit,
        )


GraphitiAdapter = AsyncGraphitiAdapter


def _episode_body(record: RetrievalRecord) -> str:
    return canonical_json_dumps(record_payload(record))


def _episode_payload(episode: object) -> dict[str, Any]:
    content = getattr(episode, "content", "")
    try:
        value = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _episode_id(episode: object) -> str:
    external_id = getattr(episode, "uuid", None)
    if not isinstance(external_id, str) or not external_id:
        raise AdapterProtocolError("Graphiti episode is missing a UUID.")
    return external_id


def _group_id(namespace: str) -> str:
    return f"statefuse-{digest_json_value(namespace)}"


def _write_result(
    record: RetrievalRecord, external_id: str, *, created: bool
) -> ExternalWriteResult:
    return ExternalWriteResult(
        AsyncGraphitiAdapter.name,
        record.projection_id,
        external_id,
        created,
        dict(record.metadata),
    )
