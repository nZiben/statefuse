from __future__ import annotations

import json
import multiprocessing as mp
import os
import signal
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

from .conflict import ConflictSet
from .materialize import MemoryState
from .resolver import Resolution, ViewConstraints
from .utils import canonical_json_dumps

SYSTEM_PROMPT = (
    "You are a merge conflict resolver for agent memory. "
    "Return JSON only with keys: chosen_claim_id, reason, confidence."
)


def _subprocess_resolve_json(
    queue,  # type: ignore[no-untyped-def]
    *,
    base_url: str | None,
    api_key: str | None,
    api_mode: str,
    timeout: float | None,
    max_retries: int | None,
    system_prompt: str,
    input_payload: Mapping[str, Any],
    model: str,
    temperature: float | None,
    seed: int | None,
) -> None:
    try:
        client = OpenAIResponsesClient(
            base_url=base_url,
            api_key=api_key,
            api_mode=api_mode,
            timeout=timeout,
            hard_timeout=None,
            max_retries=max_retries,
        )
        raw = client.resolve_json(
            system_prompt=system_prompt,
            input_payload=input_payload,
            model=model,
            temperature=temperature,
            seed=seed,
        )
        queue.put(("ok", client.last_transport_mode or "", raw))
    except Exception as exc:  # pragma: no cover - exercised via parent behavior
        queue.put(("error", type(exc).__name__, repr(exc)))


class LLMClient(Protocol):
    def resolve_json(
        self,
        *,
        system_prompt: str,
        input_payload: Mapping[str, Any],
        model: str,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> str:
        ...


@dataclass
class LLMResolver:
    client: LLMClient
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    seed: int | None = None

    def resolve(
        self,
        conflict: ConflictSet,
        constraints: ViewConstraints,
        state: MemoryState,
    ) -> Resolution:
        payload = self._build_payload(conflict, constraints, state)
        try:
            raw_response = self.client.resolve_json(
                system_prompt=SYSTEM_PROMPT,
                input_payload=payload,
                model=self.model,
                temperature=self.temperature,
                seed=self.seed,
            )
        except Exception as exc:
            return Resolution(
                chosen_claim_id=None,
                reason=f"LLM call failed; unresolved ({type(exc).__name__}).",
                metadata={
                    "client_error": repr(exc),
                    "transport_mode": getattr(self.client, "last_transport_mode", None),
                    "api_mode": getattr(self.client, "api_mode", None),
                },
            )
        parsed, parse_error = self._parse_response(raw_response, conflict)
        if parsed is None:
            return Resolution(
                chosen_claim_id=None,
                reason=f"LLM response invalid; unresolved ({parse_error}).",
                metadata={
                    "raw_response": raw_response,
                    "parse_error": parse_error,
                    "transport_mode": getattr(self.client, "last_transport_mode", None),
                    "api_mode": getattr(self.client, "api_mode", None),
                },
            )
        return Resolution(
            chosen_claim_id=parsed["chosen_claim_id"],
            reason=parsed["reason"],
            confidence=parsed.get("confidence"),
            metadata={
                "raw_response": raw_response,
                "transport_mode": getattr(self.client, "last_transport_mode", None),
                "api_mode": getattr(self.client, "api_mode", None),
            },
        )

    def _build_payload(
        self,
        conflict: ConflictSet,
        constraints: ViewConstraints,
        state: MemoryState,
    ) -> dict[str, Any]:
        candidate_payloads = []
        for claim in conflict.candidates:
            evidence_summaries: list[dict[str, Any]] = []
            for evidence_id in claim.evidence_ids:
                evidence = state.evidence_by_id.get(evidence_id)
                if evidence is None:
                    evidence_summaries.append({"evidence_id": evidence_id, "missing": True})
                    continue
                evidence_summaries.append(
                    {
                        "evidence_id": evidence.evidence_id,
                        "pointer": evidence.pointer,
                        "metadata": evidence.metadata,
                    }
                )

            candidate_payloads.append(
                {
                    "claim_id": claim.claim_id,
                    "value": claim.value,
                    "confidence": claim.confidence,
                    "timestamp": claim.timestamp,
                    "kind": claim.kind,
                    "context": claim.context,
                    "validity": claim.validity.to_dict() if claim.validity else None,
                    "evidence": evidence_summaries,
                    "provenance": claim.provenance,
                }
            )

        return {
            "conflict": {
                "conflict_id": conflict.conflict_id,
                "conflict_type": conflict.conflict_type,
                "conflict_class": conflict.conflict_class,
                "conflict_subclass": conflict.conflict_subclass,
                "key": conflict.key.to_dict(),
                "keys": [key.to_dict() for key in conflict.keys],
                "reason": conflict.reason,
                "annotations": conflict.annotations,
                "witness": conflict.witness,
            },
            "constraints": {
                "scope": constraints.scope,
                "preferred_replica_ids": list(constraints.preferred_replica_ids),
                "preferred_branch_ids": list(constraints.preferred_branch_ids),
                "metadata": constraints.metadata,
                "valid_at": constraints.valid_at,
                "context": constraints.context,
            },
            "candidates": candidate_payloads,
            "instructions": {
                "output_schema": {
                    "chosen_claim_id": "string or null",
                    "reason": "string",
                    "confidence": "number (optional)",
                }
            },
        }

    def _parse_response(
        self,
        raw_response: str,
        conflict: ConflictSet,
    ) -> tuple[dict[str, Any] | None, str | None]:
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            return None, f"invalid_json:{exc.msg}"

        if not isinstance(payload, dict):
            return None, "response_not_object"

        chosen_claim_id = payload.get("chosen_claim_id")
        reason = payload.get("reason")
        confidence = payload.get("confidence")

        if chosen_claim_id is not None and not isinstance(chosen_claim_id, str):
            return None, "chosen_claim_id_not_string_or_null"
        if not isinstance(reason, str) or not reason.strip():
            return None, "missing_or_empty_reason"
        if confidence is not None and not isinstance(confidence, (int, float)):
            return None, "confidence_not_number"

        valid_claim_ids = {claim.claim_id for claim in conflict.candidates}
        if chosen_claim_id is not None and chosen_claim_id not in valid_claim_ids:
            return None, "chosen_claim_id_not_candidate"

        normalized = {
            "chosen_claim_id": chosen_claim_id,
            "reason": reason.strip(),
        }
        if confidence is not None:
            normalized["confidence"] = float(confidence)
        return normalized, None


class OpenAIResponsesClient:
    """
    OpenAI Responses API adapter.

    This class intentionally performs imports lazily so the core package can be
    imported without llm extras installed.
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        api_mode: str = "auto",
        timeout: float | None = None,
        hard_timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        self._client = client
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.hard_timeout = hard_timeout
        self.max_retries = max_retries
        normalized_mode = api_mode.strip().lower().replace("-", "_")
        if normalized_mode not in {"auto", "responses", "chat_completions"}:
            raise ValueError("api_mode must be 'auto', 'responses', or 'chat_completions'.")
        self.api_mode = normalized_mode
        self.last_transport_mode: str | None = None

    @classmethod
    def from_env(cls) -> OpenAIResponsesClient:
        api_key = (
            os.getenv("STATEFUSE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        ).strip() or None
        base_url = (
            os.getenv("STATEFUSE_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL") or ""
        ).strip() or None
        api_mode = os.getenv("STATEFUSE_OPENAI_API_MODE", "chat_completions")
        timeout_raw = (os.getenv("STATEFUSE_OPENAI_TIMEOUT") or "").strip()
        timeout = float(timeout_raw) if timeout_raw else None
        hard_timeout_raw = (os.getenv("STATEFUSE_OPENAI_HARD_TIMEOUT") or "").strip()
        hard_timeout = float(hard_timeout_raw) if hard_timeout_raw else timeout
        retries_raw = (os.getenv("STATEFUSE_OPENAI_MAX_RETRIES") or "").strip()
        max_retries = int(retries_raw) if retries_raw else None
        return cls(
            api_key=api_key,
            base_url=base_url,
            api_mode=api_mode,
            timeout=timeout,
            hard_timeout=hard_timeout,
            max_retries=max_retries,
        )

    def configuration_summary(self, *, model: str | None = None) -> dict[str, str]:
        return {
            "base_url": self.base_url or "https://api.openai.com/v1",
            "api_mode": self.api_mode,
            "model": model or "",
            "has_api_key": "true" if bool(self.api_key) else "false",
        }

    def resolve_json(
        self,
        *,
        system_prompt: str,
        input_payload: Mapping[str, Any],
        model: str,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> str:
        if self._client is None and self.hard_timeout is not None:
            return self._resolve_json_via_subprocess(
                system_prompt=system_prompt,
                input_payload=input_payload,
                model=model,
                temperature=temperature,
                seed=seed,
            )

        client = self._get_client()
        response_error: Exception | None = None
        self.last_transport_mode = None

        with self._hard_deadline():
            if self.api_mode in {"auto", "responses"}:
                try:
                    raw = self._resolve_via_responses(
                        client=client,
                        system_prompt=system_prompt,
                        input_payload=input_payload,
                        model=model,
                        temperature=temperature,
                        seed=seed,
                    )
                    self.last_transport_mode = "responses"
                    return raw
                except Exception as exc:
                    response_error = exc
                    if self.api_mode == "responses":
                        raise

            if self.api_mode in {"auto", "chat_completions"}:
                try:
                    raw = self._resolve_via_chat_completions(
                        client=client,
                        system_prompt=system_prompt,
                        input_payload=input_payload,
                        model=model,
                        temperature=temperature,
                        seed=seed,
                    )
                    self.last_transport_mode = "chat_completions"
                    return raw
                except Exception as chat_error:
                    if response_error is not None:
                        raise response_error from chat_error
                    raise

        raise RuntimeError("No API mode available for LLM resolution.")

    def _resolve_json_via_subprocess(
        self,
        *,
        system_prompt: str,
        input_payload: Mapping[str, Any],
        model: str,
        temperature: float | None,
        seed: int | None,
    ) -> str:
        self.last_transport_mode = None
        queue: Any = mp.get_context("spawn").Queue()
        process = mp.get_context("spawn").Process(
            target=_subprocess_resolve_json,
            kwargs={
                "queue": queue,
                "base_url": self.base_url,
                "api_key": self.api_key,
                "api_mode": self.api_mode,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
                "system_prompt": system_prompt,
                "input_payload": dict(input_payload),
                "model": model,
                "temperature": temperature,
                "seed": seed,
            },
        )
        process.start()
        process.join(self.hard_timeout)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
            raise TimeoutError(f"LLM call exceeded hard timeout of {self.hard_timeout:.1f}s.")

        if process.exitcode not in {0, None}:
            raise RuntimeError(f"LLM worker exited with code {process.exitcode}.")

        try:
            status, detail, payload = queue.get(timeout=1.0)
        except Exception as exc:
            raise RuntimeError("LLM worker exited without returning a result.") from exc
        if status == "ok":
            self.last_transport_mode = str(detail)
            return str(payload)
        raise RuntimeError(f"Subprocess LLM call failed ({detail}): {payload}")

    @contextmanager
    def _hard_deadline(self):
        if self.hard_timeout is None or not hasattr(signal, "setitimer"):
            yield
            return

        def _raise_timeout(signum, frame):  # type: ignore[no-untyped-def]
            raise TimeoutError(f"LLM call exceeded hard timeout of {self.hard_timeout:.1f}s.")

        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, self.hard_timeout)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            kwargs: dict[str, Any] = {}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.timeout is not None:
                kwargs["timeout"] = self.timeout
            if self.max_retries is not None:
                kwargs["max_retries"] = self.max_retries
            self._client = OpenAI(**kwargs)
        return self._client

    def _resolve_via_responses(
        self,
        *,
        client: Any,
        system_prompt: str,
        input_payload: Mapping[str, Any],
        model: str,
        temperature: float | None,
        seed: int | None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": canonical_json_dumps(dict(input_payload))}
                    ],
                },
            ],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if seed is not None:
            kwargs["seed"] = seed

        response = client.responses.create(**kwargs)
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        fragments: list[str] = []
        for item in getattr(response, "output", []):
            for content in getattr(item, "content", []):
                text = getattr(content, "text", None)
                if isinstance(text, str):
                    fragments.append(text)
        merged = "".join(fragments).strip()
        return merged or "{}"

    def _resolve_via_chat_completions(
        self,
        *,
        client: Any,
        system_prompt: str,
        input_payload: Mapping[str, Any],
        model: str,
        temperature: float | None,
        seed: int | None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": canonical_json_dumps(dict(input_payload))},
            ],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if seed is not None:
            kwargs["seed"] = seed

        response = client.chat.completions.create(**kwargs)
        choices = getattr(response, "choices", [])
        if not choices:
            return "{}"
        message = getattr(choices[0], "message", None)
        if message is None:
            return "{}"
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip() or "{}"
        if isinstance(content, list):
            fragments: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        fragments.append(text)
                    continue
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    fragments.append(text)
            merged = "".join(fragments).strip()
            return merged or "{}"
        return "{}"
