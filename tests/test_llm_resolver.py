from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

from statefuse.materialize import materialize
from statefuse.model import Claim, ClaimKey
from statefuse.oplog import OpLog
from statefuse.ops import ClaimAdded
from statefuse.resolver import ViewConstraints
from statefuse.resolver_llm import LLMResolver, OpenAIResponsesClient
from statefuse.view import build_view

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from llm_endpoint_smoke import validate_client_configuration  # noqa: E402


def _conflicting_state() -> tuple[OpLog, str, str]:
    op1 = ClaimAdded(
        op_id="op-1",
        replica_id="a",
        timestamp="2026-03-01T00:00:00.000000Z",
        claim=Claim(
            claim_id="c1",
            key=ClaimKey(namespace="proj", subject="hq", predicate="city"),
            value="New York",
            confidence=0.9,
            timestamp="2026-03-01T00:00:00.000000Z",
            evidence_ids=(),
            provenance={"replica_id": "a"},
        ),
    )
    op2 = ClaimAdded(
        op_id="op-2",
        replica_id="b",
        timestamp="2026-03-01T00:00:01.000000Z",
        claim=Claim(
            claim_id="c2",
            key=ClaimKey(namespace="proj", subject="hq", predicate="city"),
            value="San Francisco",
            confidence=0.8,
            timestamp="2026-03-01T00:00:01.000000Z",
            evidence_ids=(),
            provenance={"replica_id": "b"},
        ),
    )
    return OpLog([op1, op2]), "c1", "c2"


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response

    def resolve_json(self, **kwargs):  # type: ignore[no-untyped-def]
        return self.response


def test_fake_llm_valid_json_resolves_conflict() -> None:
    oplog, c1, _ = _conflicting_state()
    state = materialize(oplog)
    resolver = LLMResolver(
        client=FakeLLMClient(
            f'{{"chosen_claim_id":"{c1}","reason":"higher confidence","confidence":0.95}}'
        )
    )
    projection = build_view(state, constraints=ViewConstraints(scope="task"), resolver=resolver)
    key = ClaimKey(namespace="proj", subject="hq", predicate="city")
    assert projection.selected_claims[key].claim_id == c1


def test_invalid_json_becomes_unresolved_without_crash() -> None:
    oplog, _, _ = _conflicting_state()
    state = materialize(oplog)
    resolver = LLMResolver(client=FakeLLMClient("not-json"))
    projection = build_view(state, constraints=ViewConstraints(scope="task"), resolver=resolver)
    assert len(projection.unresolved_conflicts) == 1


def test_projection_explanation_contains_raw_response() -> None:
    oplog, _, c2 = _conflicting_state()
    state = materialize(oplog)
    raw = f'{{"chosen_claim_id":"{c2}","reason":"pick c2"}}'
    resolver = LLMResolver(client=FakeLLMClient(raw))
    projection = build_view(state, constraints=ViewConstraints(scope="task"), resolver=resolver)
    explanation = projection.explanations["proj:hq:city"]
    assert "raw_response=" in explanation


class _ResponsesStub:
    def __init__(self, text: str | Exception) -> None:
        self.text = text

    def create(self, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(self.text, Exception):
            raise self.text
        return SimpleNamespace(output_text=self.text, output=[])


class _ChatCompletionsStub:
    def __init__(self, text: str) -> None:
        self.text = text

    def create(self, **kwargs):  # type: ignore[no-untyped-def]
        message = SimpleNamespace(content=self.text)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


def test_openai_client_auto_falls_back_to_chat_completions() -> None:
    fake_client = SimpleNamespace(
        responses=_ResponsesStub(RuntimeError("responses unsupported")),
        chat=SimpleNamespace(completions=_ChatCompletionsStub('{"chosen_claim_id":"c1","reason":"fallback"}')),
    )
    client = OpenAIResponsesClient(client=fake_client, api_mode="auto")
    raw = client.resolve_json(
        system_prompt="system",
        input_payload={"hello": "world"},
        model="fake-model",
        temperature=0.0,
        seed=42,
    )
    assert raw == '{"chosen_claim_id":"c1","reason":"fallback"}'


def test_openai_client_from_env_defaults_to_chat_completions(monkeypatch) -> None:
    monkeypatch.setenv("STATEFUSE_OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("STATEFUSE_OPENAI_API_MODE", raising=False)
    client = OpenAIResponsesClient.from_env()
    assert client.api_mode == "chat_completions"


def test_llm_endpoint_validation_requires_base_url_when_requested() -> None:
    client = OpenAIResponsesClient(
        client=SimpleNamespace(), api_key="test-key", api_mode="chat_completions"
    )
    try:
        validate_client_configuration(client, model="test-model", require_base_url=True)
    except RuntimeError as exc:
        assert "STATEFUSE_OPENAI_BASE_URL" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected missing base URL to fail validation")


def test_openai_client_hard_timeout_raises_timeout_error() -> None:
    class _SlowChat:
        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            time.sleep(0.05)
            return SimpleNamespace(choices=[])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=_SlowChat()))
    client = OpenAIResponsesClient(
        client=fake_client, api_mode="chat_completions", hard_timeout=0.01
    )
    try:
        client.resolve_json(
            system_prompt="system",
            input_payload={"hello": "world"},
            model="fake-model",
            temperature=0.0,
            seed=42,
        )
    except TimeoutError:
        return
    raise AssertionError("expected hard timeout to raise TimeoutError")


def test_llm_resolver_client_exception_becomes_unresolved() -> None:
    class FailingClient:
        def resolve_json(self, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("network down")

    oplog, _, _ = _conflicting_state()
    state = materialize(oplog)
    resolver = LLMResolver(client=FailingClient())
    projection = build_view(state, constraints=ViewConstraints(scope="task"), resolver=resolver)
    assert len(projection.unresolved_conflicts) == 1
    explanation = projection.explanations["proj:hq:city"]
    assert "LLM call failed" in explanation
