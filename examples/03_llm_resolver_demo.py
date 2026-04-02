from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Mapping

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from statefuse import InMemoryStore, Memory, ViewConstraints, merge, materialize
from statefuse.resolver_llm import LLMResolver, OpenAIResponsesClient
from statefuse.view import build_view


class FakeClient:
    def resolve_json(self, **kwargs):  # type: ignore[no-untyped-def]
        candidates = kwargs["input_payload"]["candidates"]
        chosen_claim_id = candidates[0]["claim_id"]
        return (
            "{"
            f'"chosen_claim_id":"{chosen_claim_id}",'
            '"reason":"Pick first candidate for deterministic demo.",'
            '"confidence":0.91'
            "}"
        )


def _load_env_file(path: pathlib.Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Environment file not found: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _truthy(value: str | None) -> bool:
    return bool(value and value.lower() in {"1", "true", "yes", "on"})


def _build_resolver(use_real_llm: bool) -> tuple[LLMResolver, str]:
    if not use_real_llm:
        return LLMResolver(client=FakeClient()), "fake-client"

    api_key = (os.getenv("STATEFUSE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or STATEFUSE_OPENAI_API_KEY is required for --real-llm mode.")

    model = os.getenv("STATEFUSE_OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("STATEFUSE_OPENAI_TEMPERATURE", "0"))
    seed_raw = os.getenv("STATEFUSE_OPENAI_SEED")
    base_url = (os.getenv("STATEFUSE_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "").strip()
    api_mode = os.getenv("STATEFUSE_OPENAI_API_MODE", "auto")
    seed = int(seed_raw) if seed_raw else None
    return (
        LLMResolver(
            client=OpenAIResponsesClient(
                api_key=api_key,
                base_url=base_url or None,
                api_mode=api_mode,
            ),
            model=model,
            temperature=temperature,
            seed=seed,
        ),
        "openai-compatible",
    )


def main(
    *,
    use_real_llm: bool = False,
    env_file: pathlib.Path | None = None,
    constraints_metadata: Mapping[str, str] | None = None,
) -> None:
    if env_file is not None:
        _load_env_file(env_file)

    env_forces_real = _truthy(os.getenv("STATEFUSE_USE_REAL_LLM"))
    resolver, resolver_name = _build_resolver(use_real_llm=use_real_llm or env_forces_real)

    mem_a = Memory(store=InMemoryStore(), replica_id="agentA")
    mem_b = Memory(store=InMemoryStore(), replica_id="agentB")

    e1 = mem_a.add_evidence(pointer="doc://a", content="city=NYC")
    e2 = mem_b.add_evidence(pointer="doc://b", content="city=SF")

    mem_a.add_claim(
        namespace="project",
        subject="hq",
        predicate="city",
        value="New York",
        confidence=0.8,
        evidence_ids=[e1],
    )
    mem_b.add_claim(
        namespace="project",
        subject="hq",
        predicate="city",
        value="San Francisco",
        confidence=0.8,
        evidence_ids=[e2],
    )

    merged = merge(mem_a.load_oplog(), mem_b.load_oplog())
    base_state = materialize(merged)
    projection = build_view(
        base_state,
        constraints=ViewConstraints(scope="demo", metadata=dict(constraints_metadata or {})),
        resolver=resolver,
    )

    print("Resolver:", resolver_name)
    print("Base conflicts:", len(base_state.conflicts))
    print("Selected claims:")
    for key, claim in projection.selected_claims.items():
        print(key.to_dict(), "->", claim.value)
    print("Unresolved conflicts:", len(projection.unresolved_conflicts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StateFuse LLM resolver demo.")
    parser.add_argument(
        "--real-llm",
        action="store_true",
        help="Use real OpenAI Responses API client instead of fake deterministic client.",
    )
    parser.add_argument(
        "--env-file",
        type=pathlib.Path,
        default=None,
        help="Optional env file to load before running (for example: .env.test).",
    )
    args = parser.parse_args()
    main(use_real_llm=args.real_llm, env_file=args.env_file)
