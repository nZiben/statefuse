from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from statefuse.resolver_llm import OpenAIResponsesClient  # noqa: E402


def _load_env_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Environment file not found: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def validate_client_configuration(
    client: OpenAIResponsesClient,
    *,
    model: str,
    require_base_url: bool = False,
) -> dict[str, str]:
    summary = client.configuration_summary(model=model)
    if summary["has_api_key"] != "true":
        raise RuntimeError("STATEFUSE_OPENAI_API_KEY or OPENAI_API_KEY must be configured.")
    if not model.strip():
        raise RuntimeError("STATEFUSE_OPENAI_MODEL or --model must be configured.")
    if require_base_url and not client.base_url:
        raise RuntimeError("STATEFUSE_OPENAI_BASE_URL is required for this smoke check.")
    return summary


def run_smoke_check(
    client: OpenAIResponsesClient,
    *,
    model: str,
    temperature: float,
) -> dict[str, str]:
    raw = client.resolve_json(
        system_prompt=(
            'Return JSON only in the exact shape {"status":"ok",'
            '"mode":"chat_completions or responses"}.'
        ),
        input_payload={
            "task": "smoke-test",
            "instructions": {"status": "literal ok", "mode": "transport mode used"},
        },
        model=model,
        temperature=temperature,
        seed=7,
    )
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError("Smoke response was not a JSON object.")
    status = str(payload.get("status", "")).strip().lower()
    mode = str(payload.get("mode", "")).strip().lower()
    if not status:
        raise RuntimeError(f"Smoke response did not include status: {raw}")
    if not mode:
        raise RuntimeError(f"Smoke response did not include mode: {raw}")
    if status not in {"ok", "ready", "success", "string"}:
        raise RuntimeError(f"Smoke response did not confirm readiness: {raw}")
    return {
        "model": model,
        "api_mode": client.api_mode,
        "transport_mode": client.last_transport_mode or "",
        "base_url": client.base_url or "https://api.openai.com/v1",
        "raw_response": raw,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an OpenAI-compatible endpoint.")
    parser.add_argument("--env-file", type=Path, default=None, help="Optional env file to load.")
    parser.add_argument("--model", type=str, default="", help="Optional model override.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for the smoke request.",
    )
    parser.add_argument(
        "--require-base-url",
        action="store_true",
        help="Fail if STATEFUSE_OPENAI_BASE_URL is missing.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    if args.env_file is not None:
        _load_env_file(args.env_file)
    client = OpenAIResponsesClient.from_env()
    model = args.model or os.getenv("STATEFUSE_OPENAI_MODEL", "").strip()
    summary = validate_client_configuration(
        client,
        model=model,
        require_base_url=args.require_base_url,
    )
    result = run_smoke_check(client, model=model, temperature=args.temperature)
    payload = {**summary, **result}
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
