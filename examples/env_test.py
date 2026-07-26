from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys


def load_env_file(path: pathlib.Path) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load env vars and run the real-LLM resolver demo against an "
            "OpenAI-compatible endpoint."
        )
    )
    parser.add_argument(
        "--env-file",
        type=pathlib.Path,
        default=pathlib.Path(".env.test"),
        help="Env file path (default: .env.test).",
    )
    args = parser.parse_args()

    load_env_file(args.env_file)

    api_key = (os.getenv("STATEFUSE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key or "your-key-here" in api_key:
        raise RuntimeError(
            "OPENAI_API_KEY or STATEFUSE_OPENAI_API_KEY is missing or placeholder. "
            "Put a real key in your env file."
        )

    os.environ["STATEFUSE_USE_REAL_LLM"] = "1"

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    demo_script = repo_root / "examples" / "03_llm_resolver_demo.py"

    cmd = [sys.executable, str(demo_script), "--real-llm"]
    print(f"Running: {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=repo_root, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
