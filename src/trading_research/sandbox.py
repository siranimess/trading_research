from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .schemas import BacktestMetrics, StrategySpec


SENSITIVE_ENV_MARKERS = (
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "OPENAI",
    "LANGSMITH",
    "ANTHROPIC",
    "TAVILY",
)


def _scrubbed_env() -> dict[str, str]:
    safe: dict[str, str] = {}

    for key, value in os.environ.items():
        upper = key.upper()
        if any(marker in upper for marker in SENSITIVE_ENV_MARKERS):
            continue
        safe[key] = value

    safe["PYTHONHASHSEED"] = "0"
    safe["PYTHONDONTWRITEBYTECODE"] = "1"

    # Keep the package importable even when the project is run directly from
    # source instead of installed with `pip install -e .`.
    source_root = str(Path(__file__).resolve().parents[1])
    existing_pythonpath = safe.get("PYTHONPATH", "")
    safe["PYTHONPATH"] = (
        source_root
        if not existing_pythonpath
        else source_root + os.pathsep + existing_pythonpath
    )
    return safe


def score_in_subprocess(
    *,
    data_path: Path,
    strategy: StrategySpec,
    transaction_cost_bps: float,
    timeout_seconds: int,
) -> BacktestMetrics:
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    with tempfile.TemporaryDirectory(prefix="trading-score-") as temp_str:
        temp_dir = Path(temp_str)
        local_data = temp_dir / "market.csv"
        local_strategy = temp_dir / "strategy.json"
        local_output = temp_dir / "metrics.json"

        shutil.copy2(data_path, local_data)
        local_strategy.write_text(
            strategy.model_dump_json(indent=2),
            encoding="utf-8",
        )

        command = [
            sys.executable,
            "-m",
            "trading_research.sandbox_worker",
            "--data",
            str(local_data),
            "--strategy",
            str(local_strategy),
            "--output",
            str(local_output),
            "--transaction-cost-bps",
            str(transaction_cost_bps),
        ]

        completed = subprocess.run(
            command,
            cwd=temp_dir,
            env=_scrubbed_env(),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                "Scoring subprocess failed.\n"
                f"STDOUT:\n{completed.stdout}\n"
                f"STDERR:\n{completed.stderr}"
            )

        payload = json.loads(local_output.read_text(encoding="utf-8"))
        return BacktestMetrics.model_validate(payload)
