from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .backtest import run_backtest
from .schemas import StrategySpec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--transaction-cost-bps", required=True, type=float)
    args = parser.parse_args()

    cwd = Path.cwd().resolve()
    data_path = Path(args.data).resolve()
    strategy_path = Path(args.strategy).resolve()
    output_path = Path(args.output).resolve()

    for path in (data_path, strategy_path, output_path):
        if cwd not in path.parents:
            raise RuntimeError("Worker refused path outside its temp directory.")

    strategy = StrategySpec.model_validate_json(
        strategy_path.read_text(encoding="utf-8")
    )
    market_data = pd.read_csv(data_path, parse_dates=["date"])

    metrics = run_backtest(
        market_data,
        strategy,
        transaction_cost_bps=args.transaction_cost_bps,
    )

    output_path.write_text(
        json.dumps(metrics.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
