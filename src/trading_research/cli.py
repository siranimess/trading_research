from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

from .agents import DeepAgentSuite, MockAgentSuite
from .config import load_config
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trading-research",
        description="Multi-Agent Trading Research System",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run",
        help="Run one complete research experiment.",
    )
    run.add_argument("--config", default="config.yaml")
    run.add_argument(
        "--mode",
        choices=["mock", "llm"],
        default="mock",
    )
    run.add_argument(
        "--synthetic",
        action="store_true",
        help="Use deterministic synthetic data instead of yfinance.",
    )
    run.add_argument("--artifact-root", default=None)
    run.add_argument("--model", default=None)

    clean = subparsers.add_parser(
        "clean",
        help="Delete local artifacts/.",
    )
    clean.add_argument("--yes", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "clean":
        path = Path("artifacts")
        if path.exists():
            if not args.yes:
                raise SystemExit("Re-run with --yes to delete artifacts/.")
            shutil.rmtree(path)
        print("Cleaned artifacts/.")
        return

    cfg = load_config(args.config)

    agents = (
        DeepAgentSuite(model=args.model)
        if args.mode == "llm"
        else MockAgentSuite()
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_root = (
        Path(args.artifact_root)
        if args.artifact_root
        else Path("artifacts") / f"run_{timestamp}"
    )

    artifacts = run_pipeline(
        cfg=cfg,
        agents=agents,
        artifact_root=artifact_root,
        synthetic=args.synthetic,
    )

    print()
    print("Experiment complete.")
    print(f"Root:     {artifacts.root}")
    print(f"Summary:  {artifacts.summary_path}")
    print(f"Champion: {artifacts.champion_path}")
    print(f"Holdout:  {artifacts.holdout_result_path}")


if __name__ == "__main__":
    main()
