from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .agents import AgentSuite
from .config import AppConfig
from .data import (
    chronological_split,
    download_etf_data,
    make_synthetic_data,
    save_partitions,
)
from .report import markdown_summary, write_validation_report
from .sandbox import score_in_subprocess
from .schemas import CoordinatorDecision, EvaluationResult, StrategySpec


@dataclass
class RunArtifacts:
    root: Path
    summary_path: Path
    champion_path: Path
    holdout_result_path: Path


def apply_gates(
    result: EvaluationResult,
    cfg: AppConfig,
) -> EvaluationResult:
    failures: list[str] = []
    metrics = result.metrics
    selection = cfg.selection

    if metrics.sharpe < selection.min_validation_sharpe:
        failures.append(
            f"sharpe {metrics.sharpe:.3f} < "
            f"{selection.min_validation_sharpe:.3f}"
        )

    if abs(metrics.max_drawdown) > selection.max_validation_drawdown:
        failures.append(
            f"|drawdown| {abs(metrics.max_drawdown):.3f} > "
            f"{selection.max_validation_drawdown:.3f}"
        )

    if metrics.annual_turnover > selection.max_annual_turnover:
        failures.append(
            f"turnover {metrics.annual_turnover:.3f} > "
            f"{selection.max_annual_turnover:.3f}"
        )

    return result.model_copy(
        update={
            "passed_gates": not failures,
            "gate_failures": failures,
        }
    )


def champion_sort_key(
    result: EvaluationResult,
) -> tuple[float, float, float, str]:
    return (
        -result.metrics.sharpe,
        abs(result.metrics.max_drawdown),
        result.metrics.annual_turnover,
        result.strategy.name,
    )


def choose_champion(results: list[EvaluationResult]) -> EvaluationResult:
    passing = [row for row in results if row.passed_gates]

    if not passing:
        raise RuntimeError(
            "No strategy passed the fixed validation gates. "
            "Holdout will remain sealed."
        )

    return min(passing, key=champion_sort_key)


def freeze_champion(
    champion: EvaluationResult,
    root: Path,
) -> Path:
    frozen_dir = root / "frozen"
    frozen_dir.mkdir(parents=True, exist_ok=True)

    strategy_path = frozen_dir / "champion_strategy.json"
    hash_path = frozen_dir / "champion_strategy.sha256"

    strategy_path.write_text(
        champion.strategy.canonical_json(),
        encoding="utf-8",
    )
    hash_path.write_text(
        champion.strategy.sha256() + "\n",
        encoding="utf-8",
    )
    return strategy_path


def load_and_verify_frozen_strategy(root: Path) -> StrategySpec:
    strategy_path = root / "frozen" / "champion_strategy.json"
    hash_path = root / "frozen" / "champion_strategy.sha256"

    strategy = StrategySpec.model_validate_json(
        strategy_path.read_text(encoding="utf-8")
    )
    expected = hash_path.read_text(encoding="utf-8").strip()

    if strategy.sha256() != expected:
        raise RuntimeError(
            "Frozen strategy hash mismatch. Holdout evaluation refused."
        )

    return strategy


def evaluate_holdout_once(
    *,
    root: Path,
    holdout_path: Path,
    cfg: AppConfig,
) -> EvaluationResult:
    marker = root / "HOLDOUT_CONSUMED"

    if marker.exists():
        raise RuntimeError(
            "This experiment's holdout was already evaluated."
        )

    strategy = load_and_verify_frozen_strategy(root)
    metrics = score_in_subprocess(
        data_path=holdout_path,
        strategy=strategy,
        transaction_cost_bps=cfg.data.transaction_cost_bps,
        timeout_seconds=cfg.experiment.subprocess_timeout_seconds,
    )

    result = EvaluationResult(
        strategy=strategy,
        metrics=metrics,
        stage="holdout",
    )

    result_path = root / "holdout_result.json"
    result_path.write_text(
        json.dumps(
            result.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    marker.write_text(
        "This experiment's holdout was evaluated exactly once.\n",
        encoding="utf-8",
    )
    return result


def run_pipeline(
    *,
    cfg: AppConfig,
    agents: AgentSuite,
    artifact_root: Path,
    synthetic: bool,
) -> RunArtifacts:
    artifact_root.mkdir(parents=True, exist_ok=False)

    market_data = (
        make_synthetic_data(cfg.data, seed=cfg.experiment.seed)
        if synthetic
        else download_etf_data(cfg.data)
    )

    train, validation, holdout = chronological_split(
        market_data,
        cfg.data,
    )
    partition_paths = save_partitions(
        train,
        validation,
        holdout,
        artifact_root / "data",
    )

    (artifact_root / "train_summary.json").write_text(
        json.dumps(
            {
                "rows": int(len(train)),
                "start": str(train["date"].min().date()),
                "end": str(train["date"].max().date()),
                "symbols": sorted(train["symbol"].unique().tolist()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    all_results: list[EvaluationResult] = []
    direction: CoordinatorDecision | None = None

    for round_index in range(1, cfg.experiment.rounds + 1):
        strategies = agents.propose(
            round_index=round_index,
            count=cfg.experiment.proposals_per_round,
            history=all_results,
            direction=direction,
        )

        round_results: list[EvaluationResult] = []

        for strategy in strategies:
            metrics = score_in_subprocess(
                data_path=partition_paths["validation"],
                strategy=strategy,
                transaction_cost_bps=cfg.data.transaction_cost_bps,
                timeout_seconds=cfg.experiment.subprocess_timeout_seconds,
            )
            result = EvaluationResult(
                strategy=strategy,
                metrics=metrics,
                stage="validation",
            )
            result = apply_gates(result, cfg)
            round_results.append(result)
            all_results.append(result)

        critique = agents.critique(round_results)
        direction = agents.coordinate(round_results, critique)

        round_dir = artifact_root / "rounds" / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        (round_dir / "strategies.json").write_text(
            json.dumps(
                [
                    strategy.model_dump(
                        mode="json",
                        exclude_none=True,
                    )
                    for strategy in strategies
                ],
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (round_dir / "results.json").write_text(
            json.dumps(
                [
                    result.model_dump(mode="json")
                    for result in round_results
                ],
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (round_dir / "critique.json").write_text(
            critique.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (round_dir / "coordinator.json").write_text(
            direction.model_dump_json(indent=2),
            encoding="utf-8",
        )

        if direction.direction == "stop":
            break

    write_validation_report(
        all_results,
        artifact_root / "validation_results.json",
    )

    champion = choose_champion(all_results)
    champion_path = freeze_champion(champion, artifact_root)

    holdout_result = evaluate_holdout_once(
        root=artifact_root,
        holdout_path=partition_paths["holdout"],
        cfg=cfg,
    )

    summary_path = artifact_root / "SUMMARY.md"
    summary_path.write_text(
        markdown_summary(
            all_results,
            champion,
            holdout_result,
        ),
        encoding="utf-8",
    )

    return RunArtifacts(
        root=artifact_root,
        summary_path=summary_path,
        champion_path=champion_path,
        holdout_result_path=artifact_root / "holdout_result.json",
    )
