from __future__ import annotations

import json
from pathlib import Path

from .schemas import EvaluationResult


def write_validation_report(
    results: list[EvaluationResult],
    path: Path,
) -> None:
    path.write_text(
        json.dumps(
            [row.model_dump(mode="json") for row in results],
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def markdown_summary(
    validation_results: list[EvaluationResult],
    champion: EvaluationResult,
    holdout: EvaluationResult,
) -> str:
    rows = []

    for row in validation_results:
        rows.append(
            "| {name} | {family} | {sh:.3f} | {dd:.2%} | {turn:.2f} | {passed} |".format(
                name=row.strategy.name,
                family=row.strategy.family.value,
                sh=row.metrics.sharpe,
                dd=row.metrics.max_drawdown,
                turn=row.metrics.annual_turnover,
                passed="yes" if row.passed_gates else "no",
            )
        )

    return "\n".join(
        [
            "# Experiment Summary",
            "",
            "## Validation candidates",
            "",
            "| Strategy | Family | Sharpe | Max DD | Annual Turnover | Passed |",
            "|---|---|---:|---:|---:|---|",
            *rows,
            "",
            "## Frozen champion",
            "",
            f"- Name: `{champion.strategy.name}`",
            f"- Family: `{champion.strategy.family.value}`",
            f"- SHA-256: `{champion.strategy.sha256()}`",
            f"- Validation Sharpe: `{champion.metrics.sharpe:.3f}`",
            f"- Validation Max DD: `{champion.metrics.max_drawdown:.2%}`",
            "",
            "## Single holdout evaluation",
            "",
            f"- Holdout Sharpe: `{holdout.metrics.sharpe:.3f}`",
            f"- Holdout Max DD: `{holdout.metrics.max_drawdown:.2%}`",
            f"- Holdout CAGR: `{holdout.metrics.cagr:.2%}`",
            f"- Holdout annual turnover: `{holdout.metrics.annual_turnover:.2f}`",
            "",
            "> Research output only; historical performance is not a guarantee.",
            "",
        ]
    )
