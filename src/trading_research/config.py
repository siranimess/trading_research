from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class ExperimentConfig(BaseModel):
    seed: int = 42
    rounds: int = Field(default=3, ge=1, le=10)
    proposals_per_round: int = Field(default=3, ge=1, le=10)
    subprocess_timeout_seconds: int = Field(default=30, ge=1, le=300)


class DataConfig(BaseModel):
    tickers: list[str] = Field(default_factory=lambda: ["SPY"])
    start: str = "2014-01-01"
    end: str = "2026-01-01"
    train_end: str = "2019-12-31"
    validation_end: str = "2022-12-31"
    transaction_cost_bps: float = Field(default=5.0, ge=0.0, le=100.0)

    @model_validator(mode="after")
    def validate_tickers(self) -> "DataConfig":
        if not self.tickers:
            raise ValueError("At least one ticker is required.")
        self.tickers = [ticker.upper().strip() for ticker in self.tickers]
        return self


class SelectionConfig(BaseModel):
    min_validation_sharpe: float = 0.50
    max_validation_drawdown: float = Field(default=0.30, gt=0.0, le=1.0)
    max_annual_turnover: float = Field(default=30.0, gt=0.0)


class AppConfig(BaseModel):
    experiment: ExperimentConfig = ExperimentConfig()
    data: DataConfig = DataConfig()
    selection: SelectionConfig = SelectionConfig()


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(raw)
