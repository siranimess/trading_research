from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class StrategyFamily(str, Enum):
    TREND = "trend"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"


class StrategySpec(BaseModel):
    name: str = Field(min_length=3, max_length=80, pattern=r"^[a-zA-Z0-9_\-]+$")
    family: StrategyFamily
    fast_window: int | None = Field(default=None, ge=2, le=250)
    slow_window: int | None = Field(default=None, ge=3, le=500)
    lookback: int | None = Field(default=None, ge=5, le=500)
    entry_z: float | None = Field(default=None, ge=0.25, le=5.0)
    exit_z: float | None = Field(default=None, ge=-2.0, le=2.0)
    exit_lookback: int | None = Field(default=None, ge=2, le=250)
    max_position: float = Field(default=1.0, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_family_parameters(self) -> "StrategySpec":
        if self.family == StrategyFamily.TREND:
            if self.fast_window is None or self.slow_window is None:
                raise ValueError("trend requires fast_window and slow_window")
            if self.fast_window >= self.slow_window:
                raise ValueError("fast_window must be less than slow_window")

        elif self.family == StrategyFamily.MEAN_REVERSION:
            if self.lookback is None or self.entry_z is None or self.exit_z is None:
                raise ValueError(
                    "mean_reversion requires lookback, entry_z and exit_z"
                )

        elif self.family == StrategyFamily.BREAKOUT:
            if self.lookback is None or self.exit_lookback is None:
                raise ValueError("breakout requires lookback and exit_lookback")
            if self.exit_lookback >= self.lookback:
                raise ValueError("exit_lookback must be smaller than lookback")

        return self

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json", exclude_none=True),
            sort_keys=True,
            separators=(",", ":"),
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class BacktestMetrics(BaseModel):
    sharpe: float
    max_drawdown: float
    annual_turnover: float
    total_return: float
    cagr: float
    annual_volatility: float
    trading_days: int


class EvaluationResult(BaseModel):
    strategy: StrategySpec
    metrics: BacktestMetrics
    passed_gates: bool = False
    gate_failures: list[str] = Field(default_factory=list)
    stage: Literal["validation", "holdout"] = "validation"


class Critique(BaseModel):
    summary: str
    risks: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class CoordinatorDecision(BaseModel):
    direction: Literal["refine", "diversify", "stop"]
    rationale: str
    preferred_family: StrategyFamily | None = None
