from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .data import validate_market_data
from .schemas import BacktestMetrics, StrategyFamily, StrategySpec


TRADING_DAYS = 252


def _stateful_mean_reversion_signal(
    close: pd.Series,
    lookback: int,
    entry_z: float,
    exit_z: float,
) -> pd.Series:
    mean = close.rolling(lookback, min_periods=lookback).mean()
    std = close.rolling(lookback, min_periods=lookback).std(ddof=0)
    zscore = (close - mean) / std.replace(0.0, np.nan)

    values = np.zeros(len(close), dtype=float)
    position = 0.0

    for i, z in enumerate(zscore.to_numpy()):
        if np.isnan(z):
            values[i] = position
            continue

        if position == 0.0 and z <= -entry_z:
            position = 1.0
        elif position == 1.0 and z >= exit_z:
            position = 0.0

        values[i] = position

    return pd.Series(values, index=close.index, dtype=float)


def signal_for_symbol(frame: pd.DataFrame, spec: StrategySpec) -> pd.Series:
    close = frame["close"].astype(float)

    if spec.family == StrategyFamily.TREND:
        fast = close.rolling(
            spec.fast_window,
            min_periods=spec.fast_window,
        ).mean()
        slow = close.rolling(
            spec.slow_window,
            min_periods=spec.slow_window,
        ).mean()
        signal = (fast > slow).astype(float)

    elif spec.family == StrategyFamily.MEAN_REVERSION:
        signal = _stateful_mean_reversion_signal(
            close,
            lookback=spec.lookback,
            entry_z=spec.entry_z,
            exit_z=spec.exit_z,
        )

    elif spec.family == StrategyFamily.BREAKOUT:
        prior_high = (
            close.rolling(spec.lookback, min_periods=spec.lookback).max().shift(1)
        )
        prior_low = (
            close.rolling(
                spec.exit_lookback,
                min_periods=spec.exit_lookback,
            )
            .min()
            .shift(1)
        )

        values = np.zeros(len(close), dtype=float)
        position = 0.0

        for i, price in enumerate(close.to_numpy()):
            high = prior_high.iloc[i]
            low = prior_low.iloc[i]

            if not np.isnan(high) and position == 0.0 and price > high:
                position = 1.0
            elif not np.isnan(low) and position == 1.0 and price < low:
                position = 0.0

            values[i] = position

        signal = pd.Series(values, index=close.index, dtype=float)

    else:
        raise ValueError(f"Unsupported strategy family: {spec.family}")

    return signal.clip(0.0, 1.0) * spec.max_position


def _metrics(returns: pd.Series, turnover: pd.Series) -> BacktestMetrics:
    returns = returns.fillna(0.0).astype(float)
    turnover = turnover.fillna(0.0).astype(float)

    n = int(len(returns))
    if n == 0:
        raise ValueError("No returns available.")

    mean = float(returns.mean())
    std = float(returns.std(ddof=0))
    sharpe = 0.0 if std == 0.0 else math.sqrt(TRADING_DAYS) * mean / std

    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min())

    total_return = float(equity.iloc[-1] - 1.0)
    years = max(n / TRADING_DAYS, 1.0 / TRADING_DAYS)
    cagr = (
        -1.0
        if equity.iloc[-1] <= 0.0
        else float(equity.iloc[-1] ** (1.0 / years) - 1.0)
    )

    return BacktestMetrics(
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        annual_turnover=float(turnover.mean() * TRADING_DAYS),
        total_return=total_return,
        cagr=cagr,
        annual_volatility=float(std * math.sqrt(TRADING_DAYS)),
        trading_days=n,
    )


def run_backtest(
    market_data: pd.DataFrame,
    spec: StrategySpec,
    *,
    transaction_cost_bps: float = 5.0,
) -> BacktestMetrics:
    data = validate_market_data(market_data)
    cost_rate = transaction_cost_bps / 10_000.0
    frames: list[pd.DataFrame] = []

    for symbol, frame in data.groupby("symbol", sort=True):
        frame = frame.sort_values("date").copy()
        signal = signal_for_symbol(frame, spec)

        # Today's close creates a signal for the next trading day.
        held_position = signal.shift(1).fillna(0.0)
        asset_return = frame["close"].pct_change().fillna(0.0)
        turnover = held_position.diff().abs().fillna(held_position.abs())
        strategy_return = held_position * asset_return - turnover * cost_rate

        frames.append(
            pd.DataFrame(
                {
                    "date": frame["date"].to_numpy(),
                    "symbol": symbol,
                    "strategy_return": strategy_return.to_numpy(),
                    "turnover": turnover.to_numpy(),
                }
            )
        )

    combined = pd.concat(frames, ignore_index=True)

    # Equal weight all ETFs on each date.
    portfolio = combined.groupby("date", sort=True).agg(
        strategy_return=("strategy_return", "mean"),
        turnover=("turnover", "mean"),
    )

    return _metrics(portfolio["strategy_return"], portfolio["turnover"])
