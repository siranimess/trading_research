import numpy as np
import pandas as pd

from trading_research.backtest import run_backtest
from trading_research.schemas import StrategySpec


def make_data() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=400, freq="B")
    close = np.linspace(100, 180, len(dates))

    return pd.DataFrame(
        {
            "date": dates,
            "symbol": "SPY",
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000,
        }
    )


def test_trend_backtest_is_deterministic():
    data = make_data()
    strategy = StrategySpec(
        name="test_trend",
        family="trend",
        fast_window=10,
        slow_window=50,
    )

    first = run_backtest(data, strategy, transaction_cost_bps=5)
    second = run_backtest(data, strategy, transaction_cost_bps=5)

    assert first == second
    assert first.trading_days == 400
    assert first.max_drawdown <= 0.0
