from trading_research.config import AppConfig
from trading_research.pipeline import apply_gates, choose_champion
from trading_research.schemas import (
    BacktestMetrics,
    EvaluationResult,
    StrategySpec,
)


def make_result(
    name: str,
    sharpe: float,
    drawdown: float,
    turnover: float,
) -> EvaluationResult:
    return EvaluationResult(
        strategy=StrategySpec(
            name=name,
            family="trend",
            fast_window=10,
            slow_window=50,
        ),
        metrics=BacktestMetrics(
            sharpe=sharpe,
            max_drawdown=drawdown,
            annual_turnover=turnover,
            total_return=0.10,
            cagr=0.05,
            annual_volatility=0.10,
            trading_days=500,
        ),
    )


def test_gates_and_champion():
    cfg = AppConfig()

    first = apply_gates(
        make_result("first", 0.8, -0.20, 5.0),
        cfg,
    )
    second = apply_gates(
        make_result("second", 0.7, -0.10, 2.0),
        cfg,
    )
    bad = apply_gates(
        make_result("bad", 0.1, -0.10, 1.0),
        cfg,
    )

    assert first.passed_gates
    assert second.passed_gates
    assert not bad.passed_gates
    assert choose_champion([second, bad, first]).strategy.name == "first"
