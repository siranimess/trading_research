from trading_research.config import DataConfig
from trading_research.data import chronological_split, make_synthetic_data


def test_split_is_chronological():
    cfg = DataConfig(
        tickers=["SPY"],
        start="2018-01-01",
        end="2024-01-01",
        train_end="2019-12-31",
        validation_end="2021-12-31",
    )

    data = make_synthetic_data(cfg, seed=123)
    train, validation, holdout = chronological_split(data, cfg)

    assert train["date"].max() < validation["date"].min()
    assert validation["date"].max() < holdout["date"].min()
    assert set(train["symbol"]) == {"SPY"}
    assert set(validation["symbol"]) == {"SPY"}
    assert set(holdout["symbol"]) == {"SPY"}
