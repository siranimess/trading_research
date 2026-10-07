from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig


REQUIRED_COLUMNS = {"date", "symbol", "open", "high", "low", "close", "volume"}


def validate_market_data(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Market data missing columns: {sorted(missing)}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out["symbol"] = out["symbol"].astype(str).str.upper()

    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="raise")

    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)

    if out.duplicated(subset=["symbol", "date"]).any():
        raise ValueError("Duplicate symbol/date rows detected.")

    return out


def download_etf_data(cfg: DataConfig) -> pd.DataFrame:
    import yfinance as yf

    frames: list[pd.DataFrame] = []

    for ticker in cfg.tickers:
        df = yf.download(
            ticker,
            start=cfg.start,
            end=cfg.end,
            auto_adjust=True,
            progress=False,
            actions=False,
            threads=False,
        )
        if df.empty:
            raise RuntimeError(f"No data downloaded for {ticker}")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(col[0]).lower() for col in df.columns]
        else:
            df.columns = [str(col).lower() for col in df.columns]

        df = df.reset_index()
        df.columns = [str(col).lower() for col in df.columns]

        frames.append(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(df["date"]).dt.tz_localize(None),
                    "symbol": ticker,
                    "open": pd.to_numeric(df["open"], errors="coerce"),
                    "high": pd.to_numeric(df["high"], errors="coerce"),
                    "low": pd.to_numeric(df["low"], errors="coerce"),
                    "close": pd.to_numeric(df["close"], errors="coerce"),
                    "volume": pd.to_numeric(df["volume"], errors="coerce"),
                }
            ).dropna()
        )

    return validate_market_data(pd.concat(frames, ignore_index=True))


def make_synthetic_data(cfg: DataConfig, *, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(cfg.start, cfg.end, freq="B", inclusive="left")
    frames: list[pd.DataFrame] = []

    for index, ticker in enumerate(cfg.tickers):
        n = len(dates)
        cycle = 0.0015 * np.sin(np.arange(n) / (25 + 3 * index))
        slow_cycle = 0.0008 * np.sin(np.arange(n) / (110 + 7 * index))
        noise = rng.normal(0.0, 0.009 + index * 0.001, size=n)
        log_returns = 0.00025 + cycle + slow_cycle + noise
        close = 100.0 * np.exp(np.cumsum(log_returns))

        overnight = rng.normal(0.0, 0.002, size=n)
        open_ = close * np.exp(overnight)
        spread = np.abs(rng.normal(0.004, 0.002, size=n))
        high = np.maximum(open_, close) * (1.0 + spread)
        low = np.minimum(open_, close) * (1.0 - spread)
        volume = rng.integers(1_000_000, 20_000_000, size=n)

        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": ticker,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
        )

    return validate_market_data(pd.concat(frames, ignore_index=True))


def chronological_split(
    df: pd.DataFrame,
    cfg: DataConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = pd.Timestamp(cfg.train_end)
    validation_end = pd.Timestamp(cfg.validation_end)

    train = df[df["date"] <= train_end].copy()
    validation = df[
        (df["date"] > train_end) & (df["date"] <= validation_end)
    ].copy()
    holdout = df[df["date"] > validation_end].copy()

    if train.empty or validation.empty or holdout.empty:
        raise ValueError("A chronological partition is empty. Check config dates.")

    expected = set(cfg.tickers)
    for name, part in (
        ("train", train),
        ("validation", validation),
        ("holdout", holdout),
    ):
        if set(part["symbol"].unique()) != expected:
            raise ValueError(f"{name} is missing one or more configured tickers.")

    return train, validation, holdout


def save_partitions(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    holdout: pd.DataFrame,
    root: Path,
) -> dict[str, Path]:
    public_dir = root / "public"
    sealed_dir = root / "sealed"
    public_dir.mkdir(parents=True, exist_ok=True)
    sealed_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": public_dir / "train.csv",
        "validation": public_dir / "validation.csv",
        "holdout": sealed_dir / "holdout.csv",
    }
    train.to_csv(paths["train"], index=False)
    validation.to_csv(paths["validation"], index=False)
    holdout.to_csv(paths["holdout"], index=False)
    return paths
