"""Download daily bars for other asset classes (stocks, bonds, gold, FX) from Yahoo Finance.

Writes the same Parquet format as src/data/download.py, so every other part of the project works
unchanged: data/raw/yahoo_<SYMBOL>_1d.parquet with timestamp (UTC), open, high, low, close,
volume, symbol.

    python -m src.data.download_yahoo                                    # SPY QQQ TLT GLD EURUSD=X since 2005
    python -m src.data.download_yahoo --symbols SPY GLD --start 2010-01-01

Prices are split- and dividend-adjusted (auto_adjust), so returns include dividends and splits
don't show up as crashes. Free Yahoo data only has long histories for daily bars. Markets without
real volume (FX) get NaN volume, so volume features are undefined there instead of misleading.

Used by: nothing else: a CLI. Its Parquet files are read with --data like the crypto ones.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.data.download import PROJECT_ROOT, save_parquet

logger = logging.getLogger(__name__)

# One liquid instrument per asset class: US large caps, US tech, long US Treasuries, gold, EUR/USD.
DEFAULT_SYMBOLS = ["SPY", "QQQ", "TLT", "GLD", "EURUSD=X"]
DEFAULT_START = "2005-01-01"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"


def to_ohlcv(history: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Convert a yfinance history table (DatetimeIndex; Open/High/Low/Close/Volume) to our format."""
    if history.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "symbol"])
    dates = pd.DatetimeIndex(history.index)
    out = pd.DataFrame({
        # Daily bars are labelled by their trading date; midnight UTC keeps them timezone-independent.
        "timestamp": pd.to_datetime(dates.date).tz_localize("UTC"),
        "open": history["Open"].to_numpy(dtype="float64"),
        "high": history["High"].to_numpy(dtype="float64"),
        "low": history["Low"].to_numpy(dtype="float64"),
        "close": history["Close"].to_numpy(dtype="float64"),
        "volume": history["Volume"].to_numpy(dtype="float64"),
    })
    if not (out["volume"].fillna(0) > 0).any():  # FX and some indices report no volume
        out["volume"] = float("nan")
    out = (out.dropna(subset=["close"])
              .sort_values("timestamp", kind="stable")
              .drop_duplicates(subset="timestamp", keep="last")
              .reset_index(drop=True))
    out["symbol"] = symbol
    return out


def download(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Daily adjusted bars for one Yahoo symbol."""
    import yfinance as yf  # imported here so the rest of the project (and its tests) don't need it

    history = yf.Ticker(symbol).history(start=start, end=end, interval="1d", auto_adjust=True)
    return to_ohlcv(history, symbol)


def output_path(symbol: str, output_dir: Path = OUTPUT_DIR) -> Path:
    """data/raw/yahoo_<SYMBOL>_1d.parquet (characters like '=' removed from the file name)."""
    safe = "".join(ch for ch in symbol if ch.isalnum() or ch in "-_")
    return Path(output_dir) / f"yahoo_{safe}_1d.parquet"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download daily bars from Yahoo Finance.")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", help="exclusive; default: today")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    failed = []
    for symbol in args.symbols:
        try:
            data = download(symbol, args.start, args.end)
        except Exception as exc:  # network / unknown symbol: report and continue with the others
            logger.error("%s: %s", symbol, exc)
            failed.append(symbol)
            continue
        if data.empty:
            logger.error("%s: no data returned", symbol)
            failed.append(symbol)
            continue
        path = save_parquet(data, output_path(symbol, args.output_dir))
        logger.info("%s: %d daily bars %s -> %s saved to %s", symbol, len(data),
                    data["timestamp"].iloc[0].date(), data["timestamp"].iloc[-1].date(), path)
    if failed:
        logger.error("Failed: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
