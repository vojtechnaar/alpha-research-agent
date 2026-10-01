"""Download historical OHLCV candles from a public exchange via CCXT and store them as Parquet.

Usage (from the repository root):
    python -m src.data.download                      # settings from configs/data.yaml
    python -m src.data.download --symbols BTC/USD --start 2024-01-01 --end 2024-01-02

Used by: data/download_yahoo.py (save_parquet, PROJECT_ROOT). Its Parquet files are what every CLI
         reads with --data.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from datetime import date
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

import ccxt
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data.yaml"

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
FALLBACK_LIMIT = 500  # candles per request when CCXT does not know the exchange maximum
MAX_BACKOFF_SECONDS = 60.0

T = TypeVar("T")


@dataclass
class DownloadConfig:
    """Settings for one download run. Dates are UTC; `end=None` means now."""

    exchange: str = "bitstamp"
    symbols: list[str] = field(default_factory=lambda: ["BTC/USD", "ETH/USD"])
    timeframe: str = "1h"
    start: str | date = "2017-01-01"
    end: str | date | None = None
    output_dir: Path = Path("data/raw")
    limit: int | None = None  # candles per request; None = exchange maximum
    max_retries: int = 5

    def __post_init__(self) -> None:
        if isinstance(self.symbols, str):
            self.symbols = [self.symbols]
        # Relative paths are anchored at the repo root so the output location doesn't depend on cwd.
        self.output_dir = Path(self.output_dir)
        if not self.output_dir.is_absolute():
            self.output_dir = PROJECT_ROOT / self.output_dir


def load_config(path: Path | None = DEFAULT_CONFIG, **overrides: Any) -> DownloadConfig:
    """Build a config from an optional YAML file, then apply any non-None keyword overrides."""
    values: dict[str, Any] = {}
    if path is not None:
        values = yaml.safe_load(Path(path).read_text()) or {}
    values.update({key: value for key, value in overrides.items() if value is not None})
    unknown = set(values) - {f.name for f in fields(DownloadConfig)}
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return DownloadConfig(**values)


def to_milliseconds(value: str | date) -> int:
    """Convert a date/datetime (object or ISO string) to epoch milliseconds; naive values are UTC."""
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def with_retries(call: Callable[[], T], max_retries: int, base_delay: float = 2.0) -> T:
    """Run `call`, retrying transient network errors with exponential backoff.

    ccxt.NetworkError covers timeouts, rate-limit (429) and DDoS-protection responses, and
    exchange outages/maintenance. Other errors (bad symbol, bad params) are raised immediately.
    """
    attempt = 0
    while True:
        try:
            return call()
        except ccxt.NetworkError as exc:
            if attempt >= max_retries:
                raise
            delay = min(base_delay * 2**attempt, MAX_BACKOFF_SECONDS)
            attempt += 1
            logger.warning(
                "%s: %s - retry %d/%d in %.0fs",
                type(exc).__name__, str(exc)[:200], attempt, max_retries, delay,
            )
            time.sleep(delay)


def create_exchange(exchange_id: str) -> ccxt.Exchange:
    """Instantiate an unauthenticated CCXT client with CCXT's built-in rate limiter enabled."""
    if exchange_id not in ccxt.exchanges:
        raise ValueError(f"Unknown CCXT exchange id: {exchange_id!r}")
    exchange: ccxt.Exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True, "timeout": 30_000})
    if not exchange.has.get("fetchOHLCV"):
        raise ValueError(f"{exchange_id} does not support fetchOHLCV")
    return exchange


def page_limit(exchange: ccxt.Exchange, requested: int | None) -> int:
    """Candles per request: the configured value, else the exchange maximum known to CCXT."""
    if requested is not None:
        return requested
    spot = (getattr(exchange, "features", None) or {}).get("spot") or {}
    return (spot.get("fetchOHLCV") or {}).get("limit") or FALLBACK_LIMIT


def fetch_ohlcv_range(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
    limit: int,
    max_retries: int = 5,
) -> list[list[float]]:
    """Page forward from `start_ms` and return raw candles that open at or after `start_ms`
    and close at or before `end_ms` (i.e. only fully closed candles).

    An empty page is treated as a gap (e.g. before the market was listed) and skipped rather
    than ending the download, so a start date earlier than the listing date is safe.
    """
    tf_ms = ccxt.Exchange.parse_timeframe(timeframe) * 1000
    candles: list[list[float]] = []
    cursor = start_ms
    while cursor < end_ms:
        fetch = partial(exchange.fetch_ohlcv, symbol, timeframe, since=cursor, limit=limit)
        page = [c for c in with_retries(fetch, max_retries) if c[0] >= cursor]
        if not page:
            cursor += limit * tf_ms
            continue
        candles.extend(c for c in page if c[0] + tf_ms <= end_ms)
        cursor = max(c[0] for c in page) + tf_ms
        logger.debug("%s: %d candles, up to %s", symbol, len(candles), pd.Timestamp(cursor, unit="ms", tz="UTC"))
    return candles


def to_dataframe(candles: list[list[float]], symbol: str) -> pd.DataFrame:
    """Convert raw CCXT candles to a frame with UTC timestamps, sorted and one row per timestamp."""
    df = pd.DataFrame(candles, columns=OHLCV_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df[OHLCV_COLUMNS[1:]] = df[OHLCV_COLUMNS[1:]].astype("float64")
    # Stable sort keeps fetch order among duplicates, so keep="last" keeps the most recent fetch.
    df = (
        df.sort_values("timestamp", kind="stable")
        .drop_duplicates(subset="timestamp", keep="last")
        .reset_index(drop=True)
    )
    df["symbol"] = symbol
    return df


def output_path(output_dir: Path, exchange_id: str, symbol: str, timeframe: str) -> Path:
    """Parquet path for one dataset, e.g. data/raw/bitstamp_BTC-USD_1h.parquet."""
    safe_symbol = symbol.replace("/", "-").replace(":", "-")
    return Path(output_dir) / f"{exchange_id}_{safe_symbol}_{timeframe}.parquet"


def save_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Write `df` via a temp file + rename so an interrupted run never leaves a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


def run(config: DownloadConfig) -> dict[str, Path]:
    """Download and save every configured symbol; return {symbol: parquet path}.

    Symbols are attempted independently. If any fail, RuntimeError is raised after the rest finish.
    """
    exchange = create_exchange(config.exchange)
    with_retries(exchange.load_markets, config.max_retries)
    if exchange.timeframes and config.timeframe not in exchange.timeframes:
        raise ValueError(
            f"{config.exchange} does not support timeframe {config.timeframe!r}; "
            f"available: {list(exchange.timeframes)}"
        )
    start_ms = to_milliseconds(config.start)
    end_ms = to_milliseconds(config.end) if config.end is not None else exchange.milliseconds()
    if start_ms >= end_ms:
        raise ValueError(f"start ({config.start}) must be before end ({config.end or 'now'})")
    limit = page_limit(exchange, config.limit)
    bar = pd.Timedelta(seconds=ccxt.Exchange.parse_timeframe(config.timeframe))

    saved: dict[str, Path] = {}
    failed: list[str] = []
    for symbol in config.symbols:
        logger.info("Downloading %s %s from %s (%d candles/request)", symbol, config.timeframe, config.exchange, limit)
        try:
            candles = fetch_ohlcv_range(
                exchange, symbol, config.timeframe, start_ms, end_ms, limit, config.max_retries
            )
        except ccxt.BaseError as exc:
            logger.error("Failed to download %s: %s", symbol, exc)
            failed.append(symbol)
            continue
        df = to_dataframe(candles, symbol)
        if df.empty:
            logger.error("No candles returned for %s in the requested range", symbol)
            failed.append(symbol)
            continue
        path = save_parquet(df, output_path(config.output_dir, config.exchange, symbol, config.timeframe))
        first, last = df["timestamp"].iloc[0], df["timestamp"].iloc[-1]
        missing = int((last - first) / bar) + 1 - len(df)
        logger.info("Saved %d rows (%s -> %s, %d missing bars) to %s", len(df), first, last, missing, path)
        saved[symbol] = path

    if failed:
        raise RuntimeError(f"Download failed for: {', '.join(failed)}")
    return saved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI arguments; any option given overrides the value from the config file."""
    parser = argparse.ArgumentParser(description="Download historical OHLCV candles to Parquet.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config (default: %(default)s)")
    parser.add_argument("--exchange", help="CCXT exchange id, e.g. bitstamp")
    parser.add_argument("--symbols", nargs="+", help="e.g. BTC/USD ETH/USD")
    parser.add_argument("--timeframe", help="e.g. 1h, 1d")
    parser.add_argument("--start", help="UTC start, inclusive, e.g. 2020-01-01")
    parser.add_argument("--end", help="UTC end, exclusive (default: now)")
    parser.add_argument("--output-dir", type=Path, help="Directory for Parquet files")
    parser.add_argument("--limit", type=int, help="Candles per request (default: exchange maximum)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log every page")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    try:
        config = load_config(
            args.config,
            exchange=args.exchange,
            symbols=args.symbols,
            timeframe=args.timeframe,
            start=args.start,
            end=args.end,
            output_dir=args.output_dir,
            limit=args.limit,
        )
        run(config)
    except (ValueError, RuntimeError, ccxt.BaseError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
