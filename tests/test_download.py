"""Offline tests for src.data.download: a fake exchange stands in for the live API."""

from __future__ import annotations

from datetime import date

import ccxt
import pandas as pd
import pytest

from src.data import download as dl

HOUR_MS = 3_600_000
T0 = 1_704_067_200_000  # 2024-01-01 00:00 UTC


class FakeExchange:
    """Serves hourly candles in [listed_ms, now_ms), Bitstamp-style: the window [since, since + limit * 1h)."""

    timeframes = {"1h": "3600"}
    features = {"spot": {"fetchOHLCV": {"limit": 5}}}

    def __init__(self, listed_ms: int = T0, now_ms: int = T0 + 24 * HOUR_MS, failures: list[Exception] | None = None):
        self.listed_ms = listed_ms
        self.now_ms = now_ms
        self.failures = list(failures or [])
        self.calls: list[int] = []

    def load_markets(self) -> dict:
        return {}

    def milliseconds(self) -> int:
        return self.now_ms

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: int, limit: int) -> list[list[float]]:
        self.calls.append(since)
        if self.failures:
            raise self.failures.pop(0)
        start = max(since, self.listed_ms)
        start += -start % HOUR_MS
        end = min(since + limit * HOUR_MS, self.now_ms)
        return [[t, 1.0, 2.0, 0.5, 1.5, 10.0] for t in range(start, end, HOUR_MS)]


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps: list[float] = []
    monkeypatch.setattr(dl.time, "sleep", sleeps.append)
    return sleeps


def test_to_milliseconds_treats_naive_as_utc() -> None:
    assert dl.to_milliseconds("2024-01-01") == T0
    assert dl.to_milliseconds(date(2024, 1, 1)) == T0
    assert dl.to_milliseconds("2024-01-01T01:00:00+01:00") == T0


def test_to_dataframe_sorts_dedups_and_uses_utc() -> None:
    candles = [[T0 + HOUR_MS, 2, 2, 2, 2, 2], [T0, 1, 1, 1, 1, 1], [T0 + HOUR_MS, 3, 3, 3, 3, 3]]
    df = dl.to_dataframe(candles, "BTC/USD")

    assert list(df.columns) == [*dl.OHLCV_COLUMNS, "symbol"]
    assert str(df["timestamp"].dt.tz) == "UTC"
    assert df["timestamp"].tolist() == [pd.Timestamp("2024-01-01 00:00", tz="UTC"), pd.Timestamp("2024-01-01 01:00", tz="UTC")]
    assert df["close"].tolist() == [1.0, 3.0]  # most recently fetched duplicate wins
    assert (df["symbol"] == "BTC/USD").all()


def test_to_dataframe_handles_empty_input() -> None:
    df = dl.to_dataframe([], "BTC/USD")
    assert df.empty
    assert list(df.columns) == [*dl.OHLCV_COLUMNS, "symbol"]


def test_fetch_paginates_over_full_range() -> None:
    ex = FakeExchange()
    candles = dl.fetch_ohlcv_range(ex, "BTC/USD", "1h", T0, T0 + 24 * HOUR_MS, limit=5)

    assert [c[0] for c in candles] == list(range(T0, T0 + 24 * HOUR_MS, HOUR_MS))
    assert len(ex.calls) == 5


def test_fetch_skips_empty_windows_before_listing() -> None:
    ex = FakeExchange(listed_ms=T0 + 12 * HOUR_MS)
    candles = dl.fetch_ohlcv_range(ex, "ETH/USD", "1h", T0, T0 + 24 * HOUR_MS, limit=5)

    assert [c[0] for c in candles] == list(range(T0 + 12 * HOUR_MS, T0 + 24 * HOUR_MS, HOUR_MS))


def test_fetch_keeps_only_closed_candles_before_end() -> None:
    ex = FakeExchange(now_ms=T0 + 3 * HOUR_MS + 30 * 60_000)  # 03:00 candle still open
    candles = dl.fetch_ohlcv_range(ex, "BTC/USD", "1h", T0, ex.now_ms, limit=5)
    assert [c[0] for c in candles] == [T0, T0 + HOUR_MS, T0 + 2 * HOUR_MS]

    candles = dl.fetch_ohlcv_range(FakeExchange(), "BTC/USD", "1h", T0, T0 + 2 * HOUR_MS, limit=5)
    assert [c[0] for c in candles] == [T0, T0 + HOUR_MS]  # end is exclusive


def test_fetch_retries_transient_errors(no_sleep: list[float]) -> None:
    ex = FakeExchange(failures=[ccxt.RequestTimeout("timeout"), ccxt.RateLimitExceeded("429")])
    candles = dl.fetch_ohlcv_range(ex, "BTC/USD", "1h", T0, T0 + 2 * HOUR_MS, limit=5, max_retries=3)

    assert len(candles) == 2
    assert no_sleep == [2.0, 4.0]


def test_with_retries_gives_up_after_max_retries(no_sleep: list[float]) -> None:
    def always_down() -> None:
        raise ccxt.ExchangeNotAvailable("down")

    with pytest.raises(ccxt.ExchangeNotAvailable):
        dl.with_retries(always_down, max_retries=2)
    assert len(no_sleep) == 2


def test_with_retries_does_not_retry_permanent_errors(no_sleep: list[float]) -> None:
    ex = FakeExchange(failures=[ccxt.BadSymbol("no such market")])
    with pytest.raises(ccxt.BadSymbol):
        dl.fetch_ohlcv_range(ex, "XXX/USD", "1h", T0, T0 + HOUR_MS, limit=5)
    assert len(ex.calls) == 1 and no_sleep == []


def test_load_config_applies_overrides(tmp_path) -> None:
    cfg_file = tmp_path / "data.yaml"
    cfg_file.write_text("exchange: kraken\nsymbols: [BTC/USD]\ntimeframe: 1d\nstart: 2020-01-01\n")

    cfg = dl.load_config(cfg_file, timeframe="4h", end=None)
    assert (cfg.exchange, cfg.symbols, cfg.timeframe) == ("kraken", ["BTC/USD"], "4h")
    assert dl.to_milliseconds(cfg.start) == dl.to_milliseconds("2020-01-01")  # YAML date object
    assert cfg.output_dir == dl.PROJECT_ROOT / "data" / "raw"

    cfg_file.write_text("exchnage: kraken\n")
    with pytest.raises(ValueError, match="Unknown config keys"):
        dl.load_config(cfg_file)


def test_default_config_file_is_valid() -> None:
    cfg = dl.load_config()
    assert cfg.symbols == ["BTC/USD", "ETH/USD"] and cfg.timeframe == "1h"
    assert cfg.exchange in ccxt.exchanges


def test_output_path_sanitizes_symbol(tmp_path) -> None:
    assert dl.output_path(tmp_path, "bitstamp", "BTC/USD", "1h") == tmp_path / "bitstamp_BTC-USD_1h.parquet"


def test_run_writes_one_parquet_per_symbol(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dl, "create_exchange", lambda _id: FakeExchange())
    cfg = dl.DownloadConfig(exchange="fake", symbols=["BTC/USD", "ETH/USD"], start="2024-01-01", output_dir=tmp_path)

    saved = dl.run(cfg)

    assert set(saved) == {"BTC/USD", "ETH/USD"}
    df = pd.read_parquet(saved["ETH/USD"])
    assert list(df.columns) == [*dl.OHLCV_COLUMNS, "symbol"]
    assert len(df) == 24 and df["timestamp"].is_monotonic_increasing and df["timestamp"].is_unique
    assert str(df["timestamp"].dt.tz) == "UTC"
    assert (df["symbol"] == "ETH/USD").all()


def test_run_continues_past_failed_symbol(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    class PartlyBroken(FakeExchange):
        def fetch_ohlcv(self, symbol, timeframe, since, limit):
            if symbol == "BAD/USD":
                raise ccxt.BadSymbol("no such market")
            return super().fetch_ohlcv(symbol, timeframe, since, limit)

    monkeypatch.setattr(dl, "create_exchange", lambda _id: PartlyBroken())
    cfg = dl.DownloadConfig(exchange="fake", symbols=["BAD/USD", "BTC/USD"], start="2024-01-01", output_dir=tmp_path)

    with pytest.raises(RuntimeError, match="BAD/USD"):
        dl.run(cfg)
    assert (tmp_path / "fake_BTC-USD_1h.parquet").exists()
