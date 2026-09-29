# Alpha Research Agent

Quantitative crypto research project.

## Data Pipeline

`src/data/download.py` downloads historical OHLCV candles from a public exchange via [CCXT](https://github.com/ccxt/ccxt) (no API keys needed) and saves one Parquet file per symbol to `data/raw/` (git-ignored).

Settings (exchange, symbols, timeframe, start/end dates, output directory) live in `configs/data.yaml`. Any of them can be overridden with CLI flags (`--exchange`, `--symbols`, `--timeframe`, `--start`, `--end`, `--output-dir`, `--limit`).

```bash
pip install -r requirements.txt
python -m pytest                                   # offline unit tests, no network

# Small smoke test: one symbol, one day
python -m src.data.download --symbols BTC/USD --start 2024-01-01 --end 2024-01-02

# Full download from configs/data.yaml (run on the server, not locally)
python -m src.data.download
```

Output: `data/raw/<exchange>_<BASE-QUOTE>_<timeframe>.parquet` (e.g. `bitstamp_BTC-USD_1h.parquet`) with columns `timestamp` (UTC), `open`, `high`, `low`, `close`, `volume`, `symbol`. Only fully closed candles in `[start, end)` are kept. Re-running overwrites the file.
