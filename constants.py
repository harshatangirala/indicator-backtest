"""
constants.py — shared config for engine.py and app.py.

Deliberately has NO heavy/optional dependencies (no yfinance, no numpy) so
that app.py (the Streamlit dashboard) can import these without pulling in
engine.py's download stack — that's what keeps the deployed dashboard's
install lean and fast to cold-start.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UNIVERSE_CSV = DATA_DIR / "universe" / "master_stock_universe.csv"
OHLCV_CACHE_DIR = DATA_DIR / "cache" / "ohlcv"
SIGNALS_CACHE_DIR = DATA_DIR / "cache" / "signals"
RESULTS_PATH = DATA_DIR / "backtest_results.parquet"

START_DATE = "2009-01-01"
END_DATE = "2026-06-26"

MIN_ROWS = 100  # skip tickers with less history than this (warm-up + viability)
TRADING_DAYS_PER_YEAR = 252

INDICATOR_NAMES = [
    "SMA", "EMA", "MACD", "PSAR", "RSI", "STOCH", "CCI", "ROC",
    "BBANDS", "ATR", "SUPERTREND", "OBV", "CMF", "VRSI", "ICHIMOKU",
]
N_INDICATORS = len(INDICATOR_NAMES)
LOGICS = ("AND", "MAJORITY")
