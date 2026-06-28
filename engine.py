"""
engine.py — Indicator-combination backtesting engine for the Nifty Master Universe.

Pipeline
--------
1. Load the stock universe from data/universe/master_stock_universe.csv and map
   each NSE symbol to its Yahoo Finance ticker (SYMBOL.NS).
2. Download (or load from cache) daily OHLCV for every ticker, 2009-01-01 .. END_DATE.
3. Compute 15 indicator signal series (-1 / 0 / 1) per stock, ONCE, and cache them.
4. Align every stock's signals/returns onto one master trading-day calendar and
   stack into 3-D numpy arrays: signals_all (days, stocks, 15), returns_all (days, stocks).
5. Enumerate all C(15,1)+C(15,2)+C(15,3)+C(15,4) = 1940 indicator combinations,
   backtest each under AND and MAJORITY-vote logic using pure numpy ops over the
   full (days, stocks) grid at once (no per-stock / per-combo Python loops over rows),
   and write per-stock + overall-market metrics to data/backtest_results.parquet.

Run:
    python engine.py                  # full pipeline (download if needed, full backtest)
    python engine.py --no-download    # reuse cached OHLCV/signals, skip re-download
    python engine.py --quick 10       # smoke-test on the first N tickers only
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from constants import (
    BASE_DIR, DATA_DIR, UNIVERSE_CSV, OHLCV_CACHE_DIR, SIGNALS_CACHE_DIR,
    RESULTS_PATH, START_DATE, END_DATE, MIN_ROWS, TRADING_DAYS_PER_YEAR,
    INDICATOR_NAMES, N_INDICATORS, LOGICS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("engine")

# A handful of days at the very start of the master calendar can have zero
# listed stocks with valid data yet (pre-IPO padding) -> harmless empty-slice mean.
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")


# --------------------------------------------------------------------------- #
# 1. Universe loading
# --------------------------------------------------------------------------- #

def load_universe(csv_path: Path = UNIVERSE_CSV) -> pd.DataFrame:
    """Load master_stock_universe.csv and map Symbol -> Yahoo Finance ticker (.NS)."""
    df = pd.read_csv(csv_path)
    df["yf_ticker"] = df["Symbol"].str.strip().str.replace("&", "%26", regex=False) + ".NS"
    log.info("Loaded universe: %d stocks from %s", len(df), csv_path.name)
    return df


# --------------------------------------------------------------------------- #
# 2. OHLCV download + cache
# --------------------------------------------------------------------------- #

def download_ohlcv(ticker: str, start: str = START_DATE, end: str = END_DATE,
                    allow_download: bool = True) -> pd.DataFrame | None:
    """Load cached OHLCV for one ticker, downloading via yfinance if missing
    (unless allow_download is False, e.g. --no-download). Returns None on failure."""
    cache_path = OHLCV_CACHE_DIR / f"{ticker.replace('.', '_')}.parquet"

    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as exc:  # corrupt cache -> redownload (if allowed)
            log.warning("Cache read failed for %s (%s); redownloading", ticker, exc)

    if not allow_download:
        log.warning("No cache for %s and downloads are disabled (--no-download)", ticker)
        return None

    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                          progress=False, multi_level_index=False)
    except Exception as exc:
        log.warning("Download failed for %s: %s", ticker, exc)
        return None

    if df is None or df.empty:
        log.warning("No data returned for %s", ticker)
        return None

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    df.index.name = "Date"
    OHLCV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    return df


# --------------------------------------------------------------------------- #
# 3. Indicator signal functions — each returns a Series of -1 / 0 / +1
# --------------------------------------------------------------------------- #

def _rsi_value(series: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI, reused for both price-RSI and Volume-RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def sig_sma(c: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    return np.sign(c.rolling(fast).mean() - c.rolling(slow).mean())


def sig_ema(c: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    fast_e = c.ewm(span=fast, adjust=False).mean()
    slow_e = c.ewm(span=slow, adjust=False).mean()
    return np.sign(fast_e - slow_e)


def sig_macd(c: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    macd_line = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return np.sign(macd_line - signal_line)


def sig_psar(h: pd.Series, l: pd.Series, c: pd.Series,
             step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    """Parabolic SAR is inherently sequential (depends on prior bar's state).
    Computed with a single O(n) numpy loop, once per stock during pre-compute —
    never re-run inside the 1940-combination backtest loop."""
    hv, lv, cv = h.to_numpy(), l.to_numpy(), c.to_numpy()
    n = len(cv)
    sar = np.zeros(n)
    trend = np.ones(n, dtype=int)
    ep = np.zeros(n)
    af = np.zeros(n)
    if n == 0:
        return pd.Series(sar, index=c.index)

    trend[0] = 1
    sar[0] = lv[0]
    ep[0] = hv[0]
    af[0] = step

    for i in range(1, n):
        prev_sar = sar[i - 1]
        if trend[i - 1] == 1:
            cur = prev_sar + af[i - 1] * (ep[i - 1] - prev_sar)
            cur = min(cur, lv[i - 1], lv[i - 2] if i >= 2 else lv[i - 1])
            if lv[i] < cur:
                trend[i] = -1
                cur = ep[i - 1]
                ep[i] = lv[i]
                af[i] = step
            else:
                trend[i] = 1
                ep[i] = max(ep[i - 1], hv[i])
                af[i] = min(af[i - 1] + step, max_step) if hv[i] > ep[i - 1] else af[i - 1]
        else:
            cur = prev_sar + af[i - 1] * (ep[i - 1] - prev_sar)
            cur = max(cur, hv[i - 1], hv[i - 2] if i >= 2 else hv[i - 1])
            if hv[i] > cur:
                trend[i] = 1
                cur = ep[i - 1]
                ep[i] = hv[i]
                af[i] = step
            else:
                trend[i] = -1
                ep[i] = min(ep[i - 1], lv[i])
                af[i] = min(af[i - 1] + step, max_step) if lv[i] < ep[i - 1] else af[i - 1]
        sar[i] = cur

    return np.sign(cv - sar)


def sig_rsi(c: pd.Series, window: int = 14, low: float = 30, high: float = 70) -> pd.Series:
    r = _rsi_value(c, window)
    sig = pd.Series(0.0, index=c.index)
    sig[r < low] = 1
    sig[r > high] = -1
    return sig


def sig_stoch(h: pd.Series, l: pd.Series, c: pd.Series,
              window: int = 14, low_th: float = 20, high_th: float = 80) -> pd.Series:
    lowest = l.rolling(window).min()
    highest = h.rolling(window).max()
    pct_k = 100 * (c - lowest) / (highest - lowest).replace(0, np.nan)
    sig = pd.Series(0.0, index=c.index)
    sig[pct_k < low_th] = 1
    sig[pct_k > high_th] = -1
    return sig


def sig_cci(h: pd.Series, l: pd.Series, c: pd.Series,
            window: int = 20, threshold: float = 100) -> pd.Series:
    tp = (h + l + c) / 3
    sma = tp.rolling(window).mean()
    mad = tp.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma) / (0.015 * mad.replace(0, np.nan))
    sig = pd.Series(0.0, index=c.index)
    sig[cci < -threshold] = 1
    sig[cci > threshold] = -1
    return sig


def sig_roc(c: pd.Series, window: int = 12) -> pd.Series:
    roc = (c - c.shift(window)) / c.shift(window) * 100
    return np.sign(roc)


def sig_bbands(c: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    mid = c.rolling(window).mean()
    std = c.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    sig = pd.Series(0.0, index=c.index)
    sig[c < lower] = 1
    sig[c > upper] = -1
    return sig


def _atr_value(h: pd.Series, l: pd.Series, c: pd.Series, window: int = 14) -> pd.Series:
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def sig_atr(h: pd.Series, l: pd.Series, c: pd.Series,
            window: int = 14, mult: float = 1.0) -> pd.Series:
    """ATR is a volatility measure, not directional on its own. Signal = breakout:
    buy when today's move exceeds +mult*ATR, sell when it exceeds -mult*ATR."""
    a = _atr_value(h, l, c, window)
    move = c.diff()
    sig = pd.Series(0.0, index=c.index)
    sig[move > mult * a] = 1
    sig[move < -mult * a] = -1
    return sig


def sig_supertrend(h: pd.Series, l: pd.Series, c: pd.Series,
                    window: int = 10, mult: float = 3.0) -> pd.Series:
    """SuperTrend trend direction is sequential (trailing-stop logic); computed
    with a single O(n) numpy loop, once per stock during pre-compute."""
    a = _atr_value(h, l, c, window)
    hl2 = (h + l) / 2
    upper_v = (hl2 + mult * a).to_numpy().copy()
    lower_v = (hl2 - mult * a).to_numpy().copy()
    cv = c.to_numpy()
    n = len(cv)
    trend = np.ones(n, dtype=int)
    if n == 0:
        return pd.Series(trend, index=c.index, dtype=float)

    for i in range(1, n):
        if cv[i - 1] <= upper_v[i - 1]:
            upper_v[i] = min(upper_v[i], upper_v[i - 1])
        if cv[i - 1] >= lower_v[i - 1]:
            lower_v[i] = max(lower_v[i], lower_v[i - 1])
        if cv[i] > upper_v[i - 1]:
            trend[i] = 1
        elif cv[i] < lower_v[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]

    return pd.Series(trend, index=c.index, dtype=float)


def sig_obv(c: pd.Series, v: pd.Series, window: int = 20) -> pd.Series:
    direction = np.sign(c.diff()).fillna(0)
    obv = (direction * v).cumsum()
    return np.sign(obv - obv.rolling(window).mean())


def sig_cmf(h: pd.Series, l: pd.Series, c: pd.Series, v: pd.Series,
            window: int = 20) -> pd.Series:
    mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    mfv = mfm * v
    cmf = mfv.rolling(window).sum() / v.rolling(window).sum()
    return np.sign(cmf)


def sig_vrsi(v: pd.Series, window: int = 14, threshold: float = 50) -> pd.Series:
    """RSI computed on the volume series — rising vs falling volume momentum."""
    r = _rsi_value(v.astype(float), window)
    return np.sign(r - threshold)


def sig_ichimoku(h: pd.Series, l: pd.Series, c: pd.Series,
                  conv: int = 9, base: int = 26, span_b: int = 52,
                  displacement: int = 26) -> pd.Series:
    conv_line = (h.rolling(conv).max() + l.rolling(conv).min()) / 2
    base_line = (h.rolling(base).max() + l.rolling(base).min()) / 2
    span_a = ((conv_line + base_line) / 2).shift(displacement)
    span_b_line = ((h.rolling(span_b).max() + l.rolling(span_b).min()) / 2).shift(displacement)
    upper = pd.concat([span_a, span_b_line], axis=1).max(axis=1)
    lower = pd.concat([span_a, span_b_line], axis=1).min(axis=1)
    sig = pd.Series(0.0, index=c.index)
    sig[c > upper] = 1
    sig[c < lower] = -1
    return sig


def compute_all_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 15 indicator signals (-1/0/1) for one stock's OHLCV frame."""
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
    out = pd.DataFrame(index=df.index)
    out["SMA"] = sig_sma(c)
    out["EMA"] = sig_ema(c)
    out["MACD"] = sig_macd(c)
    out["PSAR"] = sig_psar(h, l, c)
    out["RSI"] = sig_rsi(c)
    out["STOCH"] = sig_stoch(h, l, c)
    out["CCI"] = sig_cci(h, l, c)
    out["ROC"] = sig_roc(c)
    out["BBANDS"] = sig_bbands(c)
    out["ATR"] = sig_atr(h, l, c)
    out["SUPERTREND"] = sig_supertrend(h, l, c)
    out["OBV"] = sig_obv(c, v)
    out["CMF"] = sig_cmf(h, l, c, v)
    out["VRSI"] = sig_vrsi(v)
    out["ICHIMOKU"] = sig_ichimoku(h, l, c)
    return out[INDICATOR_NAMES]


def get_signals_and_returns(ticker: str, df: pd.DataFrame, use_cache: bool = True
                             ) -> tuple[pd.DataFrame, pd.Series]:
    """Compute (or load cached) indicator signals + daily returns for one stock."""
    cache_path = SIGNALS_CACHE_DIR / f"{ticker.replace('.', '_')}.parquet"
    if use_cache and cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            cached.index = pd.to_datetime(cached.index)
            return cached[INDICATOR_NAMES], cached["return"]
        except Exception as exc:
            log.warning("Signal cache read failed for %s (%s); recomputing", ticker, exc)

    signals = compute_all_signals(df)
    returns = df["Close"].pct_change()
    combined = signals.copy()
    combined["return"] = returns
    SIGNALS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_path)
    return signals, returns


# --------------------------------------------------------------------------- #
# 4. Build master (days, stocks, indicators) arrays
# --------------------------------------------------------------------------- #

def build_master_arrays(universe: pd.DataFrame, allow_download: bool = True,
                         limit: int | None = None
                         ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, list[str]]:
    """Download/compute signals for every ticker and stack onto one master
    trading-day calendar. Returns (signals_all, returns_all, master_dates, tickers).

    signals_all: float32 array (n_days, n_stocks, 15), NaN where no data.
    returns_all: float32 array (n_days, n_stocks), NaN where no data.
    """
    rows = universe.to_dict("records") if limit is None else universe.head(limit).to_dict("records")
    per_stock_signals: dict[str, pd.DataFrame] = {}
    per_stock_returns: dict[str, pd.Series] = {}

    t0 = time.time()
    for i, row in enumerate(rows, 1):
        ticker = row["yf_ticker"]
        df = download_ohlcv(ticker, allow_download=allow_download)
        if df is None or len(df) < MIN_ROWS:
            log.info("[%4d/%4d] SKIP  %-15s (insufficient data)", i, len(rows), ticker)
            continue
        signals, returns = get_signals_and_returns(ticker, df, use_cache=True)
        per_stock_signals[ticker] = signals
        per_stock_returns[ticker] = returns
        if i % 25 == 0 or i == len(rows):
            log.info("[%4d/%4d] processed (%.1fs elapsed)", i, len(rows), time.time() - t0)

    tickers = sorted(per_stock_signals.keys())
    if not tickers:
        raise RuntimeError("No tickers produced usable data — check network/universe CSV.")

    master_dates = sorted(set().union(*(s.index for s in per_stock_signals.values())))
    master_dates = pd.DatetimeIndex(master_dates)
    n_days, n_stocks = len(master_dates), len(tickers)
    log.info("Master calendar: %d trading days x %d stocks", n_days, n_stocks)

    signals_all = np.full((n_days, n_stocks, N_INDICATORS), np.nan, dtype=np.float32)
    returns_all = np.full((n_days, n_stocks), np.nan, dtype=np.float32)

    for j, ticker in enumerate(tickers):
        sig_aligned = per_stock_signals[ticker].reindex(master_dates)
        ret_aligned = per_stock_returns[ticker].reindex(master_dates)
        signals_all[:, j, :] = sig_aligned.to_numpy(dtype=np.float32)
        returns_all[:, j] = ret_aligned.to_numpy(dtype=np.float32)

    return signals_all, returns_all, master_dates, tickers


# --------------------------------------------------------------------------- #
# 5. Combination generation + vectorized backtest
# --------------------------------------------------------------------------- #

def generate_combinations(sizes: tuple[int, ...] = (1, 2, 3, 4)) -> list[tuple[int, ...]]:
    """All combinations of indicator INDICES (not names) for the given sizes."""
    idx = range(N_INDICATORS)
    combos = []
    for r in sizes:
        combos.extend(itertools.combinations(idx, r))
    return combos


def combo_signal(signal_slice: np.ndarray, logic: str) -> np.ndarray:
    """signal_slice: (days, stocks, k) -> combined signal (days, stocks) in {-1,0,1}."""
    k = signal_slice.shape[-1]
    buy_count = np.sum(signal_slice == 1, axis=-1)
    sell_count = np.sum(signal_slice == -1, axis=-1)
    if logic == "AND":
        buy = buy_count == k
        sell = sell_count == k
    else:  # MAJORITY
        buy = buy_count > k / 2
        sell = sell_count > k / 2
    out = np.zeros(signal_slice.shape[:-1], dtype=np.float32)
    out[buy] = 1
    out[sell] = -1
    return out


def _max_drawdown(equity: np.ndarray) -> float:
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    return float(np.nanmin(drawdown)) if drawdown.size else 0.0


def compute_metrics(strat_ret: np.ndarray, position: np.ndarray) -> dict:
    """Compute backtest metrics for one 1-D strategy daily-return series.
    NaN entries (no data that day) are treated as flat/zero return for compounding
    purposes but excluded from win-rate / mean / std statistics."""
    valid = ~np.isnan(strat_ret)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return dict(total_return=np.nan, cagr=np.nan, sharpe=np.nan, win_rate=np.nan,
                     profit_factor=np.nan, max_drawdown=np.nan, num_trades=0, n_days=0)

    ret_filled = np.nan_to_num(strat_ret, nan=0.0)
    # A short position colliding with an extreme single-day data artifact (e.g. an
    # unadjusted demerger/bonus issue) can otherwise push a day's factor negative,
    # which makes (1+total_return) negative and raises a complex number when taken
    # to a fractional power for CAGR. Floor each day's equity multiplier at 0 — you
    # can't lose more than 100% of a position in one day in reality.
    day_factor = np.clip(1.0 + ret_filled, 0.0, None)
    equity = np.cumprod(day_factor)
    total_return = float(equity[-1] - 1.0)
    years = n_valid / TRADING_DAYS_PER_YEAR
    base = 1.0 + total_return
    cagr = float(base ** (1.0 / years) - 1.0) if years > 0 and base >= 0 else np.nan

    active_ret = strat_ret[valid]
    mean_r, std_r = np.mean(active_ret), np.std(active_ret)
    sharpe = float(mean_r / std_r * np.sqrt(TRADING_DAYS_PER_YEAR)) if std_r > 0 else np.nan

    pos_valid = position[valid] != 0
    active_trade_ret = active_ret[pos_valid]
    win_rate = float(np.mean(active_trade_ret > 0)) if active_trade_ret.size else np.nan

    gains = active_trade_ret[active_trade_ret > 0].sum()
    losses = active_trade_ret[active_trade_ret < 0].sum()
    profit_factor = float(gains / abs(losses)) if losses != 0 else np.nan

    pos_filled = np.nan_to_num(position, nan=0.0)
    entries = (pos_filled != 0) & (np.roll(pos_filled, 1) != pos_filled)
    entries[0] = pos_filled[0] != 0
    num_trades = int(entries.sum())

    max_dd = _max_drawdown(equity)

    return dict(total_return=total_return, cagr=cagr, sharpe=sharpe, win_rate=win_rate,
                 profit_factor=profit_factor, max_drawdown=max_dd,
                 num_trades=num_trades, n_days=n_valid)


def run_backtest(signals_all: np.ndarray, returns_all: np.ndarray,
                  master_dates: pd.DatetimeIndex, tickers: list[str],
                  combos: list[tuple[int, ...]] | None = None) -> pd.DataFrame:
    """Backtest every indicator combination (1..4 indicators) under AND and
    MAJORITY logic, vectorized across (days, stocks) per combo. Returns a long
    DataFrame with one row per (combo, logic, scope) where scope is a ticker
    or 'OVERALL'."""
    if combos is None:
        combos = generate_combinations()
    log.info("Backtesting %d combinations x %d logics x (%d stocks + overall)",
              len(combos), len(LOGICS), len(tickers))

    rows = []
    t0 = time.time()
    for ci, idx in enumerate(combos, 1):
        combo_str = "+".join(INDICATOR_NAMES[i] for i in idx)
        n_ind = len(idx)
        sig_slice = signals_all[:, :, idx]  # (days, stocks, k)

        for logic in LOGICS:
            combo_sig = combo_signal(sig_slice, logic)        # (days, stocks)
            position = np.roll(combo_sig, 1, axis=0)
            position[0, :] = 0
            strat_ret = position * returns_all                # (days, stocks)

            # Per-stock rows
            for j, ticker in enumerate(tickers):
                m = compute_metrics(strat_ret[:, j], position[:, j])
                rows.append({"combo": combo_str, "n_indicators": n_ind, "logic": logic,
                             "scope": ticker, **m})

            # Overall/aggregated row: equal-weighted cross-sectional mean return per day
            agg_ret = np.nanmean(strat_ret, axis=1)
            agg_pos = np.nanmean(position, axis=1)  # nonzero if any stock is active that day
            m = compute_metrics(agg_ret, agg_pos)
            rows.append({"combo": combo_str, "n_indicators": n_ind, "logic": logic,
                         "scope": "OVERALL", **m})

        if ci % 100 == 0 or ci == len(combos):
            elapsed = time.time() - t0
            log.info("[%4d/%4d] combinations backtested (%.1fs elapsed)", ci, len(combos), elapsed)

    results = pd.DataFrame(rows)
    for col in ("combo", "logic", "scope"):
        results[col] = results[col].astype("category")
    for col in ("total_return", "cagr", "sharpe", "win_rate", "profit_factor", "max_drawdown"):
        results[col] = results[col].astype("float32")
    results["n_indicators"] = results["n_indicators"].astype("int8")
    results["num_trades"] = results["num_trades"].astype("int32")
    results["n_days"] = results["n_days"].astype("int32")
    return results


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Indicator-combination backtest engine")
    parser.add_argument("--no-download", action="store_true",
                         help="Reuse cached OHLCV/signals only; do not hit the network")
    parser.add_argument("--quick", type=int, default=None,
                         help="Smoke-test on only the first N tickers in the universe")
    args = parser.parse_args()

    universe = load_universe()
    n_combos = len(generate_combinations())
    expected = sum(__import__("math").comb(N_INDICATORS, r) for r in (1, 2, 3, 4))
    assert n_combos == expected, f"Combination count mismatch: {n_combos} != {expected}"
    log.info("Indicator pool: %d | Combinations to test: %d", N_INDICATORS, n_combos)

    signals_all, returns_all, master_dates, tickers = build_master_arrays(
        universe, allow_download=not args.no_download, limit=args.quick,
    )

    results = run_backtest(signals_all, returns_all, master_dates, tickers)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results.to_parquet(RESULTS_PATH, index=False)
    log.info("Wrote %d result rows to %s", len(results), RESULTS_PATH)


if __name__ == "__main__":
    sys.exit(main())
