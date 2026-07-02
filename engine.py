"""
engine.py — Indicator-combination backtesting engine for the Nifty Master Universe.

Pipeline
--------
1. Load the stock universe from data/universe/master_stock_universe.csv and map
   each NSE symbol to its Yahoo Finance ticker (SYMBOL.NS).
2. Download (or load from cache) daily OHLCV for every ticker, 2009-01-01 .. END_DATE.
3. Compute 23 indicator signal series (-1 / 0 / +1) per stock, ONCE, and cache them.
4. Align every stock's signals/returns onto one master trading-day calendar and
   stack into 3-D numpy arrays: signals_all (days, stocks, 23), returns_all (days, stocks).
5. Enumerate all C(23,1)+C(23,2)+C(23,3)+C(23,4) = 10,902 indicator combinations,
   backtest each under AND and MAJORITY-vote logic using pure numpy ops over the
   full (days, stocks) grid at once (no per-stock / per-combo Python loops over rows),
   and write per-stock + overall-market metrics to data/backtest_results.parquet.

Signal design
-------------
Moving averages (SMA9/20/30/100/200, EMA9/20/30/100/200):
    +1 when close > MA,  -1 when close < MA  — always directional.

Trend (MACD, PSAR, ROC, SUPERTREND):
    Already always directional; no neutral zone except warm-up period.

Momentum oscillators (RSI, STOCH, CCI):
    Momentum interpretation — above midpoint (50 / 50 / 0) = bullish, below = bearish.
    This generates a signal every trading day, unlike the classic overbought/oversold
    mode which fires on only ~5-10% of days (producing far too few trades to evaluate).

Volatility (BBANDS, ATR):
    Momentum breakout: fires +1 on strong upside moves, -1 on strong downside moves,
    0 in quiet/inside-band conditions.

Volume (OBV, CMF, VRSI):
    All directional with respect to volume pressure direction.

Cloud (ICHIMOKU):
    +1 above cloud, -1 below cloud, 0 inside cloud (genuine neutral zone).

Run:
    python engine.py                  # full pipeline
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

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")


# --------------------------------------------------------------------------- #
# 1. Universe loading
# --------------------------------------------------------------------------- #

def load_universe(csv_path: Path = UNIVERSE_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["yf_ticker"] = df["Symbol"].str.strip().str.replace("&", "%26", regex=False) + ".NS"
    log.info("Loaded universe: %d stocks from %s", len(df), csv_path.name)
    return df


# --------------------------------------------------------------------------- #
# 2. OHLCV download + cache
# --------------------------------------------------------------------------- #

def download_ohlcv(ticker: str, start: str = START_DATE, end: str = END_DATE,
                    allow_download: bool = True) -> pd.DataFrame | None:
    cache_path = OHLCV_CACHE_DIR / f"{ticker.replace('.', '_')}.parquet"

    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as exc:
            log.warning("Cache read failed for %s (%s); redownloading", ticker, exc)

    if not allow_download:
        log.warning("No cache for %s and downloads disabled (--no-download)", ticker)
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
# 3. Indicator signal functions  (-1 / 0 / +1)
# --------------------------------------------------------------------------- #

def _rsi_value(series: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI, reused for price-RSI and Volume-RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def sig_sma(c: pd.Series, period: int) -> pd.Series:
    """Price vs simple moving average: +1 above MA, −1 below, 0 during warm-up."""
    ma = c.rolling(period, min_periods=period).mean()
    return np.sign(c - ma).fillna(0)


def sig_ema(c: pd.Series, period: int) -> pd.Series:
    """Price vs exponential moving average: +1 above MA, −1 below."""
    ma = c.ewm(span=period, adjust=False, min_periods=period).mean()
    return np.sign(c - ma).fillna(0)


def sig_macd(c: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD line vs signal line — always directional once warm-up completes."""
    macd_line = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return np.sign(macd_line - signal_line)


def sig_psar(h: pd.Series, l: pd.Series, c: pd.Series,
             step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    """Parabolic SAR — sequential O(n) loop computed once per stock."""
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


def sig_roc(c: pd.Series, window: int = 12) -> pd.Series:
    """Rate of change: +1 if price is higher than N days ago, −1 if lower."""
    roc = (c - c.shift(window)) / c.shift(window) * 100
    return np.sign(roc)


def sig_supertrend(h: pd.Series, l: pd.Series, c: pd.Series,
                    window: int = 10, mult: float = 3.0) -> pd.Series:
    """SuperTrend — sequential O(n) loop computed once per stock."""
    a = _atr_value(h, l, c, window)
    hl2 = (h + l) / 2
    # .copy() avoids read-only view errors with Arrow-backed pandas 3.x dtypes
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


def sig_rsi(c: pd.Series, window: int = 14) -> pd.Series:
    """RSI in momentum mode: +1 when RSI > 50, −1 when RSI < 50.
    Generates a signal every day; never neutral except at exactly 50."""
    r = _rsi_value(c, window)
    return np.sign(r - 50)


def sig_stoch(h: pd.Series, l: pd.Series, c: pd.Series, window: int = 14) -> pd.Series:
    """%K in momentum mode: +1 when %K > 50, −1 when %K < 50."""
    lowest = l.rolling(window).min()
    highest = h.rolling(window).max()
    pct_k = 100 * (c - lowest) / (highest - lowest).replace(0, np.nan)
    return np.sign(pct_k - 50)


def sig_cci(h: pd.Series, l: pd.Series, c: pd.Series, window: int = 20) -> pd.Series:
    """CCI in momentum mode: +1 when CCI > 0, −1 when CCI < 0."""
    tp = (h + l + c) / 3
    sma = tp.rolling(window).mean()
    mad = tp.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma) / (0.015 * mad.replace(0, np.nan))
    return np.sign(cci)


def sig_bbands(c: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    """Bollinger Band momentum breakout: +1 above upper band, −1 below lower band.
    Inside the bands = 0 (no clear signal). Fires on ~5-10% of days."""
    mid = c.rolling(window).mean()
    std = c.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    sig = pd.Series(0.0, index=c.index)
    sig[c > upper] = 1    # momentum breakout above band
    sig[c < lower] = -1   # momentum breakdown below band
    return sig


def _atr_value(h: pd.Series, l: pd.Series, c: pd.Series, window: int = 14) -> pd.Series:
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def sig_atr(h: pd.Series, l: pd.Series, c: pd.Series,
            window: int = 14, mult: float = 1.0) -> pd.Series:
    """Volatility breakout: +1 when day's move > mult×ATR, −1 when < −mult×ATR."""
    a = _atr_value(h, l, c, window)
    move = c.diff()
    sig = pd.Series(0.0, index=c.index)
    sig[move > mult * a] = 1
    sig[move < -mult * a] = -1
    return sig


def sig_obv(c: pd.Series, v: pd.Series, window: int = 20) -> pd.Series:
    """OBV vs its 20-day MA: +1 above (accumulation), −1 below (distribution)."""
    direction = np.sign(c.diff()).fillna(0)
    obv = (direction * v).cumsum()
    return np.sign(obv - obv.rolling(window).mean())


def sig_cmf(h: pd.Series, l: pd.Series, c: pd.Series, v: pd.Series,
            window: int = 20) -> pd.Series:
    """Chaikin Money Flow: positive = buying pressure, negative = selling."""
    mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    mfv = mfm * v
    cmf = mfv.rolling(window).sum() / v.rolling(window).sum()
    return np.sign(cmf)


def sig_vrsi(v: pd.Series, window: int = 14) -> pd.Series:
    """Volume RSI in momentum mode: +1 when Volume RSI > 50, −1 when < 50."""
    r = _rsi_value(v.astype(float), window)
    return np.sign(r - 50)


def sig_ichimoku(h: pd.Series, l: pd.Series, c: pd.Series,
                  conv: int = 9, base: int = 26, span_b: int = 52,
                  displacement: int = 26) -> pd.Series:
    """Ichimoku Cloud: +1 above cloud, −1 below, 0 inside (genuine neutral)."""
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
    """Compute all 23 indicator signals (−1/0/+1) for one stock's OHLCV frame."""
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
    out = pd.DataFrame(index=df.index)

    # Moving averages (price vs MA)
    out["SMA9"]   = sig_sma(c, 9)
    out["SMA20"]  = sig_sma(c, 20)
    out["SMA30"]  = sig_sma(c, 30)
    out["SMA100"] = sig_sma(c, 100)
    out["SMA200"] = sig_sma(c, 200)
    out["EMA9"]   = sig_ema(c, 9)
    out["EMA20"]  = sig_ema(c, 20)
    out["EMA30"]  = sig_ema(c, 30)
    out["EMA100"] = sig_ema(c, 100)
    out["EMA200"] = sig_ema(c, 200)

    # Trend
    out["MACD"]       = sig_macd(c)
    out["PSAR"]       = sig_psar(h, l, c)
    out["ROC"]        = sig_roc(c)
    out["SUPERTREND"] = sig_supertrend(h, l, c)

    # Momentum (midpoint threshold — always directional)
    out["RSI"]   = sig_rsi(c)
    out["STOCH"] = sig_stoch(h, l, c)
    out["CCI"]   = sig_cci(h, l, c)

    # Volatility
    out["BBANDS"] = sig_bbands(c)
    out["ATR"]    = sig_atr(h, l, c)

    # Volume
    out["OBV"]  = sig_obv(c, v)
    out["CMF"]  = sig_cmf(h, l, c, v)
    out["VRSI"] = sig_vrsi(v)

    # Cloud
    out["ICHIMOKU"] = sig_ichimoku(h, l, c)

    return out[INDICATOR_NAMES]


def get_signals_and_returns(ticker: str, df: pd.DataFrame, use_cache: bool = True
                             ) -> tuple[pd.DataFrame, pd.Series]:
    """Compute (or load cached) indicator signals + daily returns for one stock.
    Automatically recomputes if the cached file is missing any of the current
    INDICATOR_NAMES (handles the case where the indicator set has changed)."""
    cache_path = SIGNALS_CACHE_DIR / f"{ticker.replace('.', '_')}.parquet"
    if use_cache and cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            cached.index = pd.to_datetime(cached.index)
            # Validate all expected columns are present (cache invalidation on schema change)
            required = set(INDICATOR_NAMES) | {"return"}
            if required.issubset(set(cached.columns)):
                return cached[INDICATOR_NAMES], cached["return"]
            log.info("Signal cache for %s is stale (indicator set changed); recomputing", ticker)
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

    signals_all : float32 (n_days, n_stocks, N_INDICATORS), NaN where no data.
    returns_all : float32 (n_days, n_stocks), NaN where no data.
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
    idx = range(N_INDICATORS)
    combos = []
    for r in sizes:
        combos.extend(itertools.combinations(idx, r))
    return combos


def combo_signal(signal_slice: np.ndarray, logic: str) -> np.ndarray:
    """signal_slice: (days, stocks, k) → combined signal (days, stocks) in {−1, 0, +1}.

    AND: all k signals must unanimously agree on the same non-zero direction.
    MAJORITY: strict majority (> k/2) of signals must agree.
    """
    k = signal_slice.shape[-1]
    buy_count  = np.sum(signal_slice == 1,  axis=-1)
    sell_count = np.sum(signal_slice == -1, axis=-1)
    if logic == "AND":
        buy  = buy_count == k
        sell = sell_count == k
    else:  # MAJORITY
        buy  = buy_count  > k / 2
        sell = sell_count > k / 2
    out = np.zeros(signal_slice.shape[:-1], dtype=np.float32)
    out[buy]  = 1
    out[sell] = -1
    return out


def _max_drawdown(equity: np.ndarray) -> float:
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    return float(np.nanmin(drawdown)) if drawdown.size else 0.0


def compute_metrics(strat_ret: np.ndarray, position: np.ndarray) -> dict:
    """Compute backtest metrics for one 1-D daily-return series.

    NaN entries (no data that day) are treated as flat/zero return for compounding
    but excluded from win-rate / Sharpe / profit-factor statistics.
    """
    valid = ~np.isnan(strat_ret)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return dict(total_return=np.nan, cagr=np.nan, sharpe=np.nan, win_rate=np.nan,
                     profit_factor=np.nan, max_drawdown=np.nan, num_trades=0, n_days=0,
                     pct_in_market=np.nan, avg_hold_days=np.nan)

    ret_filled = np.nan_to_num(strat_ret, nan=0.0)
    # Floor day-factor at 0: can't lose more than 100% in one day (guards against
    # unadjusted demerger/bonus artifacts that produce negative equity → complex CAGR).
    day_factor = np.clip(1.0 + ret_filled, 0.0, None)
    equity = np.cumprod(day_factor)
    total_return = float(equity[-1] - 1.0)
    years = n_valid / TRADING_DAYS_PER_YEAR
    base = 1.0 + total_return
    cagr = float(base ** (1.0 / years) - 1.0) if years > 0 and base >= 0 else np.nan

    active_ret = strat_ret[valid]
    mean_r, std_r = np.mean(active_ret), np.std(active_ret)
    sharpe = float(mean_r / std_r * np.sqrt(TRADING_DAYS_PER_YEAR)) if std_r > 0 else np.nan

    pos_valid      = position[valid] != 0
    active_trade_ret = active_ret[pos_valid]
    win_rate = float(np.mean(active_trade_ret > 0)) if active_trade_ret.size else np.nan

    gains  = active_trade_ret[active_trade_ret > 0].sum()
    losses = active_trade_ret[active_trade_ret < 0].sum()
    profit_factor = float(gains / abs(losses)) if losses != 0 else np.nan

    pos_filled = np.nan_to_num(position, nan=0.0)
    entries    = (pos_filled != 0) & (np.roll(pos_filled, 1) != pos_filled)
    entries[0] = pos_filled[0] != 0
    num_trades = int(entries.sum())

    max_dd = _max_drawdown(equity)

    n_in_market   = int(pos_valid.sum())
    pct_in_market = n_in_market / n_valid if n_valid > 0 else np.nan
    avg_hold_days = n_in_market / num_trades if num_trades > 0 else np.nan

    return dict(total_return=total_return, cagr=cagr, sharpe=sharpe, win_rate=win_rate,
                 profit_factor=profit_factor, max_drawdown=max_dd,
                 num_trades=num_trades, n_days=n_valid,
                 pct_in_market=pct_in_market, avg_hold_days=avg_hold_days)


def run_backtest(signals_all: np.ndarray, returns_all: np.ndarray,
                  master_dates: pd.DatetimeIndex, tickers: list[str],
                  combos: list[tuple[int, ...]] | None = None) -> pd.DataFrame:
    """Backtest every indicator combination under AND and MAJORITY logic, vectorized
    across (days, stocks) per combo. Returns a long DataFrame with one row per
    (combo, logic, scope) where scope is a ticker or 'OVERALL'.

    Also appends a BUY_HOLD benchmark row (scope=OVERALL and per-stock) with
    combo='BUY_HOLD', logic='NONE', n_indicators=0 — for dashboard comparison.
    """
    if combos is None:
        combos = generate_combinations()
    log.info("Backtesting %d combinations × %d logics × (%d stocks + overall)",
              len(combos), len(LOGICS), len(tickers))

    rows = []
    t0 = time.time()

    for ci, idx in enumerate(combos, 1):
        combo_str = "+".join(INDICATOR_NAMES[i] for i in idx)
        n_ind     = len(idx)
        sig_slice = signals_all[:, :, idx]   # (days, stocks, k)

        for logic in LOGICS:
            combo_sig = combo_signal(sig_slice, logic)   # (days, stocks)
            position  = np.roll(combo_sig, 1, axis=0)
            position[0, :] = 0
            strat_ret = position * returns_all            # (days, stocks)

            for j, ticker in enumerate(tickers):
                m = compute_metrics(strat_ret[:, j], position[:, j])
                rows.append({"combo": combo_str, "n_indicators": n_ind,
                             "logic": logic, "scope": ticker, **m})

            # Overall: equal-weighted cross-sectional mean return per day
            agg_ret = np.nanmean(strat_ret, axis=1)
            agg_pos = np.nanmean(position,  axis=1)
            m = compute_metrics(agg_ret, agg_pos)
            rows.append({"combo": combo_str, "n_indicators": n_ind,
                         "logic": logic, "scope": "OVERALL", **m})

        if ci % 200 == 0 or ci == len(combos):
            elapsed = time.time() - t0
            log.info("[%5d/%5d] combinations backtested (%.1fs)", ci, len(combos), elapsed)

    # ---- Buy-and-hold benchmark ---- #
    log.info("Computing buy-and-hold benchmark...")
    bh_pos = np.ones_like(returns_all, dtype=np.float32)
    for j, ticker in enumerate(tickers):
        m = compute_metrics(returns_all[:, j], bh_pos[:, j])
        rows.append({"combo": "BUY_HOLD", "n_indicators": 0,
                     "logic": "NONE", "scope": ticker, **m})
    agg_ret = np.nanmean(returns_all, axis=1)
    agg_pos = np.ones(len(agg_ret), dtype=np.float32)
    m = compute_metrics(agg_ret, agg_pos)
    rows.append({"combo": "BUY_HOLD", "n_indicators": 0,
                 "logic": "NONE", "scope": "OVERALL", **m})

    results = pd.DataFrame(rows)
    for col in ("combo", "logic", "scope"):
        results[col] = results[col].astype("category")
    for col in ("total_return", "cagr", "sharpe", "win_rate", "profit_factor",
                "max_drawdown", "pct_in_market", "avg_hold_days"):
        results[col] = results[col].astype("float32")
    results["n_indicators"] = results["n_indicators"].astype("int8")
    results["num_trades"]   = results["num_trades"].astype("int32")
    results["n_days"]       = results["n_days"].astype("int32")
    return results


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Indicator-combination backtest engine")
    parser.add_argument("--no-download", action="store_true",
                         help="Reuse cached OHLCV/signals; do not hit the network")
    parser.add_argument("--quick", type=int, default=None,
                         help="Smoke-test on only the first N tickers in the universe")
    args = parser.parse_args()

    universe = load_universe()
    combos = generate_combinations()
    import math
    expected = sum(math.comb(N_INDICATORS, r) for r in (1, 2, 3, 4))
    assert len(combos) == expected, f"Combination count mismatch: {len(combos)} != {expected}"
    log.info("Indicator pool: %d | Combinations to test: %d", N_INDICATORS, len(combos))

    signals_all, returns_all, master_dates, tickers = build_master_arrays(
        universe, allow_download=not args.no_download, limit=args.quick,
    )

    results = run_backtest(signals_all, returns_all, master_dates, tickers, combos)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results.to_parquet(RESULTS_PATH, index=False)
    log.info("Wrote %d result rows to %s", len(results), RESULTS_PATH)


if __name__ == "__main__":
    sys.exit(main())
