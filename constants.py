"""
constants.py — shared config for engine.py and app.py.

No heavy dependencies — importable by the Streamlit dashboard without pulling
in yfinance/numpy from engine.py.
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

MIN_ROWS = 100
TRADING_DAYS_PER_YEAR = 252

# 23 indicators — SMA & EMA each offered at 5 timeframes; oscillators use
# momentum interpretation (above/below midpoint) so they always produce a
# directional signal rather than firing only at extremes.
INDICATOR_NAMES = [
    # Moving averages: price vs MA (+1 above, −1 below)
    "SMA9", "SMA20", "SMA30", "SMA100", "SMA200",
    "EMA9", "EMA20", "EMA30", "EMA100", "EMA200",
    # Trend (always directional)
    "MACD", "PSAR", "ROC", "SUPERTREND",
    # Momentum — RSI/STOCH/CCI above/below their midpoints
    "RSI", "STOCH", "CCI",
    # Volatility — breakout/breakdown signals
    "BBANDS", "ATR",
    # Volume
    "OBV", "CMF", "VRSI",
    # Multi-factor
    "ICHIMOKU",
]
N_INDICATORS = len(INDICATOR_NAMES)
LOGICS = ("AND", "MAJORITY")

# --------------------------------------------------------------------------- #
# Indicator metadata — used by the dashboard's Indicator Guide tab and the
# Strategy Explorer's per-indicator description panel.
# --------------------------------------------------------------------------- #
INDICATOR_INFO = {
    "SMA9": {
        "category": "Moving Average",
        "full_name": "9-Day Simple Moving Average",
        "signal": "+1 when Close > SMA(9)  ·  −1 when Close < SMA(9)",
        "desc": (
            "The fastest SMA in this set. Reacts quickly to price changes — "
            "used by swing traders to spot very short-term momentum. Being above "
            "the 9-SMA means price is running ahead of its recent 2-week average."
        ),
        "params": "Period: 9 days",
    },
    "SMA20": {
        "category": "Moving Average",
        "full_name": "20-Day Simple Moving Average",
        "signal": "+1 when Close > SMA(20)  ·  −1 when Close < SMA(20)",
        "desc": (
            "Represents ~1 trading month. One of the most commonly watched levels "
            "by retail and institutional traders alike. Price holding above SMA20 "
            "is considered short-term bullish."
        ),
        "params": "Period: 20 days",
    },
    "SMA30": {
        "category": "Moving Average",
        "full_name": "30-Day Simple Moving Average",
        "signal": "+1 when Close > SMA(30)  ·  −1 when Close < SMA(30)",
        "desc": (
            "~1.5 months. Sits between the fast SMA20 and the medium SMA100. "
            "Helps filter out 20-day noise while still being responsive to "
            "medium-term trend changes."
        ),
        "params": "Period: 30 days",
    },
    "SMA100": {
        "category": "Moving Average",
        "full_name": "100-Day Simple Moving Average",
        "signal": "+1 when Close > SMA(100)  ·  −1 when Close < SMA(100)",
        "desc": (
            "Major institutional support/resistance. Breaching SMA100 in either "
            "direction often triggers fund-level buy or sell programs. Strong "
            "medium-to-long-term trend filter."
        ),
        "params": "Period: 100 days",
    },
    "SMA200": {
        "category": "Moving Average",
        "full_name": "200-Day Simple Moving Average",
        "signal": "+1 when Close > SMA(200)  ·  −1 when Close < SMA(200)",
        "desc": (
            "The most-watched moving average on Wall Street and Dalal Street. "
            "Price above 200-SMA = secular bull market. Below = bear market. "
            "Golden cross (SMA50 > SMA200) and death cross are defined by this line."
        ),
        "params": "Period: 200 days",
    },
    "EMA9": {
        "category": "Moving Average",
        "full_name": "9-Day Exponential Moving Average",
        "signal": "+1 when Close > EMA(9)  ·  −1 when Close < EMA(9)",
        "desc": (
            "EMA weights recent prices more heavily than SMA — so it reacts faster. "
            "The 9-EMA is extremely sensitive; used by day traders and scalpers for "
            "real-time momentum confirmation."
        ),
        "params": "Span: 9 days",
    },
    "EMA20": {
        "category": "Moving Average",
        "full_name": "20-Day Exponential Moving Average",
        "signal": "+1 when Close > EMA(20)  ·  −1 when Close < EMA(20)",
        "desc": (
            "Popular with swing traders. EMA20 hugs price more closely than SMA20 "
            "during trending phases, giving earlier signal changes while still "
            "filtering out minor intraday noise."
        ),
        "params": "Span: 20 days",
    },
    "EMA30": {
        "category": "Moving Average",
        "full_name": "30-Day Exponential Moving Average",
        "signal": "+1 when Close > EMA(30)  ·  −1 when Close < EMA(30)",
        "desc": (
            "Medium-term EMA. Provides a balance between responsiveness and "
            "smoothness. When combined with EMA9/20, crossovers of EMA30 confirm "
            "that a short-term move has medium-term backing."
        ),
        "params": "Span: 30 days",
    },
    "EMA100": {
        "category": "Moving Average",
        "full_name": "100-Day Exponential Moving Average",
        "signal": "+1 when Close > EMA(100)  ·  −1 when Close < EMA(100)",
        "desc": (
            "Long-term EMA. Reacts somewhat faster than SMA100 to recent price "
            "action. Useful as a dynamic support/resistance for position traders."
        ),
        "params": "Span: 100 days",
    },
    "EMA200": {
        "category": "Moving Average",
        "full_name": "200-Day Exponential Moving Average",
        "signal": "+1 when Close > EMA(200)  ·  −1 when Close < EMA(200)",
        "desc": (
            "Long-term trend anchor with reduced lag vs SMA200. Institutions use "
            "EMA200 as an alternative to SMA200 to get slightly earlier signals on "
            "major trend changes."
        ),
        "params": "Span: 200 days",
    },
    "MACD": {
        "category": "Trend",
        "full_name": "Moving Average Convergence Divergence",
        "signal": "+1 when MACD line > signal line  ·  −1 when MACD < signal",
        "desc": (
            "Measures momentum by comparing two EMAs (12-day vs 26-day). When the "
            "faster EMA pulls ahead of the slower, momentum is building. The signal "
            "line (9-day EMA of MACD) acts as a trigger. Always directional — no "
            "neutral zone."
        ),
        "params": "Fast EMA: 12 · Slow EMA: 26 · Signal EMA: 9",
    },
    "PSAR": {
        "category": "Trend",
        "full_name": "Parabolic SAR (Stop and Reverse)",
        "signal": "+1 when Close > SAR  ·  −1 when Close < SAR",
        "desc": (
            "A trailing stop that flips when price reverses. Dots below price = "
            "uptrend; dots above = downtrend. Parabolic shape means it tightens "
            "as the trend matures — captures most of a trend while limiting "
            "give-back."
        ),
        "params": "Acceleration: 0.02 · Max acceleration: 0.2",
    },
    "ROC": {
        "category": "Trend",
        "full_name": "Rate of Change (12-day)",
        "signal": "+1 when ROC > 0  ·  −1 when ROC < 0",
        "desc": (
            "Simply measures the % price change over the last 12 days. Positive ROC "
            "means the stock is higher than it was 12 days ago — upward momentum. "
            "Always directional; strong complement to MA-based signals."
        ),
        "params": "Period: 12 days",
    },
    "SUPERTREND": {
        "category": "Trend",
        "full_name": "SuperTrend",
        "signal": "+1 in uptrend  ·  −1 in downtrend",
        "desc": (
            "An ATR-based trailing band system popularised in Indian retail markets. "
            "Price closing above the upper band locks in a downtrend flip to uptrend "
            "(and vice versa). Very popular on NSE/BSE charts."
        ),
        "params": "ATR period: 10 · Multiplier: 3.0×",
    },
    "RSI": {
        "category": "Momentum",
        "full_name": "Relative Strength Index (momentum mode)",
        "signal": "+1 when RSI > 50  ·  −1 when RSI < 50",
        "desc": (
            "RSI compares average up-closes to average down-closes over 14 days. "
            "Used here in momentum mode: RSI > 50 means recent gains dominate — "
            "bullish. RSI < 50 means losses dominate — bearish. This generates a "
            "signal every day, unlike the classic 30/70 overbought-oversold mode."
        ),
        "params": "Period: 14 days · Midpoint: 50",
    },
    "STOCH": {
        "category": "Momentum",
        "full_name": "Stochastic Oscillator (momentum mode)",
        "signal": "+1 when %K > 50  ·  −1 when %K < 50",
        "desc": (
            "Measures where today's close sits within the last 14 days' high–low "
            "range. %K > 50 means closing in the upper half of the range — "
            "buying pressure. Used here in momentum mode (midpoint 50) rather than "
            "the classic overbought/oversold extremes."
        ),
        "params": "Period: 14 days · Midpoint: 50",
    },
    "CCI": {
        "category": "Momentum",
        "full_name": "Commodity Channel Index (momentum mode)",
        "signal": "+1 when CCI > 0  ·  −1 when CCI < 0",
        "desc": (
            "Measures how far the typical price (H+L+C/3) has moved from its 20-day "
            "average, normalised by average deviation. CCI > 0 = above-average price "
            "= bullish; CCI < 0 = below-average = bearish. Using 0 as threshold "
            "(not ±100) gives a signal every day."
        ),
        "params": "Period: 20 days · Threshold: 0",
    },
    "BBANDS": {
        "category": "Volatility",
        "full_name": "Bollinger Band Breakout",
        "signal": "+1 above upper band  ·  −1 below lower band  ·  0 inside bands",
        "desc": (
            "Bollinger Bands envelope ±2 standard deviations around the 20-day SMA. "
            "Used here as a MOMENTUM BREAKOUT signal: closing above the upper band "
            "signals strong buying pressure; below the lower band signals strong "
            "selling. Fires on roughly 5–10% of days."
        ),
        "params": "Period: 20 days · Bands: ±2.0σ",
    },
    "ATR": {
        "category": "Volatility",
        "full_name": "ATR Volatility Breakout",
        "signal": "+1 when day's move > +1×ATR  ·  −1 when move < −1×ATR",
        "desc": (
            "Fires only when today's price change exceeds the 14-day Average True "
            "Range — i.e., an abnormally large move. This captures volatility "
            "expansion / momentum continuation. Good complement to trend indicators "
            "to confirm a breakout is real."
        ),
        "params": "ATR period: 14 · Multiplier: 1.0×",
    },
    "OBV": {
        "category": "Volume",
        "full_name": "On-Balance Volume",
        "signal": "+1 when OBV > its 20-day MA  ·  −1 when OBV < MA",
        "desc": (
            "Accumulates volume on up-days and subtracts on down-days. Rising OBV "
            "means more volume on up-days = smart money buying. OBV above its "
            "20-day MA = sustained accumulation phase. 'Volume precedes price.'"
        ),
        "params": "OBV smoothing MA: 20 days",
    },
    "CMF": {
        "category": "Volume",
        "full_name": "Chaikin Money Flow",
        "signal": "+1 when CMF > 0 (buying pressure)  ·  −1 when CMF < 0",
        "desc": (
            "Weights volume by where price closes within the day's high–low range. "
            "Closing near the high = buying pressure; near the low = selling pressure. "
            "Positive CMF = institutional accumulation. Negative = distribution."
        ),
        "params": "Period: 20 days",
    },
    "VRSI": {
        "category": "Volume",
        "full_name": "Volume RSI",
        "signal": "+1 when Volume RSI > 50  ·  −1 when Volume RSI < 50",
        "desc": (
            "RSI applied to the volume series rather than price. Rising Volume RSI "
            "means recent volume surges dominate — activity is accelerating. When "
            "aligned with price direction, it confirms that a price move is backed "
            "by real participation."
        ),
        "params": "Period: 14 days · Midpoint: 50",
    },
    "ICHIMOKU": {
        "category": "Cloud",
        "full_name": "Ichimoku Cloud",
        "signal": "+1 above cloud (bullish)  ·  −1 below cloud (bearish)  ·  0 inside (neutral)",
        "desc": (
            "Japanese multi-line system combining trend, momentum, and support/ "
            "resistance. The 'cloud' (kumo) is the area between Senkou Span A and B. "
            "Price above cloud = strong uptrend. Below = downtrend. Inside the cloud "
            "= consolidation / no clear signal."
        ),
        "params": "Tenkan: 9 · Kijun: 26 · Senkou B: 52 · Displacement: 26",
    },
}

INDICATOR_CATEGORIES = {
    "Moving Average": ["SMA9", "SMA20", "SMA30", "SMA100", "SMA200",
                       "EMA9", "EMA20", "EMA30", "EMA100", "EMA200"],
    "Trend":          ["MACD", "PSAR", "ROC", "SUPERTREND"],
    "Momentum":       ["RSI", "STOCH", "CCI"],
    "Volatility":     ["BBANDS", "ATR"],
    "Volume":         ["OBV", "CMF", "VRSI"],
    "Cloud":          ["ICHIMOKU"],
}

CATEGORY_COLORS = {
    "Moving Average": "#1f77b4",
    "Trend":          "#2ca02c",
    "Momentum":       "#ff7f0e",
    "Volatility":     "#d62728",
    "Volume":         "#9467bd",
    "Cloud":          "#8c564b",
}
