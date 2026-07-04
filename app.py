"""
app.py — Streamlit dashboard for the 15-indicator combination backtest.

Every query uses parquet predicate pushdown — only the small slice needed for a
given tab is ever materialised in memory (keeps the app inside free-tier limits).

Tabs
----
🌐 Global Overview   — top combos across the entire universe, vs buy-and-hold
🏢 Stock Analysis    — best combos for a single stock
🧩 Strategy Explorer — build a custom 1–4 indicator combo; see what it means & how it performed
📚 Indicator Guide   — what every indicator measures and how its signal is generated
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from constants import (
    CATEGORY_COLORS, INDICATOR_CATEGORIES, INDICATOR_INFO,
    INDICATOR_NAMES, LOGICS, RESULTS_PATH, UNIVERSE_CSV,
)

st.set_page_config(
    page_title="Indicator Combo Backtester",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------- #
# Global CSS
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
.metric-row  { display:flex; gap:14px; margin:8px 0 4px 0; flex-wrap:wrap; }
.kpi-card    { flex:1; min-width:120px; background:#fff; border:1px solid #e5e4df;
               border-radius:8px; padding:12px 16px; text-align:center; }
.kpi-value   { font-size:21px; font-weight:700; color:#1a1a18; margin:4px 0; }
.kpi-label   { font-size:11px; color:#888780; text-transform:uppercase; letter-spacing:.05em; }
.kpi-delta   { font-size:11px; margin-top:4px; }
.ind-card    { border-radius:8px; padding:14px 16px; margin-bottom:10px; border-left:4px solid; }
.signal-box  { background:#f0f0ee; border-radius:4px; padding:8px 10px;
               font-family:monospace; font-size:12px; margin-top:8px; line-height:1.7; }
.summary-box { background:#f8f7f4; border:1px solid #e5e4df; border-radius:6px;
               padding:10px 14px; margin:6px 0 12px 0; font-size:13px; color:#555; line-height:1.6; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
METRIC_OPTIONS = {
    "CAGR":          "cagr",
    "Sharpe Ratio":  "sharpe",
    "Win Rate":      "win_rate",
    "Total Return":  "total_return",
    "Profit Factor": "profit_factor",
}

DISPLAY_COLS = [
    "combo", "n_indicators", "logic",
    "win_rate", "total_return", "cagr", "sharpe", "profit_factor",
    "max_drawdown", "num_trades", "avg_hold_days", "pct_in_market",
]

COL_LABELS = {
    "combo":        "Combination",
    "n_indicators": "#",
    "logic":        "Logic",
    "win_rate":     "Day Win%",
    "total_return": "Total Ret",
    "cagr":         "CAGR",
    "sharpe":       "Sharpe",
    "profit_factor":"Profit F.",
    "max_drawdown": "Max DD",
    "num_trades":   "Trades",
    "avg_hold_days":"Avg Hold",
    "pct_in_market":"In Mkt%",
}

PCT_COLS    = ["win_rate", "total_return", "cagr", "max_drawdown", "pct_in_market"]
FLOAT2_COLS = ["sharpe", "profit_factor", "avg_hold_days"]


# --------------------------------------------------------------------------- #
# Data loading — all use predicate pushdown
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def load_scopes(path: str) -> list[str]:
    col = pd.read_parquet(path, columns=["scope"])["scope"]
    return sorted(col.astype(str).unique())


@st.cache_data(show_spinner=False)
def load_overall(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path, filters=[("scope", "==", "OVERALL")])
    return _stringify(df)


@st.cache_data(show_spinner=False, max_entries=64)
def load_stock(path: str, ticker: str) -> pd.DataFrame:
    df = pd.read_parquet(path, filters=[("scope", "==", ticker)])
    return _stringify(df)


@st.cache_data(show_spinner=False, max_entries=128)
def load_combo(path: str, combo_str: str, logic: str) -> pd.DataFrame:
    df = pd.read_parquet(path, filters=[("combo", "==", combo_str), ("logic", "==", logic)])
    return _stringify(df)


@st.cache_data(show_spinner=False)
def load_universe_names(path: str) -> dict:
    uni = pd.read_csv(path)
    uni["yf_ticker"] = uni["Symbol"].str.strip().str.replace("&", "%26", regex=False) + ".NS"
    return dict(zip(uni["yf_ticker"], uni["Company Name"]))


def _stringify(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("combo", "logic", "scope"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def ticker_label(ticker: str, name_map: dict) -> str:
    name = name_map.get(ticker, "")
    base = ticker.replace(".NS", "")
    return f"{base} — {name}" if name else base


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def fmt_pct(v) -> str:
    return f"{v:.1%}" if pd.notna(v) else "—"

def fmt_f(v, d=2) -> str:
    return f"{v:.{d}f}" if pd.notna(v) else "—"


def style_table(df: pd.DataFrame):
    fmt = {}
    for c in PCT_COLS:
        if c in df.columns:
            fmt[c] = "{:.1%}"
    for c in FLOAT2_COLS:
        if c in df.columns:
            fmt[c] = "{:.2f}"
    if "num_trades" in df.columns:
        fmt["num_trades"] = "{:.0f}"

    s = (df.rename(columns=COL_LABELS)
           .style
           .format({COL_LABELS.get(k, k): v for k, v in fmt.items()}, na_rep="—"))

    label = COL_LABELS.get
    grad_cols = {
        label("win_rate"):       ("RdYlGn", 0.35, 0.65),
        label("sharpe"):         ("RdYlGn", 0.0,  2.0),
        label("cagr"):           ("RdYlGn", -0.05, 0.30),
        label("max_drawdown"):   ("RdYlGn", -0.80, 0.0),
        label("profit_factor"):  ("RdYlGn", 0.8,  2.5),
    }
    for col, (cmap, vmin, vmax) in grad_cols.items():
        if col in s.data.columns:
            s = s.background_gradient(subset=[col], cmap=cmap, vmin=vmin, vmax=vmax)
    return s


def kpi_card(label: str, value: str, delta: str = "", delta_good: bool | None = None) -> str:
    delta_color = ""
    if delta_good is True:
        delta_color = "color:#2ca02c"
    elif delta_good is False:
        delta_color = "color:#d62728"
    delta_html = f'<div class="kpi-delta" style="{delta_color}">{delta}</div>' if delta else ""
    return (f'<div class="kpi-card">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'{delta_html}</div>')


def summary_box(text: str):
    st.markdown(f'<div class="summary-box">{text}</div>', unsafe_allow_html=True)


def indicator_category(ind: str) -> str:
    return INDICATOR_INFO.get(ind, {}).get("category", "Other")


# --------------------------------------------------------------------------- #
# Page header
# --------------------------------------------------------------------------- #
st.markdown("## 📈 15-Indicator Combination Backtester")
st.caption(
    "10,902 combinations (1–4 indicators from a pool of 23) × AND + MAJORITY logic × "
    "Nifty master universe — 17 years of daily data, 2009–2026."
)

# Metric definitions expander (always visible at top)
with st.expander("📖 Metric Definitions — click to understand every number"):
    st.markdown("""
| Metric | What it means | How it's computed |
|--------|--------------|-------------------|
| **Day Win%** | Hit rate on active trading days — % of days *in-market* where the strategy earned a positive daily P&L. Not the same as "% of completed trades that were profitable." | `mean(daily_return > 0)` on days where position ≠ 0 |
| **Total Return** | Total compounded return over the full backtest period (17 yrs). | `equity_final − 1`, where equity = cumulative product of `(1 + daily_return)` |
| **CAGR** | Compound Annual Growth Rate — the geometric-mean annual return. | `(1 + Total Return)^(1 / years) − 1` where years = trading days / 252 |
| **Sharpe Ratio** | Return per unit of risk, annualised. Computed over the **full** daily return series including flat (out-of-market) days. No risk-free rate subtracted. | `(mean_daily_return / std_daily_return) × √252` |
| **Profit Factor** | Gross profits ÷ Gross losses on active days. PF > 1 means the strategy makes more than it loses. "—" can mean *no losing days* (PF = ∞) or insufficient data. | `sum(positive_returns) / abs(sum(negative_returns))` |
| **Max Drawdown** | Worst peak-to-trough decline in the equity curve. More negative = bigger loss from a peak. | `min((equity − running_max) / running_max)` |
| **Trades** | Number of position entries or reversals (long→short counts as 1 trade). | Count of days where position changes to a non-zero value |
| **Avg Hold** | Average number of trading days the strategy stays in one position. | Days in market ÷ number of trades |
| **In Mkt%** | Fraction of trading days where the strategy holds a position (long or short). Strategies with low In Mkt% generate fewer but potentially higher-conviction signals. | Days with non-zero position ÷ total valid trading days |

**Important caveats:** All results assume zero transaction costs and frictionless short-selling.
In reality, commissions, impact costs, and shorting constraints would reduce performance.
Backtests are in-sample — actual future performance will differ.
""")

# --------------------------------------------------------------------------- #
# Guard: parquet must exist
# --------------------------------------------------------------------------- #

if not Path(RESULTS_PATH).exists():
    st.error(
        f"Results file not found at `{RESULTS_PATH}`.\n\n"
        "Run the backtest engine first:\n\n"
        "```bash\npython engine.py\n```\n\n"
        "For a 5-ticker smoke-test: `python engine.py --quick 5`"
    )
    st.stop()

results_path_str = str(RESULTS_PATH)
name_map    = load_universe_names(str(UNIVERSE_CSV)) if Path(UNIVERSE_CSV).exists() else {}
all_tickers = [t for t in load_scopes(results_path_str) if t not in ("OVERALL", "BUY_HOLD")]

# Shared buy-and-hold benchmark row
_overall_all = load_overall(results_path_str)
_bh_row      = _overall_all[_overall_all["combo"] == "BUY_HOLD"]
BH: dict | None = _bh_row.iloc[0].to_dict() if not _bh_row.empty else None

tab_global, tab_stock, tab_explorer, tab_guide = st.tabs(
    ["🌐 Global Overview", "🏢 Stock Analysis", "🧩 Strategy Explorer", "📚 Indicator Guide"]
)

# =========================================================================== #
# Tab 1 — Global Overview
# =========================================================================== #
with tab_global:

    # --- Benchmark banner ---
    if BH:
        bh_cagr = BH.get("cagr",  float("nan"))
        bh_sh   = BH.get("sharpe", float("nan"))
        bh_dd   = BH.get("max_drawdown", float("nan"))
        bh_tr   = BH.get("total_return", float("nan"))
        st.markdown(
            f"**📊 Buy-and-Hold Benchmark** (equal-weight Nifty universe, 2009–2026)  |  "
            f"CAGR **{fmt_pct(bh_cagr)}** &nbsp;·&nbsp; "
            f"Sharpe **{fmt_f(bh_sh)}** &nbsp;·&nbsp; "
            f"Total Return **{fmt_pct(bh_tr)}** &nbsp;·&nbsp; "
            f"Max DD **{fmt_pct(bh_dd)}**"
        )
        summary_box(
            "The Buy-and-Hold benchmark represents buying all stocks in the universe equally on day 1 "
            "and holding without trading. Every active strategy is compared against this passive baseline. "
            "Beating Buy-and-Hold on CAGR is necessary but not sufficient — a strategy must also have an "
            "acceptable Sharpe Ratio and manageable Drawdown to be worth implementing in practice."
        )
        st.divider()

    st.markdown("#### Top performing combinations — aggregated across the entire market")

    # --- Filters ---
    c1, c2, c3, c4 = st.columns([2, 1.5, 2, 2])
    with c1:
        metric_label = st.selectbox("Rank by", list(METRIC_OPTIONS.keys()), key="g_metric")
    with c2:
        logic_filter = st.selectbox("Logic", ["Both"] + list(LOGICS), key="g_logic")
    with c3:
        n_ind_filter = st.multiselect("# Indicators", [1, 2, 3, 4], default=[1, 2, 3, 4], key="g_n")
    with c4:
        min_trades_g = st.slider(
            "Min trades", 0, 200, 30, 10, key="g_mt",
            help="Remove combos with fewer than this many trades. Below 30 trades the statistics are unreliable (not enough samples for the Central Limit Theorem to apply)."
        )

    metric_col = METRIC_OPTIONS[metric_label]
    overall    = _overall_all[_overall_all["combo"] != "BUY_HOLD"].copy()

    if logic_filter != "Both":
        overall = overall[overall["logic"] == logic_filter]
    overall = overall[overall["n_indicators"].isin(n_ind_filter)]
    overall = overall[overall["num_trades"] >= min_trades_g]
    overall = overall.dropna(subset=[metric_col])

    top10 = overall.sort_values(metric_col, ascending=False).head(10)

    if top10.empty:
        st.warning("No combinations match the current filters. Try reducing Min Trades or broadening the Logic/Indicators filter.")
    else:
        summary_box(
            f"These are the top-10 indicator combinations ranked by <strong>{metric_label}</strong>, "
            f"each tested as an equal-weight portfolio across all stocks in the universe over 17 years "
            f"(2009–2026). The ranking is based on the <em>aggregate</em> portfolio performance — individual "
            f"stock results vary (see Stock Analysis and Strategy Explorer). "
            f"Green = strong on that metric · Red = weak. Max Drawdown: less negative = greener."
        )

        disp = top10[DISPLAY_COLS].set_index("combo")
        st.dataframe(style_table(disp), width="stretch", height=380)

        # Bar chart
        fig = px.bar(
            top10, x="combo", y=metric_col, color="logic",
            text=top10[metric_col].apply(
                lambda v: f"{v:.1%}" if metric_col in PCT_COLS else f"{v:.2f}"
            ),
            title=f"Top 10 combinations by {metric_label}",
            labels={metric_col: metric_label, "combo": ""},
            color_discrete_map={"AND": "#1f77b4", "MAJORITY": "#ff7f0e"},
        )
        if BH and metric_col in BH and pd.notna(BH[metric_col]):
            bh_val = BH[metric_col]
            label_str = f"{bh_val:.1%}" if metric_col in PCT_COLS else f"{bh_val:.2f}"
            fig.add_hline(
                y=bh_val, line_dash="dash", line_color="red",
                annotation_text=f"Buy & Hold ({label_str})",
                annotation_position="bottom right",
            )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           margin=dict(l=0, r=0, t=40, b=0), xaxis_tickangle=-30, font_size=12)
        st.plotly_chart(fig, width="stretch")
        summary_box(
            "The red dashed line marks the Buy-and-Hold benchmark. Bars above the line mean "
            "the combination outperformed passive investing on this metric. Blue bars use AND logic "
            "(all indicators must unanimously agree) — these generate fewer but higher-conviction "
            "signals. Orange bars use MAJORITY logic (>50% of indicators agree) — more signals, "
            "slightly lower conviction. Note: zero transaction costs are assumed."
        )

        # Indicator frequency chart
        st.markdown("##### Which indicators appear most often in the top 50 combos?")
        top50 = overall.sort_values(metric_col, ascending=False).head(50)
        counter = Counter()
        for cs in top50["combo"]:
            for ind in cs.split("+"):
                counter[ind] += 1
        if counter:
            freq_df = pd.DataFrame(counter.most_common(), columns=["Indicator", "Count"])
            freq_df["Category"] = freq_df["Indicator"].map(indicator_category)
            fig2 = px.bar(
                freq_df, x="Count", y="Indicator", orientation="h",
                color="Category", color_discrete_map=CATEGORY_COLORS,
                title="Indicator frequency in top-50 combinations",
                labels={"Count": "Appearances in top-50"},
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=40, b=0),
                yaxis={"categoryorder": "total ascending"},
                font_size=12, legend_title_text="Category",
            )
            st.plotly_chart(fig2, width="stretch")
            summary_box(
                "An indicator appearing frequently across many different top-50 combinations means "
                "it reliably contributes to performance in multiple contexts — it is not just one "
                "lucky grouping. For example, if SMA200 appears in 30 of the top-50 combos, it means "
                "whether combined with MACD, RSI, SuperTrend, or any other indicator, SMA200 consistently "
                "improves the result. This is a strong signal of robustness, not data-mining."
            )


# =========================================================================== #
# Tab 2 — Stock Analysis
# =========================================================================== #
with tab_stock:
    st.markdown("#### Best-performing combinations for a single stock")
    summary_box(
        "Select any stock from the universe to see which indicator combinations worked best "
        "on that specific ticker over the full backtest period. Individual stock results can "
        "differ significantly from the universe-wide average — a combination that ranks #1 overall "
        "may not even be in the top-50 for a specific stock, and vice versa."
    )

    label_to_ticker = {ticker_label(t, name_map): t for t in all_tickers}
    chosen_label  = st.selectbox("Select a stock", sorted(label_to_ticker.keys()), key="s_stock")
    chosen_ticker = label_to_ticker[chosen_label]

    c1, c2, c3 = st.columns(3)
    with c1:
        s_metric_label = st.selectbox("Rank by", list(METRIC_OPTIONS.keys()), key="s_metric")
    with c2:
        s_logic = st.selectbox("Logic", ["Both"] + list(LOGICS), key="s_logic")
    with c3:
        min_trades_s = st.slider(
            "Min trades", 0, 100, 10, 5, key="s_mt",
            help="With per-stock data spanning 17 years, 10+ trades is a reasonable minimum for individual stock analysis."
        )

    s_metric_col = METRIC_OPTIONS[s_metric_label]
    stock_df     = load_stock(results_path_str, chosen_ticker)

    bh_stock = stock_df[stock_df["combo"] == "BUY_HOLD"]
    stock_df = stock_df[stock_df["combo"] != "BUY_HOLD"].copy()

    if s_logic != "Both":
        stock_df = stock_df[stock_df["logic"] == s_logic]
    stock_df = stock_df[stock_df["num_trades"] >= min_trades_s]
    stock_df = stock_df.dropna(subset=[s_metric_col])

    if not bh_stock.empty:
        bh_s = bh_stock.iloc[0]
        st.info(
            f"**Buy & Hold — {chosen_label.split(' — ')[0]}:** "
            f"CAGR {fmt_pct(bh_s.get('cagr'))}  ·  "
            f"Sharpe {fmt_f(bh_s.get('sharpe'))}  ·  "
            f"Total Return {fmt_pct(bh_s.get('total_return'))}  ·  "
            f"Max DD {fmt_pct(bh_s.get('max_drawdown'))}  ·  "
            f"Win Rate {fmt_pct(bh_s.get('win_rate'))}"
        )
        summary_box(
            "The Buy-and-Hold numbers above are the passive baseline for this specific stock — "
            "what you'd have earned by simply buying and holding it with no indicator signals. "
            "Every active combination below is compared against this. A strategy must beat these "
            "numbers AND do so with lower risk (better Sharpe, smaller drawdown) to be worth trading."
        )

    if stock_df.empty:
        st.warning("No combinations match the current filters for this stock.")
    else:
        top5 = stock_df.sort_values(s_metric_col, ascending=False).head(5)
        st.markdown(f"**Top 5 combinations for {chosen_label}**")
        st.dataframe(style_table(top5[DISPLAY_COLS].set_index("combo")),
                     width="stretch", height=220)
        summary_box(
            f"These 5 combinations had the highest <strong>{s_metric_label}</strong> when applied to "
            f"{chosen_label.split(' — ')[0]} over the full backtest. "
            f"Check the Trades column — combinations with very few trades may have lucky statistics "
            f"(high CAGR from 1–2 big moves) rather than a genuine systematic edge. "
            f"Look for consistent metrics across Win%, Sharpe, and Profit Factor, not just one standout number."
        )

        fig3 = px.bar(
            top5, x="combo", y=[s_metric_col], barmode="group",
            title=f"Top 5 by {s_metric_label} — {chosen_label.split(' — ')[0]}",
            labels={"value": s_metric_label, "combo": ""},
        )
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=0, r=0, t=40, b=0), xaxis_tickangle=-20)
        st.plotly_chart(fig3, width="stretch")
        summary_box(
            f"Visual ranking of the top-5 combinations by {s_metric_label}. "
            f"Use this as a quick comparison — the combination with the tallest bar "
            f"dominates on this metric. Switch the 'Rank by' dropdown to verify that "
            f"the top combination also performs well on other metrics (CAGR, Sharpe, Win Rate), "
            f"not just the one you ranked by."
        )

        fig4 = px.scatter(
            stock_df, x="win_rate", y="cagr",
            color="n_indicators", symbol="logic",
            hover_data=["combo", "num_trades", "sharpe", "max_drawdown"],
            title=f"All combinations — {chosen_label.split(' — ')[0]}  (Win Rate vs CAGR)",
            labels={"win_rate": "Day Win Rate", "cagr": "CAGR", "n_indicators": "# Indicators"},
            color_continuous_scale="Viridis",
        )
        if not bh_stock.empty and pd.notna(bh_s.get("cagr")):
            fig4.add_hline(
                y=bh_s["cagr"], line_dash="dash", line_color="red",
                annotation_text="Buy & Hold CAGR",
                annotation_position="bottom right",
            )
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig4, width="stretch")
        summary_box(
            "Each dot is one combination applied to this stock. The <strong>ideal quadrant is top-right</strong> "
            "(high win rate AND high CAGR). Dots above the red dashed line beat the Buy-and-Hold CAGR for "
            "this stock. Dots clustered near the origin = mediocre combinations. "
            "Hover over any dot to see the full combination name, number of trades, Sharpe, and drawdown. "
            "A high CAGR with very few trades (check hover) may not be statistically meaningful."
        )


# =========================================================================== #
# Tab 3 — Strategy Explorer
# =========================================================================== #
with tab_explorer:
    st.markdown("#### Build a custom combination and understand exactly what it does")
    summary_box(
        "Select 1 to 4 indicators and a combination logic to see: (1) a plain-English explanation "
        "of what each indicator measures and how it generates its signal, (2) the aggregate portfolio "
        "performance across the entire Nifty universe, and (3) a per-stock breakdown so you can see "
        "whether the edge is broad (works on most stocks) or concentrated (a few lucky names)."
    )

    col_sel, col_logic = st.columns([3, 1])
    with col_sel:
        selected = st.multiselect(
            "Select 1 to 4 indicators", INDICATOR_NAMES, max_selections=4, key="e_ind",
            help="Order: MA → Trend → Momentum → Volatility → Volume → Cloud",
        )
    with col_logic:
        e_logic = st.radio(
            "Signal logic", list(LOGICS), horizontal=False, key="e_logic",
            help="AND = all selected indicators must agree on the same direction simultaneously.\n"
                 "MAJORITY = more than half of indicators must agree.",
        )

    min_trades_e = st.slider(
        "Min trades (per-stock table)", 0, 100, 5, 5, key="e_mt",
        help="Hides per-stock rows with fewer than this many trades. Fewer trades = less reliable statistics.",
    )

    if not selected:
        st.info("Pick 1–4 indicators above to see how this combination performed.")
        st.stop()

    # --- Indicator description cards ---
    st.markdown("##### What each selected indicator measures")
    summary_box(
        "Before looking at the numbers, understand <em>why</em> this combination might work. "
        "Each indicator below converts daily price/volume data into a +1 (bullish) or −1 (bearish) "
        "signal. The combination fires a trade only when indicators agree (based on the selected logic)."
    )
    for ind in selected:
        info = INDICATOR_INFO.get(ind, {})
        cat  = info.get("category", "Other")
        col  = CATEGORY_COLORS.get(cat, "#888")
        st.markdown(f"""
<div class="ind-card" style="border-left-color:{col}; background:#fafaf8">
  <span style="background:{col}20; color:{col}; padding:2px 8px; border-radius:4px;
        font-size:11px; font-weight:600">{cat}</span>
  <strong style="font-size:15px; margin-left:8px">{ind}</strong>
  <span style="color:#666; font-size:13px"> — {info.get('full_name','')}</span>
  <p style="color:#555; font-size:13px; margin:6px 0 0 0; line-height:1.55">{info.get('desc','')}</p>
  <div class="signal-box">
    🎯 <b>Signal rule:</b> {info.get('signal','')}<br>
    ⚙️ <b>Parameters:</b> {info.get('params','')}
  </div>
</div>
""", unsafe_allow_html=True)

    # Logic explanation
    if e_logic == "AND":
        logic_explain = (
            f"<strong>AND logic</strong> — a trade is entered only when ALL {len(selected)} selected "
            f"indicator(s) simultaneously point in the same direction. "
            f"This produces fewer but higher-conviction signals."
        )
    else:
        thresh = len(selected) // 2 + 1
        logic_explain = (
            f"<strong>MAJORITY logic</strong> — a trade is entered when at least "
            f"{thresh} of {len(selected)} indicator(s) agree on the same direction. "
            f"This produces more signals, each with slightly lower conviction."
        )
    st.markdown(
        f'<div class="summary-box">{logic_explain} '
        f"Position is set the day <em>after</em> the signal fires (next-day execution — "
        f"no look-ahead bias). Long (+1) when indicators are bullish; Short (−1) when bearish; "
        f"Flat (0) when no consensus.</div>",
        unsafe_allow_html=True,
    )

    # --- Load data ---
    combo_str  = "+".join(sorted(selected, key=INDICATOR_NAMES.index))
    combo_rows = load_combo(results_path_str, combo_str, e_logic)

    if combo_rows.empty:
        st.warning(
            f"No results found for `{combo_str}` / {e_logic}. "
            "Has the full backtest been run? Try `python engine.py`."
        )
        st.stop()

    # --- Overall KPI cards ---
    overall_row = combo_rows[combo_rows["scope"] == "OVERALL"]
    if not overall_row.empty:
        r = overall_row.iloc[0]
        st.markdown(f"**Overall market performance — `{combo_str}` ({e_logic})**")
        summary_box(
            "The KPI cards below show how this combination performed as an equal-weight portfolio: "
            "the strategy runs on every stock in the universe independently, and the daily P&L is "
            "averaged across all stocks. This smooths out individual stock variance. "
            "The 'vs B&H' delta shows how much better (green +) or worse (red −) this strategy "
            "performed compared to just buying and holding the same stocks."
        )

        bh_cagr_ref = BH.get("cagr") if BH else None
        bh_sh_ref   = BH.get("sharpe") if BH else None
        bh_dd_ref   = BH.get("max_drawdown") if BH else None

        def _delta(val, bh_val, is_drawdown=False):
            if val is None or bh_val is None or pd.isna(val) or pd.isna(bh_val):
                return "", None
            diff = val - bh_val
            good = diff > 0 if not is_drawdown else diff > 0
            sign = "+" if diff > 0 else ""
            return f"vs B&H {sign}{diff:.1%}", good

        d_cagr,  g_cagr  = _delta(r.get("cagr"),         bh_cagr_ref)
        d_sh,    g_sh    = _delta(r.get("sharpe"),        bh_sh_ref)
        d_dd,    g_dd    = _delta(r.get("max_drawdown"),   bh_dd_ref, is_drawdown=True)

        cards = [
            kpi_card("Day Win Rate",  fmt_pct(r.get("win_rate")),
                     delta=f"Active days: {fmt_pct(r.get('pct_in_market'))}"),
            kpi_card("CAGR",          fmt_pct(r.get("cagr")),      delta=d_cagr, delta_good=g_cagr),
            kpi_card("Sharpe Ratio",  fmt_f(r.get("sharpe")),      delta=d_sh,   delta_good=g_sh),
            kpi_card("Max Drawdown",  fmt_pct(r.get("max_drawdown")), delta=d_dd, delta_good=g_dd),
            kpi_card("Trades",        fmt_f(r.get("num_trades"), 0),
                     delta=f"Avg hold {fmt_f(r.get('avg_hold_days'), 1)} days"),
        ]
        st.markdown(
            '<div class="metric-row">' + "".join(cards) + "</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Sharpe is computed on the full daily return series, including days when the strategy is flat "
            "(zero return). This is the correct full-period Sharpe — strategies that trade rarely will "
            "show lower Sharpe than always-in-market strategies even if their active-day returns are excellent."
        )

    # --- Per-stock breakdown ---
    per_stock = combo_rows[combo_rows["scope"] != "OVERALL"].copy()
    per_stock = per_stock[per_stock["num_trades"] >= min_trades_e]
    per_stock["label"] = per_stock["scope"].apply(lambda t: ticker_label(t, name_map))

    e_metric_label = st.selectbox("Rank stocks by", list(METRIC_OPTIONS.keys()), key="e_metric")
    e_metric_col   = METRIC_OPTIONS[e_metric_label]
    per_stock      = per_stock.dropna(subset=[e_metric_col]).sort_values(e_metric_col, ascending=False)

    st.markdown(f"**Per-stock results — `{combo_str}` ({e_logic})**")
    summary_box(
        f"The same combination applied to each of the {len(per_stock)} individual stocks. "
        f"<strong>If most stocks show consistent positive results</strong> (e.g., CAGR > Buy-and-Hold, "
        f"Sharpe > 0.5), the combination has a genuine market-wide edge — it works broadly. "
        f"<strong>If only 5–10 stocks drive the top results</strong>, the combination may be "
        f"overfit to those specific names. Scroll the table to see the full distribution."
    )
    display_ps = per_stock[["label", "win_rate", "total_return", "cagr",
                             "sharpe", "profit_factor", "max_drawdown",
                             "num_trades", "avg_hold_days", "pct_in_market"]].set_index("label")
    st.dataframe(style_table(display_ps), width="stretch", height=380)

    # Histogram
    fig5 = px.histogram(
        per_stock, x=e_metric_col, nbins=30, color_discrete_sequence=["#1f77b4"],
        title=f"Distribution of {e_metric_label} across stocks — `{combo_str}` ({e_logic})",
        labels={e_metric_col: e_metric_label},
    )
    if BH and e_metric_col in BH and pd.notna(BH[e_metric_col]):
        fig5.add_vline(
            x=BH[e_metric_col], line_dash="dash", line_color="red",
            annotation_text="Buy & Hold",
            annotation_position="top right",
        )
    fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig5, width="stretch")
    summary_box(
        f"This histogram shows how {e_metric_label} is distributed across all ~{len(per_stock)} stocks. "
        f"A distribution skewed <strong>to the right</strong> of the red Buy-and-Hold line means the "
        f"combination generates alpha on the majority of individual stocks — the strongest evidence of a "
        f"robust, non-data-mined edge. A bell curve <strong>centred on the Buy-and-Hold line</strong> "
        f"means the strategy breaks even on average. A distribution to the <strong>left</strong> means "
        f"the strategy underperforms passive investing on most stocks."
    )


# =========================================================================== #
# Tab 4 — Indicator Guide
# =========================================================================== #
with tab_guide:
    st.markdown("#### What every indicator measures and how its signal is generated")
    summary_box(
        "All 23 indicators are reduced to a single +1 (bullish) / −1 (bearish) / 0 (neutral) signal "
        "each trading day. Trend and MA indicators are <em>always directional</em> — they produce +1 or −1 "
        "every day. Oscillators (RSI, STOCH, CCI) use the momentum interpretation: above their midpoint = "
        "bullish, below = bearish. Volatility indicators (BBANDS, ATR) and Ichimoku can produce 0 (neutral) "
        "when no clear signal is present. Select a category below to filter the guide."
    )

    cat_filter = st.radio(
        "Filter by category", ["All"] + list(INDICATOR_CATEGORIES.keys()),
        horizontal=True, key="guide_cat",
    )

    cats_to_show = list(INDICATOR_CATEGORIES.keys()) if cat_filter == "All" else [cat_filter]

    for cat in cats_to_show:
        indicators = INDICATOR_CATEGORIES[cat]
        cat_color  = CATEGORY_COLORS[cat]
        st.markdown(
            f'<h5 style="color:{cat_color}; margin-top:20px; margin-bottom:4px">'
            f'● {cat} Indicators</h5>',
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        for i, ind in enumerate(indicators):
            info = INDICATOR_INFO.get(ind, {})
            with cols[i % 2]:
                st.markdown(f"""
<div style="border:1px solid {cat_color}40; border-left:4px solid {cat_color};
     border-radius:8px; padding:14px 16px; margin-bottom:12px; background:#fafaf8">
  <div style="display:flex; align-items:baseline; gap:8px; margin-bottom:4px">
    <span style="font-size:17px; font-weight:700">{ind}</span>
    <span style="font-size:12px; color:{cat_color}; font-weight:600">{cat}</span>
  </div>
  <div style="font-size:13px; font-weight:600; color:#333; margin-bottom:6px">
    {info.get('full_name','')}
  </div>
  <div style="font-size:13px; color:#555; margin-bottom:8px; line-height:1.55">
    {info.get('desc','')}
  </div>
  <div style="background:#f0f0ee; border-radius:4px; padding:8px 10px; font-size:12px;
       font-family:monospace; line-height:1.7">
    🎯 {info.get('signal','')}<br>
    ⚙️ {info.get('params','')}
  </div>
</div>
""", unsafe_allow_html=True)
