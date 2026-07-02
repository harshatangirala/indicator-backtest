"""
app.py — Streamlit dashboard for the 15-indicator combination backtest.

Reads ONLY from data/backtest_results.parquet (produced by `python engine.py`).
Every query uses parquet predicate pushdown so only a small slice of the
1.29 M+ row table is materialised — keeps the dashboard inside Streamlit Cloud /
Hugging Face Spaces free-tier memory limits.

Tabs
----
🌐 Global Overview   — top combos across the entire universe, vs buy-and-hold
🏢 Stock Analysis    — best combos for a single stock
🧩 Strategy Explorer — build a custom 1–4 indicator combo, see what it means & how it performed
📚 Indicator Guide   — what every indicator measures and how its signal is generated
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
# Styling
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
.metric-row { display:flex; gap:16px; margin-bottom:8px; }
.kpi-card {
    flex:1; background:#fff; border:1px solid #e5e4df; border-radius:8px;
    padding:14px 18px; text-align:center;
}
.kpi-value { font-size:22px; font-weight:700; color:#1a1a18; margin:4px 0; }
.kpi-label { font-size:12px; color:#888780; text-transform:uppercase; letter-spacing:.04em; }
.kpi-delta { font-size:12px; margin-top:4px; }
.ind-card {
    border-radius:8px; padding:14px 16px; margin-bottom:10px;
    border-left:4px solid;
}
.badge {
    display:inline-block; padding:2px 8px; border-radius:4px;
    font-size:11px; font-weight:600; margin-bottom:8px;
}
.signal-box {
    background:#f5f5f5; border-radius:4px; padding:8px 10px;
    font-family:monospace; font-size:12px; margin-top:8px;
}
hr.thin { border:0; border-top:1px solid #e5e4df; margin:12px 0; }
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
    "combo": "Combination", "n_indicators": "#", "logic": "Logic",
    "win_rate": "Day Win%", "total_return": "Total Ret", "cagr": "CAGR",
    "sharpe": "Sharpe", "profit_factor": "Profit F.",
    "max_drawdown": "Max DD", "num_trades": "Trades",
    "avg_hold_days": "Avg Hold", "pct_in_market": "In Mkt%",
}

PCT_COLS  = ["win_rate", "total_return", "cagr", "max_drawdown", "pct_in_market"]
FLOAT2_COLS = ["sharpe", "profit_factor", "avg_hold_days"]

# --------------------------------------------------------------------------- #
# Data loading — all use predicate pushdown; never load the full table
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
# Shared helpers
# --------------------------------------------------------------------------- #

def fmt_pct(v) -> str:
    return f"{v:.1%}" if pd.notna(v) else "—"

def fmt_f(v, d=2) -> str:
    return f"{v:.{d}f}" if pd.notna(v) else "—"


def style_table(df: pd.DataFrame) -> pd.DataFrame.style:
    """Apply consistent formatting + colour gradients to a results dataframe."""
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

    # Green = good, red = bad for key metrics
    label = COL_LABELS.get
    grad_cols = {
        label("win_rate"):      ("RdYlGn", 0.35, 0.65),
        label("sharpe"):        ("RdYlGn", 0.0,  2.0),
        label("cagr"):          ("RdYlGn", -0.05, 0.30),
        label("max_drawdown"):  ("RdYlGn", -0.80, 0.0),   # less negative = greener
        label("profit_factor"): ("RdYlGn", 0.8,  2.5),
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


def indicator_category(ind: str) -> str:
    return INDICATOR_INFO.get(ind, {}).get("category", "Other")


# --------------------------------------------------------------------------- #
# Guard: parquet must exist
# --------------------------------------------------------------------------- #

st.markdown("## 📈 15-Indicator Combination Backtester")
st.caption(
    "10,902 indicator combinations (1–4 from a pool of 23) backtested under "
    "strict-AND and majority-vote logic across the Nifty master universe."
)

if not Path(RESULTS_PATH).exists():
    st.error(
        f"Results file not found at `{RESULTS_PATH}`.\n\n"
        "Run the backtest engine first:\n\n"
        "```bash\npython engine.py\n```\n\n"
        "For a quick smoke-test (5 tickers): `python engine.py --quick 5`"
    )
    st.stop()

results_path_str = str(RESULTS_PATH)
name_map    = load_universe_names(str(UNIVERSE_CSV)) if Path(UNIVERSE_CSV).exists() else {}
all_tickers = [t for t in load_scopes(results_path_str) if t not in ("OVERALL", "BUY_HOLD")]

# --------------------------------------------------------------------------- #
# Shared buy-and-hold benchmark (OVERALL row)
# --------------------------------------------------------------------------- #
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
    # Buy-and-hold benchmark banner
    if BH:
        bh_cagr = BH.get("cagr", float("nan"))
        bh_sh   = BH.get("sharpe", float("nan"))
        bh_dd   = BH.get("max_drawdown", float("nan"))
        bh_tr   = BH.get("total_return", float("nan"))
        st.markdown(
            f"**📊 Buy-and-Hold Benchmark** (equal-weight Nifty universe, "
            f"2009–2026): &nbsp;"
            f"CAGR **{fmt_pct(bh_cagr)}** &nbsp;·&nbsp; "
            f"Sharpe **{fmt_f(bh_sh)}** &nbsp;·&nbsp; "
            f"Total Return **{fmt_pct(bh_tr)}** &nbsp;·&nbsp; "
            f"Max DD **{fmt_pct(bh_dd)}**"
        )
        st.divider()

    st.markdown("#### Top performing combinations — aggregated across the entire market")

    c1, c2, c3, c4 = st.columns([2, 1.5, 2, 2])
    with c1:
        metric_label = st.selectbox("Rank by", list(METRIC_OPTIONS.keys()), key="g_metric")
    with c2:
        logic_filter = st.selectbox("Logic", ["Both"] + list(LOGICS), key="g_logic")
    with c3:
        n_ind_filter = st.multiselect("# Indicators", [1, 2, 3, 4], default=[1, 2, 3, 4], key="g_n")
    with c4:
        min_trades_g = st.slider("Min trades", 0, 200, 30, 10, key="g_mt",
                                  help="Exclude combos with fewer trades (statistical noise filter)")

    metric_col = METRIC_OPTIONS[metric_label]
    overall    = _overall_all[_overall_all["combo"] != "BUY_HOLD"].copy()

    if logic_filter != "Both":
        overall = overall[overall["logic"] == logic_filter]
    overall = overall[overall["n_indicators"].isin(n_ind_filter)]
    overall = overall[overall["num_trades"] >= min_trades_g]
    overall = overall.dropna(subset=[metric_col])

    top10 = overall.sort_values(metric_col, ascending=False).head(10)

    if top10.empty:
        st.warning("No combinations match the current filters.")
    else:
        # Add vs-benchmark delta column
        if BH and pd.notna(BH.get("cagr")):
            top10 = top10.copy()
            top10["vs_bh_cagr"] = top10["cagr"] - BH["cagr"]

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
            fig.add_hline(
                y=BH[metric_col], line_dash="dash", line_color="red",
                annotation_text=f"Buy & Hold ({BH[metric_col]:.2%})"
                if metric_col in PCT_COLS else f"Buy & Hold ({BH[metric_col]:.2f})",
                annotation_position="bottom right",
            )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           margin=dict(l=0, r=0, t=40, b=0), xaxis_tickangle=-30,
                           font_size=12)
        st.plotly_chart(fig, width="stretch")

        st.markdown("##### Which indicators appear most often in the top 50 combos?")
        top50 = overall.sort_values(metric_col, ascending=False).head(50)
        counter = Counter()
        for cs in top50["combo"]:
            for ind in cs.split("+"):
                counter[ind] += 1
        if counter:
            freq_df = pd.DataFrame(counter.most_common(), columns=["Indicator", "Count"])
            freq_df["Category"] = freq_df["Indicator"].map(indicator_category)
            freq_df["Color"]    = freq_df["Category"].map(CATEGORY_COLORS)
            fig2 = px.bar(
                freq_df, x="Count", y="Indicator", orientation="h",
                color="Category",
                color_discrete_map=CATEGORY_COLORS,
                title="Indicator frequency in top 50 combos",
                labels={"Count": "Appearances in top 50"},
            )
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                margin=dict(l=0, r=0, t=40, b=0), yaxis={"categoryorder": "total ascending"},
                                font_size=12, legend_title_text="Category")
            st.plotly_chart(fig2, width="stretch")


# =========================================================================== #
# Tab 2 — Stock Analysis
# =========================================================================== #
with tab_stock:
    st.markdown("#### Best-performing combinations for a single stock")

    label_to_ticker = {ticker_label(t, name_map): t for t in all_tickers}
    chosen_label  = st.selectbox("Select a stock", sorted(label_to_ticker.keys()), key="s_stock")
    chosen_ticker = label_to_ticker[chosen_label]

    c1, c2, c3 = st.columns(3)
    with c1:
        s_metric_label = st.selectbox("Rank by", list(METRIC_OPTIONS.keys()), key="s_metric")
    with c2:
        s_logic = st.selectbox("Logic", ["Both"] + list(LOGICS), key="s_logic")
    with c3:
        min_trades_s = st.slider("Min trades", 0, 100, 10, 5, key="s_mt")

    s_metric_col = METRIC_OPTIONS[s_metric_label]
    stock_df     = load_stock(results_path_str, chosen_ticker)

    # Buy-and-hold for this stock
    bh_stock = stock_df[stock_df["combo"] == "BUY_HOLD"]
    stock_df = stock_df[stock_df["combo"] != "BUY_HOLD"].copy()

    if s_logic != "Both":
        stock_df = stock_df[stock_df["logic"] == s_logic]
    stock_df = stock_df[stock_df["num_trades"] >= min_trades_s]
    stock_df = stock_df.dropna(subset=[s_metric_col])

    if not bh_stock.empty:
        bh_s = bh_stock.iloc[0]
        st.info(
            f"**Buy & Hold** for {chosen_label.split(' — ')[0]}: "
            f"CAGR {fmt_pct(bh_s.get('cagr'))} · "
            f"Sharpe {fmt_f(bh_s.get('sharpe'))} · "
            f"Total Return {fmt_pct(bh_s.get('total_return'))} · "
            f"Max DD {fmt_pct(bh_s.get('max_drawdown'))}"
        )

    if stock_df.empty:
        st.warning("No combinations match filters for this stock.")
    else:
        top5 = stock_df.sort_values(s_metric_col, ascending=False).head(5)
        st.markdown(f"**Top 5 combinations for {chosen_label}**")
        st.dataframe(style_table(top5[DISPLAY_COLS].set_index("combo")),
                     width="stretch", height=220)

        fig3 = px.bar(
            top5, x="combo", y=[s_metric_col], barmode="group",
            title=f"Top 5 by {s_metric_label}",
            labels={"value": s_metric_label, "combo": ""},
        )
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=0, r=0, t=40, b=0), xaxis_tickangle=-20)
        st.plotly_chart(fig3, width="stretch")

        # Scatter: all combos, win rate vs CAGR
        fig4 = px.scatter(
            stock_df, x="win_rate", y="cagr", color="n_indicators",
            symbol="logic", hover_data=["combo", "num_trades", "sharpe"],
            title=f"All combos for {chosen_label.split(' — ')[0]} — win rate vs CAGR",
            labels={"win_rate": "Day Win Rate", "cagr": "CAGR", "n_indicators": "# Indicators"},
            color_continuous_scale="Viridis",
        )
        if not bh_stock.empty:
            fig4.add_hline(y=bh_s.get("cagr", 0), line_dash="dash", line_color="red",
                           annotation_text="Buy & Hold CAGR", annotation_position="bottom right")
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig4, width="stretch")


# =========================================================================== #
# Tab 3 — Strategy Explorer
# =========================================================================== #
with tab_explorer:
    st.markdown("#### Build a custom 1–4 indicator combination and understand it")

    col_sel, col_logic = st.columns([3, 1])
    with col_sel:
        selected = st.multiselect(
            "Select 1 to 4 indicators", INDICATOR_NAMES, max_selections=4, key="e_ind",
            help="Indicators are listed in order: MA → Trend → Momentum → Volatility → Volume → Cloud",
        )
    with col_logic:
        e_logic = st.radio("Signal logic", list(LOGICS), horizontal=False, key="e_logic",
                            help="AND = all selected indicators must agree. MAJORITY = more than half must agree.")

    min_trades_e = st.slider("Min trades (for per-stock table)", 0, 100, 5, 5, key="e_mt")

    if not selected:
        st.info("Pick 1–4 indicators above to see how this combination performs.")
        st.stop()

    # ---- Indicator description cards ---- #
    st.markdown("##### What each selected indicator measures")
    for ind in selected:
        info = INDICATOR_INFO.get(ind, {})
        cat  = info.get("category", "Other")
        col  = CATEGORY_COLORS.get(cat, "#888")
        st.markdown(f"""
<div class="ind-card" style="border-left-color:{col}; background:#fafaf8">
  <span class="badge" style="background:{col}20; color:{col}">{cat}</span>
  <strong style="font-size:15px"> {ind}</strong> &nbsp;
  <span style="color:#666; font-size:13px">{info.get('full_name','')}</span>
  <p style="color:#555; font-size:13px; margin:6px 0 0 0">{info.get('desc','')}</p>
  <div class="signal-box">
    🎯 <b>Signal:</b> {info.get('signal','')}<br>
    ⚙️ <b>Params:</b> {info.get('params','')}
  </div>
</div>
""", unsafe_allow_html=True)

    # Logic explanation
    if e_logic == "AND":
        agree_text = "ALL selected indicators must simultaneously point the same direction."
    else:
        agree_text = "More than half of the selected indicators must point the same direction."
    st.markdown(
        f"**Combined rule ({e_logic}):** {agree_text} "
        f"Position is entered the day after the signal fires; "
        f"long (+1) or short (−1), flat (0) when no consensus."
    )

    # ---- Load data ---- #
    combo_str  = "+".join(sorted(selected, key=INDICATOR_NAMES.index))
    combo_rows = load_combo(results_path_str, combo_str, e_logic)

    if combo_rows.empty:
        st.warning(
            f"No results for `{combo_str}` ({e_logic}). "
            "Has the full backtest been run? Try `python engine.py`."
        )
        st.stop()

    # ---- Overall KPIs ---- #
    overall_row = combo_rows[combo_rows["scope"] == "OVERALL"]
    if not overall_row.empty:
        r = overall_row.iloc[0]
        st.markdown(f"**Overall market performance — `{combo_str}` ({e_logic})**")

        bh_cagr_ref = BH.get("cagr") if BH else None
        bh_sh_ref   = BH.get("sharpe") if BH else None

        cards = [
            kpi_card("Day Win Rate",   fmt_pct(r.get("win_rate")),
                     delta="vs B&H " + fmt_pct(r.get("win_rate", 0) - (BH.get("win_rate", 0) if BH else 0))
                     if BH else ""),
            kpi_card("CAGR",           fmt_pct(r.get("cagr")),
                     delta="vs B&H " + fmt_pct(r.get("cagr", 0) - (bh_cagr_ref or 0)) if BH else "",
                     delta_good=(r.get("cagr", 0) > (bh_cagr_ref or 0)) if BH else None),
            kpi_card("Sharpe Ratio",   fmt_f(r.get("sharpe")),
                     delta="vs B&H " + fmt_f(r.get("sharpe", 0) - (bh_sh_ref or 0)) if BH else "",
                     delta_good=(r.get("sharpe", 0) > (bh_sh_ref or 0)) if BH else None),
            kpi_card("Max Drawdown",   fmt_pct(r.get("max_drawdown")),
                     delta="vs B&H " + fmt_pct(r.get("max_drawdown", 0) - (BH.get("max_drawdown", 0) if BH else 0))
                     if BH else "",
                     delta_good=(r.get("max_drawdown", 0) > (BH.get("max_drawdown", 0) if BH else -1)
                                 if BH else None)),
            kpi_card("# Trades",       fmt_f(r.get("num_trades"), 0),
                     delta=f"Avg hold {fmt_f(r.get('avg_hold_days'), 1)} days"),
        ]
        st.markdown(
            '<div class="metric-row">' + "".join(cards) + "</div>",
            unsafe_allow_html=True,
        )

    # ---- Per-stock table ---- #
    per_stock = combo_rows[combo_rows["scope"] != "OVERALL"].copy()
    per_stock = per_stock[per_stock["num_trades"] >= min_trades_e]
    per_stock["label"] = per_stock["scope"].apply(lambda t: ticker_label(t, name_map))

    e_metric_label = st.selectbox("Rank stocks by", list(METRIC_OPTIONS.keys()), key="e_metric")
    e_metric_col   = METRIC_OPTIONS[e_metric_label]
    per_stock      = per_stock.dropna(subset=[e_metric_col]).sort_values(e_metric_col, ascending=False)

    st.markdown(f"**Per-stock results — `{combo_str}` ({e_logic})**")
    display_ps = per_stock[["label", "win_rate", "total_return", "cagr",
                             "sharpe", "profit_factor", "max_drawdown",
                             "num_trades", "avg_hold_days", "pct_in_market"]].set_index("label")
    st.dataframe(style_table(display_ps), width="stretch", height=360)

    # Distribution histogram
    fig5 = px.histogram(
        per_stock, x=e_metric_col, nbins=30, color_discrete_sequence=["#1f77b4"],
        title=f"Distribution of {e_metric_label} across stocks — `{combo_str}` ({e_logic})",
        labels={e_metric_col: e_metric_label},
    )
    if BH and e_metric_col in BH and pd.notna(BH[e_metric_col]):
        fig5.add_vline(x=BH[e_metric_col], line_dash="dash", line_color="red",
                       annotation_text="Buy & Hold", annotation_position="top right")
    fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig5, width="stretch")


# =========================================================================== #
# Tab 4 — Indicator Guide
# =========================================================================== #
with tab_guide:
    st.markdown("#### What every indicator measures and how its signal is generated")
    st.caption(
        "All 23 indicators reduced to a +1 (bullish) / −1 (bearish) / 0 (neutral) signal "
        "each day. Combinations of 1–4 indicators are aggregated under AND or MAJORITY logic."
    )

    cat_filter = st.radio(
        "Filter by category", ["All"] + list(INDICATOR_CATEGORIES.keys()),
        horizontal=True, key="guide_cat",
    )

    cats_to_show = (list(INDICATOR_CATEGORIES.keys()) if cat_filter == "All"
                    else [cat_filter])

    for cat in cats_to_show:
        indicators = INDICATOR_CATEGORIES[cat]
        cat_color  = CATEGORY_COLORS[cat]
        st.markdown(
            f'<h5 style="color:{cat_color}; margin-top:20px">'
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
  <div style="font-size:13px; color:#555; margin-bottom:8px; line-height:1.5">
    {info.get('desc','')}
  </div>
  <div style="background:#f0f0ee; border-radius:4px; padding:8px 10px; font-size:12px; font-family:monospace; line-height:1.7">
    🎯 {info.get('signal','')}<br>
    ⚙️ {info.get('params','')}
  </div>
</div>
""", unsafe_allow_html=True)
