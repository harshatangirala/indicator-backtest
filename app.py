"""
app.py — Streamlit dashboard for the 15-indicator combination backtest.

Reads ONLY from data/backtest_results.parquet (produced by `python engine.py`).
Run with:
    streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from engine import INDICATOR_NAMES, LOGICS, RESULTS_PATH, UNIVERSE_CSV

st.set_page_config(page_title="Indicator Combo Backtester", layout="wide", page_icon="📈")

METRIC_OPTIONS = {
    "Win Rate": "win_rate",
    "Total Return": "total_return",
    "Sharpe Ratio": "sharpe",
    "Profit Factor": "profit_factor",
    "CAGR": "cagr",
}


@st.cache_data
def load_results(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for col in ("combo", "logic", "scope"):
        df[col] = df[col].astype(str)
    return df


@st.cache_data
def load_universe_names(path: str) -> dict:
    uni = pd.read_csv(path)
    uni["yf_ticker"] = uni["Symbol"].str.strip().str.replace("&", "%26", regex=False) + ".NS"
    return dict(zip(uni["yf_ticker"], uni["Company Name"]))


def ticker_label(ticker: str, name_map: dict) -> str:
    name = name_map.get(ticker, "")
    base = ticker.replace(".NS", "")
    return f"{base} — {name}" if name else base


def style_pct_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame.style:
    fmt = {c: "{:.2%}" for c in cols}
    for c in ("sharpe", "profit_factor"):
        if c in df.columns:
            fmt[c] = "{:.2f}"
    return df.style.format(fmt)


st.markdown("## 📈 15-Indicator Combination Backtester")
st.caption(
    "Backtests every 1–4 indicator combination (1,940 total) from a 15-indicator pool, "
    "under strict-AND and majority-vote logic, across the Nifty master universe."
)

if not Path(RESULTS_PATH).exists():
    st.error(
        f"No results file found at `{RESULTS_PATH}`.\n\n"
        "Run the backtest engine first:\n\n"
        "```\npython engine.py\n```"
    )
    st.stop()

results = load_results(str(RESULTS_PATH))
name_map = load_universe_names(str(UNIVERSE_CSV)) if Path(UNIVERSE_CSV).exists() else {}
all_tickers = sorted(t for t in results["scope"].unique() if t != "OVERALL")

tab_global, tab_stock, tab_explorer = st.tabs(
    ["🌐 Global Overview", "🏢 Stock Analysis", "🧩 Strategy Explorer"]
)

# --------------------------------------------------------------------------- #
# Tab 1 — Global Overview
# --------------------------------------------------------------------------- #
with tab_global:
    st.markdown("#### Top performing combinations — aggregated across the entire market")

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_label = st.selectbox("Rank by", list(METRIC_OPTIONS.keys()), key="g_metric")
    with c2:
        logic_filter = st.selectbox("Logic", ["Both"] + list(LOGICS), key="g_logic")
    with c3:
        n_ind_filter = st.multiselect(
            "# Indicators", [1, 2, 3, 4], default=[1, 2, 3, 4], key="g_n_ind"
        )

    metric_col = METRIC_OPTIONS[metric_label]
    overall = results[results["scope"] == "OVERALL"].copy()
    if logic_filter != "Both":
        overall = overall[overall["logic"] == logic_filter]
    overall = overall[overall["n_indicators"].isin(n_ind_filter)]
    overall = overall.dropna(subset=[metric_col])

    top10 = overall.sort_values(metric_col, ascending=False).head(10)
    display_cols = ["combo", "n_indicators", "logic", "win_rate", "total_return",
                     "sharpe", "profit_factor", "cagr", "num_trades"]

    st.dataframe(
        style_pct_cols(top10[display_cols].set_index("combo"),
                       ["win_rate", "total_return", "cagr"]),
        use_container_width=True,
    )

    fig = px.bar(
        top10, x="combo", y=metric_col, color="logic",
        title=f"Top 10 combinations by {metric_label}",
        labels={metric_col: metric_label, "combo": "Indicator combination"},
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       margin=dict(l=0, r=0, t=40, b=0), xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# Tab 2 — Stock Analysis
# --------------------------------------------------------------------------- #
with tab_stock:
    st.markdown("#### Best-performing combinations for a single stock")

    label_to_ticker = {ticker_label(t, name_map): t for t in all_tickers}
    chosen_label = st.selectbox("Select a stock", sorted(label_to_ticker.keys()), key="s_stock")
    chosen_ticker = label_to_ticker[chosen_label]

    c1, c2 = st.columns(2)
    with c1:
        s_metric_label = st.selectbox("Rank by", list(METRIC_OPTIONS.keys()), key="s_metric")
    with c2:
        s_logic_filter = st.selectbox("Logic", ["Both"] + list(LOGICS), key="s_logic")

    s_metric_col = METRIC_OPTIONS[s_metric_label]
    stock_df = results[results["scope"] == chosen_ticker].copy()
    if s_logic_filter != "Both":
        stock_df = stock_df[stock_df["logic"] == s_logic_filter]
    stock_df = stock_df.dropna(subset=[s_metric_col])

    top5 = stock_df.sort_values(s_metric_col, ascending=False).head(5)
    st.dataframe(
        style_pct_cols(top5[display_cols].set_index("combo"),
                       ["win_rate", "total_return", "cagr"]),
        use_container_width=True,
    )

    fig2 = px.bar(
        top5, x="combo", y=["win_rate", "total_return", "sharpe"],
        barmode="group", title=f"Top 5 combinations for {chosen_label}",
        labels={"value": "Metric value", "combo": "Indicator combination"},
    )
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0, r=0, t=40, b=0), xaxis_tickangle=-35)
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(
        stock_df, x="win_rate", y="total_return", color="n_indicators", symbol="logic",
        hover_data=["combo"], title=f"All combinations for {chosen_label} — win rate vs. total return",
    )
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig3, use_container_width=True)

# --------------------------------------------------------------------------- #
# Tab 3 — Strategy Explorer
# --------------------------------------------------------------------------- #
with tab_explorer:
    st.markdown("#### Manually build a 1–4 indicator combination and see how it performed")

    selected = st.multiselect(
        "Select 1 to 4 indicators", INDICATOR_NAMES, max_selections=4, key="e_indicators"
    )
    e_logic = st.radio("Combination logic", list(LOGICS), horizontal=True, key="e_logic")

    if not selected:
        st.info("Pick at least one indicator above to see its backtest results.")
    else:
        combo_str = "+".join(sorted(selected, key=INDICATOR_NAMES.index))
        combo_rows = results[(results["combo"] == combo_str) & (results["logic"] == e_logic)]

        if combo_rows.empty:
            st.warning(
                f"No results found for `{combo_str}` ({e_logic}). "
                "Did the backtest engine finish running for the full universe?"
            )
        else:
            overall_row = combo_rows[combo_rows["scope"] == "OVERALL"]
            if not overall_row.empty:
                r = overall_row.iloc[0]
                st.markdown(f"**Overall market performance — `{combo_str}` ({e_logic})**")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Win Rate", f"{r['win_rate']:.1%}" if pd.notna(r["win_rate"]) else "n/a")
                m2.metric("Total Return", f"{r['total_return']:.1%}" if pd.notna(r["total_return"]) else "n/a")
                m3.metric("Sharpe", f"{r['sharpe']:.2f}" if pd.notna(r["sharpe"]) else "n/a")
                m4.metric("Profit Factor", f"{r['profit_factor']:.2f}" if pd.notna(r["profit_factor"]) else "n/a")
                m5.metric("Max Drawdown", f"{r['max_drawdown']:.1%}" if pd.notna(r["max_drawdown"]) else "n/a")

            per_stock = combo_rows[combo_rows["scope"] != "OVERALL"].copy()
            per_stock["label"] = per_stock["scope"].apply(lambda t: ticker_label(t, name_map))
            e_metric_label = st.selectbox("Rank stocks by", list(METRIC_OPTIONS.keys()), key="e_metric")
            e_metric_col = METRIC_OPTIONS[e_metric_label]
            per_stock = per_stock.dropna(subset=[e_metric_col]).sort_values(e_metric_col, ascending=False)

            st.markdown(f"**Per-stock performance — `{combo_str}` ({e_logic})**")
            st.dataframe(
                style_pct_cols(
                    per_stock[["label", "win_rate", "total_return", "sharpe",
                               "profit_factor", "cagr", "num_trades"]].set_index("label"),
                    ["win_rate", "total_return", "cagr"],
                ),
                use_container_width=True,
            )

            fig4 = px.histogram(
                per_stock, x=e_metric_col, nbins=30,
                title=f"Distribution of {e_metric_label} across stocks — `{combo_str}` ({e_logic})",
            )
            fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig4, use_container_width=True)
