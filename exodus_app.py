# exodus_app.py — MM Exodus Algo vs Competitors — Live Tracking

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------- Files ----------
EXODUS_PATH = Path("mm_exodus_algo.csv")              # cols: Ticker, Amount, Date, Action
MAP_PATH = Path("exodus_ticker_map.csv")              # cols: User Ticker, Resolved Ticker [, Currency]
PARQUET_PATH = Path("exodus_prices_cache.parquet")    # written daily by GitHub Actions
COMPETITORS = ["SPY", "QQQ"]                          # fixed competitors

EXPECTED_UPDATE_UTC = "22:20 UTC"  # matches the GitHub Action cron

# ---------- Page ----------
st.set_page_config(page_title="MM Exodus Algo vs Competitors — Live Tracking", layout="wide")
st.title("MM Exodus Algo vs Competitors — Live Tracking")

st.caption(
    """
- **What we do:** The **MM Exodus Algo** tracks trades and compares their performance to selected competitors.
- **Fair benchmark:** For each Exodus trade, the same **amount** is applied to **SPY** and **QQQ** on the same dates, rolling forward to the next valid price date if the market was closed.
- **Action handling:** **BUY** is treated as long exposure; **SELL** is treated as short exposure.
- **ROI metric:** **(Portfolio value − invested amount) ÷ invested amount**.
- **Breakdown:** Hover over any point on the chart to see ROI, cumulative profit, total value, invested amount, and active trade count for that date.
"""
)

# ---------- Helpers ----------
def exchsym_to_yahoo(resolved: str) -> str | None:
    if not isinstance(resolved, str):
        return None
    resolved = resolved.strip()
    if not resolved or resolved.upper() in {"N/A", "NA", "NONE", "NULL"}:
        return None
    if ":" not in resolved:
        return resolved.upper()

    exch, sym = resolved.split(":", 1)
    exch = exch.strip().upper()
    sym = sym.strip().upper()
    suffix = {
        "NASDAQ": "",
        "NYSE": "",
        "AMEX": "",
        "NYSEARCA": "",
        "LON": ".L",
        "LSE": ".L",
        "AMS": ".AS",
    }.get(exch, "")
    return sym + suffix


def load_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "Ticker": "user_ticker",
            "Amount": "amount",
            "Date": "trade_date",
            "Action": "action",
        }
    )
    df["user_ticker"] = df["user_ticker"].astype(str).str.strip().str.upper()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["action"] = df["action"].astype(str).str.strip().str.upper()

    df = df.dropna(subset=["user_ticker", "trade_date", "amount"]).reset_index(drop=True)
    df = df[df["amount"] > 0].copy()
    df["action"] = df["action"].where(df["action"].isin(["BUY", "SELL"]), "BUY")
    return df


def load_map(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"User Ticker": "user_ticker", "Resolved Ticker": "resolved"})
    df["user_ticker"] = df["user_ticker"].astype(str).str.strip().str.upper()
    df["resolved"] = df["resolved"].astype(str).str.strip()
    df["yf_ticker"] = df["resolved"].apply(exchsym_to_yahoo)
    return df[["user_ticker", "yf_ticker"]]


@st.cache_data(show_spinner=False)
def load_prices_parquet(path: Path, version: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    idx = pd.to_datetime(df.index, errors="coerce")
    try:
        idx = idx.tz_localize(None)
    except (TypeError, AttributeError, ValueError):
        pass
    df.index = idx
    df = df[~df.index.isna()].sort_index()
    df = df.loc[:, df.notna().any(axis=0)]
    return df


def first_valid_on_or_after(s: pd.Series, when: pd.Timestamp) -> pd.Timestamp | None:
    sub = s.loc[s.index >= when]
    sub = sub[sub.notna()]
    if sub.empty:
        return None
    return sub.index[0]


def build_position_series(
    price_series: pd.Series,
    date_index: pd.DatetimeIndex,
    entry_date: pd.Timestamp,
    amount: float,
    action: str,
) -> tuple[pd.Series | None, pd.Series | None, pd.Timestamp | None]:
    ent = first_valid_on_or_after(price_series, entry_date)
    if ent is None:
        return None, None, None

    p0 = price_series.at[ent]
    if pd.isna(p0) or p0 == 0:
        return None, None, None

    ratio = price_series / p0

    # Long BUY:  current value = amount * (price / entry_price)
    # Short SELL: current value = amount * (2 - price / entry_price)
    if action == "SELL":
        value = (2.0 - ratio) * amount
    else:
        value = ratio * amount

    value = value.where(date_index >= ent, 0.0)
    invested = pd.Series(np.where(date_index >= ent, amount, 0.0), index=date_index, dtype=float)

    return value, invested, ent


def aggregate_positions(
    value_wide: pd.DataFrame,
    invested_wide: pd.DataFrame,
) -> pd.DataFrame:
    dates = pd.to_datetime(value_wide.columns)

    vals = value_wide.to_numpy(dtype=float)
    invs = invested_wide.to_numpy(dtype=float)

    total_value = pd.Series(vals.sum(axis=0), index=dates, name="total_value")
    invested_amount = pd.Series(invs.sum(axis=0), index=dates, name="invested_amount")
    active_trades = pd.Series((invs > 0).sum(axis=0), index=dates, name="active_trades")
    cumulative_profit = (total_value - invested_amount).rename("cumulative_profit")
    roi = (cumulative_profit / invested_amount.replace(0, np.nan)).rename("roi")

    out = pd.concat(
        [total_value, invested_amount, active_trades, cumulative_profit, roi],
        axis=1,
    ).reset_index(names="date")
    out["date"] = pd.to_datetime(out["date"])
    return out


def build_chart(benchmarks_df: pd.DataFrame, start_date: pd.Timestamp) -> go.Figure:
    fig = go.Figure()

    groups = list(benchmarks_df.groupby("series"))
    exodus_groups = [g for g in groups if g[0] == "MM Exodus"]
    other_groups = [g for g in groups if g[0] != "MM Exodus"]
    ordered_groups = exodus_groups + other_groups

    for name, df in ordered_groups:
        df = df.sort_values("date")
        custom = np.stack(
            [
                df["total_value"].to_numpy(),
                df["invested_amount"].to_numpy(),
                df["active_trades"].fillna(0).astype(int).to_numpy(),
                df["cumulative_profit"].to_numpy(),
                df["roi"].to_numpy(),
            ],
            axis=-1,
        )

        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["roi"],
                mode="lines",
                name=name,
                line=dict(width=3),
                customdata=custom,
                hovertemplate=(
                    "<b>%{x|%Y-%m-%d}</b><br>"
                    + name + ": ROI %{y:.2%}<br>"
                    "Cumulative Profit: %{customdata[3]:.4f}<br>"
                    "Total Value: %{customdata[0]:.4f}<br>"
                    "Invested Amount: %{customdata[1]:.4f}<br>"
                    "Active Trades: %{customdata[2]:d}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        template="plotly_white",
        xaxis=dict(
            title="Date",
            type="date",
            range=[pd.to_datetime(start_date), None],
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(title="ROI", rangemode="tozero", tickformat=".0%"),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0.5, xanchor="center"),
        margin=dict(l=60, r=60, t=60, b=80),
    )
    return fig


# ---------- Main ----------
try:
    trades = load_trades(EXODUS_PATH)
    tmap = load_map(MAP_PATH)
except Exception as e:
    st.error(f"Failed to read CSVs: {e}")
    st.stop()

if not PARQUET_PATH.exists():
    st.error("Missing exodus_prices_cache.parquet. The GitHub Action must write it first.")
    st.stop()

try:
    try:
        parquet_version = PARQUET_PATH.stat().st_mtime_ns
    except Exception:
        parquet_version = 0
    prices = load_prices_parquet(PARQUET_PATH, parquet_version)
except Exception as e:
    st.error(f"Failed to load exodus_prices_cache.parquet: {e}")
    st.stop()

# Map to Yahoo symbols and keep only those present
mapped = trades.merge(tmap, on="user_ticker", how="left")
mapped["yf_ticker"] = np.where(
    mapped["yf_ticker"].isna() | (mapped["yf_ticker"].astype(str).str.len() == 0),
    mapped["user_ticker"].astype(str).str.upper(),
    mapped["yf_ticker"],
)

available = set(prices.columns.astype(str))
mapped = mapped[mapped["yf_ticker"].isin(available)].reset_index(drop=True)

if mapped.empty:
    st.error("No portfolio symbols are present in the price cache.")
    st.stop()

first_trade_date = pd.to_datetime(trades["trade_date"].min()).normalize()

# Restrict working calendar to start at first trade date
date_index = prices.index[prices.index >= first_trade_date]
prices = prices.loc[date_index]

# ---------- Build MM Exodus portfolio ----------
portfolio_values = []
portfolio_invested = []
entry_records = []

for _, row in mapped.iterrows():
    sym = row["yf_ticker"]
    trade_date = row["trade_date"]
    amount = float(row["amount"])
    action = row["action"]

    s = prices[sym]
    value_s, invested_s, ent = build_position_series(
        price_series=s,
        date_index=date_index,
        entry_date=trade_date,
        amount=amount,
        action=action,
    )
    if value_s is None or invested_s is None or ent is None:
        continue

    portfolio_values.append(value_s)
    portfolio_invested.append(invested_s)
    entry_records.append(
        {
            "entry_date": ent,
            "amount": amount,
            "action": action,
            "user_ticker": row["user_ticker"],
            "yf_ticker": sym,
        }
    )

if not portfolio_values:
    st.error("No valid portfolio entries after symbol/date alignment.")
    st.stop()

portfolio_value_mat = pd.DataFrame(portfolio_values, index=range(len(portfolio_values)))
portfolio_invested_mat = pd.DataFrame(portfolio_invested, index=range(len(portfolio_invested)))

portfolio_value_mat.columns = portfolio_value_mat.columns.strftime("%Y-%m-%d")
portfolio_invested_mat.columns = portfolio_invested_mat.columns.strftime("%Y-%m-%d")

portfolio_df = aggregate_positions(portfolio_value_mat, portfolio_invested_mat)
portfolio_df["series"] = "MM Exodus"

# ---------- Build competitors using same dates, amounts, and actions ----------
def competitor_series(sym: str) -> pd.DataFrame:
    if sym not in prices.columns:
        return pd.DataFrame()

    s = prices[sym]
    value_list = []
    invested_list = []

    for rec in entry_records:
        value_s, invested_s, _ = build_position_series(
            price_series=s,
            date_index=date_index,
            entry_date=rec["entry_date"],
            amount=float(rec["amount"]),
            action=rec["action"],
        )
        if value_s is None or invested_s is None:
            continue
        value_list.append(value_s)
        invested_list.append(invested_s)

    if not value_list:
        return pd.DataFrame()

    value_mat = pd.DataFrame(value_list)
    invested_mat = pd.DataFrame(invested_list)

    value_mat.columns = value_mat.columns.strftime("%Y-%m-%d")
    invested_mat.columns = invested_mat.columns.strftime("%Y-%m-%d")

    ts = aggregate_positions(value_mat, invested_mat)
    ts["series"] = sym
    return ts


bench_long = [portfolio_df]
for comp in COMPETITORS:
    cdf = competitor_series(comp)
    if not cdf.empty:
        bench_long.append(cdf)

benchmarks_df = pd.concat(bench_long, ignore_index=True)
benchmarks_df = benchmarks_df[benchmarks_df["date"] >= first_trade_date].reset_index(drop=True)

# Filter out dates where no series has valid ROI data
valid_dates = benchmarks_df.groupby("date")["roi"].apply(lambda x: x.notna().any())
benchmarks_df = benchmarks_df[
    benchmarks_df["date"].isin(valid_dates[valid_dates].index)
].reset_index(drop=True)

# Plot
fig = build_chart(benchmarks_df, start_date=first_trade_date)
st.plotly_chart(fig, use_container_width=True)

# Footer
last_date = pd.to_datetime(prices.index.max()).date()
st.caption(
    f"Last price date in cache: **{last_date.isoformat()}**. "
    f"Updates happen once per trading day via GitHub Actions (≈{EXPECTED_UPDATE_UTC}, after U.S. market close)."
)

# Download
html_bytes = fig.to_html(full_html=True, include_plotlyjs="inline").encode("utf-8")
st.download_button(
    label="Download chart as HTML",
    file_name=f"mm_exodus_vs_competitors_{datetime.utcnow().date().isoformat()}.html",
    data=html_bytes,
    mime="text/html",
)
