# exodus_app.py — MM Exodus Algo Trades vs Competitors


from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------- Files ----------
EXODUS_PATH = Path("mm_exodus_algo.csv")           # cols: Ticker, Amount, Date, Action
MAP_PATH = Path("exodus_ticker_map.csv")           # cols: User Ticker, Resolved Ticker, Currency
PARQUET_PATH = Path("exodus_prices_cache.parquet")
COMPETITORS = ["SPY", "QQQ"]  # fixed competitors

EXPECTED_UPDATE_UTC = "22:20 UTC"  # matches the GitHub Action cron

# ---------- Page ----------
st.set_page_config(page_title="MM Exodus Algo vs Competitors", layout="wide")
st.title("MM Exodus Algo vs Competitors — Live Tracking")

st.caption(
    """
- **What we do:** The **MM Exodus Algo** tracks trades and compares their performance to selected competitors.
- **Benchmarks:** For each buy, the same amount is invested in each competitor on those same dates.
- **ROI metric:** (Portfolio value − cost) ÷ cost.
- **Breakdown:** Hover over any point on the chart to see ROI, cumulative profit, total value, and active buy count for any specific date.
"""
)

# ---------- Helpers ----------
def exchsym_to_yahoo(resolved: str) -> str | None:
    if not isinstance(resolved, str) or ":" not in resolved:
        return None
    exch, sym = resolved.split(":", 1)
    exch = exch.strip().upper()
    sym = sym.strip().upper()
    suffix = {
        "NASDAQ": "", "NYSE": "", "AMEX": "", "NYSEARCA": "",
        "LON": ".L", "LSE": ".L", "AMS": ".AS",
    }.get(exch, "")
    return sym + suffix

def load_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"Ticker": "user_ticker", "Amount": "amount", "Date": "buy_date", "Action": "action"})
    df["user_ticker"] = df["user_ticker"].astype(str).str.strip()
    df["buy_date"] = pd.to_datetime(df["buy_date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["action"] = df["action"].astype(str).str.strip().str.upper()
    return df.dropna(subset=["user_ticker", "buy_date", "amount"]).reset_index(drop=True)

def load_map(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"User Ticker": "user_ticker", "Resolved Ticker": "resolved"})
    df["user_ticker"] = df["user_ticker"].astype(str).str.strip()
    df["resolved"] = df["resolved"].astype(str).str.strip()
    df["yf_ticker"] = df["resolved"].apply(exchsym_to_yahoo)
    return df[["user_ticker", "yf_ticker"]]

@st.cache_data(show_spinner=False)
def load_prices_parquet(path: Path, version: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    df = df.loc[:, df.notna().any(axis=0)]
    return df

def first_valid_on_or_after(s: pd.Series, when: pd.Timestamp) -> pd.Timestamp | None:
    sub = s.loc[s.index >= when]
    sub = sub[sub.notna()]
    if sub.empty:
        return None
    return sub.index[0]

def aggregate_matrix(values_wide: pd.DataFrame, date_col_start_idx: int = 3) -> pd.DataFrame:
    dates = pd.to_datetime(values_wide.columns[date_col_start_idx:])
    vals = values_wide.iloc[:, date_col_start_idx:].to_numpy(dtype=float)
    total_value = pd.Series(vals.sum(axis=0), index=dates, name="total_value")
    active_buys = pd.Series((vals > 0).sum(axis=0), index=dates, name="active_buys")
    cumulative_profit = (total_value - active_buys).rename("cumulative_profit")
    roi = (cumulative_profit / active_buys.replace(0, np.nan)).rename("roi")
    out = pd.concat([total_value, active_buys, cumulative_profit, roi], axis=1).reset_index(names="date")
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
            [df["total_value"].to_numpy(),
             df["active_buys"].fillna(0).astype(int).to_numpy(),
             df["cumulative_profit"].to_numpy(),
             df["roi"].to_numpy()],
            axis=-1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["date"], y=df["roi"], mode="lines", name=name,
                line=dict(width=3),
                customdata=custom,
                hovertemplate=(
                    "<b>%{x|%Y-%m-%d}</b><br>"
                    + name + ": ROI %{y:.2%}<br>"
                    "Cumulative Profit: %{customdata[2]:.4f}<br>"
                    "Total Value: %{customdata[0]:.4f}<br>"
                    "Active Buys: %{customdata[1]:d}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(title="Date", type="date", range=[start_date, None], rangeslider=dict(visible=False)),
        yaxis=dict(title="ROI", rangemode="tozero", tickformat=".0%"),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0.5, xanchor="center"),
        margin=dict(l=60, r=60, t=60, b=80),
    )
    return fig

# ---------- Main ----------
# Load inputs & prices
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
        parquet_version = PARQUET_PATH.stat().st_mtime_ns  # cache-buster
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

first_buy_date = pd.to_datetime(trades["buy_date"].min()).normalize()

# Portfolio per-purchase values (per-symbol roll-forward)
date_index = prices.index[prices.index >= first_buy_date]
prices = prices.loc[date_index]
per_purchase_values, row_keys = [], []
for _, row in mapped.iterrows():
    sym, bdt, amt = row["yf_ticker"], row["buy_date"], row["amount"]
    s = prices[sym]
    ent = first_valid_on_or_after(s, bdt)
    if ent is None:
        continue
    p0 = s.at[ent]
    if pd.isna(p0) or p0 == 0:
        continue
    rel = (s / p0 * amt).where(date_index >= ent, 0.0)
    per_purchase_values.append(rel)
    row_keys.append((row["user_ticker"], ent.date().isoformat(), sym))

if not per_purchase_values:
    st.error("No valid portfolio entries after symbol/date alignment.")
    st.stop()

permat = pd.DataFrame(per_purchase_values)
permat.columns = permat.columns.strftime("%Y-%m-%d")
permat.insert(0, "Buy Date", [k[1] for k in row_keys])
permat.insert(0, "Ticker",   [k[0] for k in row_keys])
permat.insert(2, "Yahoo Symbol", [k[2] for k in row_keys])

portfolio_df = aggregate_matrix(permat)
portfolio_df["series"] = "MM Exodus"

# Portfolio-derived buy dates (after roll-forward)
value_cols = permat.columns[3:]
value_dt_index = pd.to_datetime(value_cols)
entry_dates = []
for _, r in permat.iterrows():
    vals = r[value_cols].astype(float).to_numpy()
    nz = np.flatnonzero(vals > 0)
    if nz.size:
        entry_dates.append(value_dt_index[nz[0]])

def competitor_series(sym: str) -> pd.DataFrame:
    if sym not in prices.columns:
        return pd.DataFrame()
    s = prices[sym]
    per_list = []
    for ent, amt in zip(entry_dates, mapped["amount"]):
        ent2 = first_valid_on_or_after(s, ent)
        if ent2 is None:
            continue
        p0 = s.at[ent2]
        if pd.isna(p0) or p0 == 0:
            continue
        rel = (s / p0 * amt).where(date_index >= ent2, 0.0)
        per_list.append(rel)
    if not per_list:
        return pd.DataFrame()
    mat = pd.DataFrame(per_list)
    mat.columns = mat.columns.strftime("%Y-%m-%d")
    mat.insert(0, "Buy Date", [""] * len(mat))
    mat.insert(0, "Ticker", [""] * len(mat))
    mat.insert(2, "Yahoo Symbol", [sym] * len(mat))
    ts = aggregate_matrix(mat)
    ts["series"] = sym
    return ts

bench_long = [portfolio_df]
for comp in COMPETITORS:
    cdf = competitor_series(comp)
    if not cdf.empty:
        bench_long.append(cdf)

benchmarks_df = pd.concat(bench_long, ignore_index=True)
benchmarks_df = benchmarks_df[benchmarks_df["date"] >= first_buy_date].reset_index(drop=True)

# Filter out dates where no series has valid ROI data (e.g., market holidays)
valid_dates = benchmarks_df.groupby("date")["roi"].apply(lambda x: x.notna().any())
benchmarks_df = benchmarks_df[benchmarks_df["date"].isin(valid_dates[valid_dates].index)].reset_index(drop=True)

# Plot
fig = build_chart(benchmarks_df, start_date=first_buy_date)
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

def load_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"Ticker":"user_ticker","Amount":"amount","Date":"date","Action":"action"})
    df["user_ticker"] = df["user_ticker"].astype(str).str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["action"] = df["action"].astype(str).str.strip().str.upper()
    return df.dropna(subset=["user_ticker","date","amount"]).reset_index(drop=True)

def load_map(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"User Ticker":"user_ticker","Resolved Ticker":"resolved_ticker","Currency":"currency"})
    df["user_ticker"] = df["user_ticker"].astype(str).str.strip().str.upper()
    df["resolved_ticker"] = df["resolved_ticker"].astype(str).str.strip().str.upper()
    return df[["user_ticker","resolved_ticker"]]

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

def aggregate_matrix(values_wide: pd.DataFrame, date_col_start_idx: int = 3) -> pd.DataFrame:
    dates = pd.to_datetime(values_wide.columns[date_col_start_idx:])
    vals = values_wide.iloc[:, date_col_start_idx:].to_numpy(dtype=float)
    total_value = pd.Series(vals.sum(axis=0), index=dates, name="total_value")
    active_buys = pd.Series((vals > 0).sum(axis=0), index=dates, name="active_buys")
    cumulative_profit = (total_value - active_buys).rename("cumulative_profit")
    roi = (cumulative_profit / active_buys.replace(0, np.nan)).rename("roi")
    out = pd.concat([total_value, active_buys, cumulative_profit, roi], axis=1).reset_index(names="date")
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
            [df["total_value"].to_numpy(),
             df["active_buys"].fillna(0).astype(int).to_numpy(),
             df["cumulative_profit"].to_numpy(),
             df["roi"].to_numpy()],
            axis=-1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["date"], y=df["roi"], mode="lines", name=name,
                line=dict(width=3),
                customdata=custom,
                hovertemplate=(
                    "<b>%{x|%Y-%m-%d}</b><br>"
                    + name + ": ROI %{y:.2%}<br>"
                    "Cumulative Profit: %{customdata[2]:.4f}<br>"
                    "Total Value: %{customdata[0]:.4f}<br>"
                    "Active Buys: %{customdata[1]:d}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(title="Date", type="date", range=[start_date, None], rangeslider=dict(visible=False)),
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
    version = PARQUET_PATH.stat().st_mtime_ns
    prices = load_prices_parquet(PARQUET_PATH, version)
except Exception as e:
    st.error(f"Failed to load exodus_prices_cache.parquet: {e}")
    st.stop()

mapped = trades.merge(tmap, on="user_ticker", how="left")
mapped["resolved_ticker"] = mapped["resolved_ticker"].fillna("").astype(str)
available = set(prices.columns.astype(str))
mapped = mapped[mapped["resolved_ticker"].isin(available)].reset_index(drop=True)

if mapped.empty:
    st.error("No portfolio tickers are present in the price cache.")
    st.stop()

first_buy_date = pd.to_datetime(trades["date"].min()).normalize()

date_index = prices.index[prices.index >= first_buy_date]
prices = prices.loc[date_index]

# Per-purchase values
per_purchase_values, row_keys = [], []
for _, row in mapped.iterrows():
    tid, bdt, amt = row["resolved_ticker"], row["date"], row["amount"]
    s = prices[tid]
    ent = first_valid_on_or_after(s, bdt)
    if ent is None:
        continue
    p0 = s.at[ent]
    if pd.isna(p0) or p0 == 0:
        continue
    rel = (s / p0 * amt).where(date_index >= ent, 0.0)
    per_purchase_values.append(rel)
    row_keys.append((row["user_ticker"], ent.date().isoformat(), tid))

if not per_purchase_values:
    st.error("No valid portfolio entries after symbol/date alignment.")
    st.stop()

permat = pd.DataFrame(per_purchase_values)
permat.columns = permat.columns.strftime("%Y-%m-%d")
permat.insert(0, "Buy Date", [k[1] for k in row_keys])
permat.insert(0, "Ticker",   [k[0] for k in row_keys])
permat.insert(2, "Resolved Ticker",    [k[2] for k in row_keys])

portfolio_df = aggregate_matrix(permat)
portfolio_df["series"] = "MM Exodus"

# Entry dates (for competitor)
value_cols = permat.columns[3:]
value_dt_index = pd.to_datetime(value_cols)
entry_dates = []
for _, r in permat.iterrows():
    vals = r[value_cols].astype(float).to_numpy()
    nz = np.flatnonzero(vals > 0)
    if nz.size:
        entry_dates.append(value_dt_index[nz[0]])

def competitor_series(tid: str, label: str) -> pd.DataFrame:
    if tid not in prices.columns:
        return pd.DataFrame()
    s = prices[tid]
    per_list = []
    for ent, amt in zip(entry_dates, mapped["amount"]):
        ent2 = first_valid_on_or_after(s, ent)
        if ent2 is None:
            continue
        p0 = s.at[ent2]
        if pd.isna(p0) or p0 == 0:
            continue
        rel = (s / p0 * amt).where(date_index >= ent2, 0.0)
        per_list.append(rel)
    if not per_list:
        return pd.DataFrame()
    mat = pd.DataFrame(per_list)
    mat.columns = mat.columns.strftime("%Y-%m-%d")
    mat.insert(0, "Buy Date", [""] * len(mat))
    mat.insert(0, "Ticker", [""] * len(mat))
    mat.insert(2, "Resolved Ticker", [tid] * len(mat))
    ts = aggregate_matrix(mat)
    ts["series"] = label
    return ts

bench_long = [portfolio_df]
for tid, label in COMPETITORS:
    competitor_df = competitor_series(tid, label)
    if not competitor_df.empty:
        bench_long.append(competitor_df)

benchmarks_df = pd.concat(bench_long, ignore_index=True)
benchmarks_df = benchmarks_df[benchmarks_df["date"] >= first_buy_date].reset_index(drop=True)

fig = build_chart(benchmarks_df, start_date=first_buy_date)
st.plotly_chart(fig, use_container_width=True)

last_date = pd.to_datetime(prices.index.max()).date()
st.caption(f"Last price date in exodus cache: **{last_date.isoformat()}**.")

html_bytes = fig.to_html(full_html=True, include_plotlyjs="inline").encode("utf-8")
st.download_button(
    label="Download chart as HTML",
    file_name=f"mm_exodus_vs_competitors_{datetime.utcnow().date().isoformat()}.html",
    data=html_bytes,
    mime="text/html",
)
