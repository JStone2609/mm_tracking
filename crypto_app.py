# crypto_app.py — MM Crypto vs Competitors — ROI

from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

MM_PATH = Path("mm_top_crypto.csv")           # cols: Ticker, Date
MAP_PATH = Path("crypto_map.csv")             # cols: User Symbol, CoinGecko ID, Currency
PARQUET_PATH = Path("crypto_prices_cache.parquet")
COMPETITORS = [
    ("bitcoin", "BTC"),
    ("ethereum", "ETH"),
    ("ripple", "XRP"),
]

st.set_page_config(page_title="MM Top Crypto vs Competitors — ROI", layout="wide")
st.title("MM Top Crypto vs Competitors — Live Tracking")

st.caption(
    """
- **What we do:** The **MM Crypto** selects a set of crypto assets each month and dollar-cost averages **1 unit** into each on its specified buy date.
- **Benchmarks:** **BTC, ETH, and XRP** also invest **1 unit** on those same buy dates (first valid day with a price), holding thereafter.
- **ROI metric:** (Portfolio value − cost) ÷ cost.
- **Breakdown:** Hover a point to see ROI, cumulative profit, total value, and active buy count.
"""
)

def load_buys(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"Ticker":"user_symbol","Date":"buy_date"})
    df["user_symbol"] = df["user_symbol"].astype(str).str.strip().str.upper()
    df["buy_date"] = pd.to_datetime(df["buy_date"], errors="coerce")
    return df.dropna(subset=["user_symbol","buy_date"]).reset_index(drop=True)

def load_map(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"User Symbol":"user_symbol","CoinGecko ID":"cg_id","Currency":"currency"})
    df["user_symbol"] = df["user_symbol"].astype(str).str.strip().str.upper()
    df["cg_id"] = df["cg_id"].astype(str).str.strip().str.lower()
    return df[["user_symbol","cg_id"]]

@st.cache_data(show_spinner=False)
def load_prices_parquet(path: Path, version: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # ✅ Make index tz-naive to avoid invalid comparison
    idx = pd.to_datetime(df.index, errors="coerce")
    try:
        # If the index is tz-aware, drop the tz; if not, this raises and we fall back
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
    for name, df in benchmarks_df.groupby("series"):
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
    buys = load_buys(MM_PATH)
    tmap = load_map(MAP_PATH)
except Exception as e:
    st.error(f"Failed to read CSVs: {e}")
    st.stop()

if not PARQUET_PATH.exists():
    st.error("Missing crypto_prices_cache.parquet. The GitHub Action must write it first.")
    st.stop()

try:
    version = PARQUET_PATH.stat().st_mtime_ns
    prices = load_prices_parquet(PARQUET_PATH, version)
except Exception as e:
    st.error(f"Failed to load crypto_prices_cache.parquet: {e}")
    st.stop()

mapped = buys.merge(tmap, on="user_symbol", how="left")
mapped["cg_id"] = mapped["cg_id"].fillna("").astype(str)
available = set(prices.columns.astype(str))
mapped = mapped[mapped["cg_id"].isin(available)].reset_index(drop=True)

if mapped.empty:
    st.error("No portfolio CoinGecko IDs are present in the price cache.")
    st.stop()

first_buy_date = pd.to_datetime(buys["buy_date"].min()).normalize()

date_index = prices.index[prices.index >= first_buy_date]
prices = prices.loc[date_index]

# Per-purchase values
per_purchase_values, row_keys = [], []
for _, row in mapped.iterrows():
    cid, bdt = row["cg_id"], row["buy_date"]
    s = prices[cid]
    ent = first_valid_on_or_after(s, bdt)
    if ent is None:
        continue
    p0 = s.at[ent]
    if pd.isna(p0) or p0 == 0:
        continue
    rel = (s / p0).where(date_index >= ent, 0.0)
    per_purchase_values.append(rel)
    row_keys.append((row["user_symbol"], ent.date().isoformat(), cid))

if not per_purchase_values:
    st.error("No valid portfolio entries after symbol/date alignment.")
    st.stop()

permat = pd.DataFrame(per_purchase_values)
permat.columns = permat.columns.strftime("%Y-%m-%d")
permat.insert(0, "Buy Date", [k[1] for k in row_keys])
permat.insert(0, "Ticker",   [k[0] for k in row_keys])
permat.insert(2, "CG ID",    [k[2] for k in row_keys])

portfolio_df = aggregate_matrix(permat)
portfolio_df["series"] = "MM Crypto"

# Entry dates (for competitor)
value_cols = permat.columns[3:]
value_dt_index = pd.to_datetime(value_cols)
entry_dates = []
for _, r in permat.iterrows():
    vals = r[value_cols].astype(float).to_numpy()
    nz = np.flatnonzero(vals > 0)
    if nz.size:
        entry_dates.append(value_dt_index[nz[0]])

def competitor_series(cg_id: str, label: str) -> pd.DataFrame:
    if cg_id not in prices.columns:
        return pd.DataFrame()
    s = prices[cg_id]
    per_list = []
    for ent in entry_dates:
        ent2 = first_valid_on_or_after(s, ent)
        if ent2 is None:
            continue
        p0 = s.at[ent2]
        if pd.isna(p0) or p0 == 0:
            continue
        rel = (s / p0).where(date_index >= ent2, 0.0)
        per_list.append(rel)
    if not per_list:
        return pd.DataFrame()
    mat = pd.DataFrame(per_list)
    mat.columns = mat.columns.strftime("%Y-%m-%d")
    mat.insert(0, "Buy Date", [""] * len(mat))
    mat.insert(0, "Ticker", [""] * len(mat))
    mat.insert(2, "CG ID", [cg_id] * len(mat))
    ts = aggregate_matrix(mat)
    ts["series"] = label
    return ts

bench_long = [portfolio_df]
for cg_id, label in COMPETITORS:
    competitor_df = competitor_series(cg_id, label)
    if not competitor_df.empty:
        bench_long.append(competitor_df)

benchmarks_df = pd.concat(bench_long, ignore_index=True)
benchmarks_df = benchmarks_df[benchmarks_df["date"] >= first_buy_date].reset_index(drop=True)

fig = build_chart(benchmarks_df, start_date=first_buy_date)
st.plotly_chart(fig, use_container_width=True)

last_date = pd.to_datetime(prices.index.max()).date()
st.caption(f"Last price date in crypto cache: **{last_date.isoformat()}**.")

html_bytes = fig.to_html(full_html=True, include_plotlyjs="inline").encode("utf-8")
st.download_button(
    label="Download chart as HTML",
    file_name=f"mm_crypto_vs_competitors_{datetime.utcnow().date().isoformat()}.html",
    data=html_bytes,
    mime="text/html",
)
