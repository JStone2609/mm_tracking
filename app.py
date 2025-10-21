import io
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ---------- Page setup ----------
st.set_page_config(page_title="MM Top 20 vs Competitors — ROI", layout="wide")
st.title("MM Top 20 vs Competitors — ROI")
st.caption("Level stakes; competitors buy on the same entry dates as MM Top 20 (rolled to next open day).")

# ---------- Config: files bundled in repo ----------
MM_PATH = "mm_top_20.csv"
MAP_PATH = "ticker_map.csv"

DEFAULT_COMPETITORS = [
    "SPY", "QQQ", "VTI", "RSP", "QQQE",
    "IWM", "DIA", "ACWI", "EFA", "EEM",
    "XLK", "SMH", "XLE", "XLF", "XLV",
    "VNQ", "GLD", "TLT", "BTC-USD"
]

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    competitors = st.multiselect(
        "Competitors",
        options=DEFAULT_COMPETITORS,
        default=["SPY", "QQQ"]
    )
    price_field = st.selectbox("Price field", ["Adj Close", "Close", "Open"], index=0)
    auto_adjust = st.checkbox("Auto-adjust (recommended)", value=True)
    start_buffer_days = st.number_input("Start buffer (days before earliest buy)", min_value=0, max_value=30, value=5)
    business_days = st.checkbox("Business-day calendar & ffill", value=True)
    cache_ttl = st.select_slider("Refresh cache (minutes)", options=[5, 15, 30, 60, 180, 360], value=60)
    refresh = st.button("Refresh now", type="primary", use_container_width=True)

# ---------- Helpers ----------
def exchsym_to_yahoo(resolved: str) -> str | None:
    if not isinstance(resolved, str) or ":" not in resolved:
        return None
    exch, sym = resolved.split(":", 1)
    exch = exch.strip().upper()
    sym = sym.strip().upper()
    suffix_map = {
        "NASDAQ": "", "NYSE": "", "AMEX": "", "NYSEARCA": "",
        "LON": ".L", "LSE": ".L", "AMS": ".AS",
    }
    suf = suffix_map.get(exch, "")
    return sym + suf

def load_buys_path(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"Ticker","Date"}.issubset(df.columns):
        raise ValueError("mm_top_20.csv must have columns: Ticker, Date")
    out = df[["Ticker","Date"]].copy()
    out.columns = ["user_ticker","buy_date"]
    out["user_ticker"] = out["user_ticker"].astype(str).str.strip()
    out["buy_date"] = pd.to_datetime(out["buy_date"], errors="coerce")
    if out["buy_date"].isna().any():
        raise ValueError("Some buy dates in mm_top_20.csv could not be parsed.")
    return out.dropna(subset=["user_ticker","buy_date"]).reset_index(drop=True)

def load_map_path(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"User Ticker","Resolved Ticker"}.issubset(df.columns):
        raise ValueError("ticker_map.csv must have columns: User Ticker, Resolved Ticker, (optional) Currency")
    tmap = df[["User Ticker","Resolved Ticker"]].copy()
    tmap.columns = ["user_ticker","resolved"]
    tmap["user_ticker"] = tmap["user_ticker"].astype(str).str.strip()
    tmap["resolved"] = tmap["resolved"].astype(str).str.strip()
    tmap["yf_ticker"] = tmap["resolved"].apply(exchsym_to_yahoo)
    return tmap

# Cache Yahoo downloads for speed; invalidate when user hits Refresh or TTL expires
@st.cache_data(show_spinner=False, ttl=60*60)  # default 60 min, set dynamically later
def download_prices(symbols, start_date, end_date, field, auto_adjust_flag, _cache_tag):
    """
    Reliable per-symbol downloader: single-threaded + retries + clear diagnostics.
    """
    ok = {}
    failed = []

    def _fetch(sym: str):
        for attempt in range(3):  # small retry loop
            try:
                dfp = yf.download(
                    sym,
                    start=start_date,
                    end=end_date,         # exclusive; we already set end = tomorrow
                    interval="1d",
                    auto_adjust=auto_adjust_flag,
                    progress=False,
                    threads=False,        # <- important on Cloud
                    timeout=30
                )
                if dfp is not None and not dfp.empty:
                    return dfp
            except Exception:
                pass
        return None

    for sym in symbols:
        dfp = _fetch(sym)
        if dfp is None or dfp.empty:
            failed.append(sym)
            continue

        use_field = field
        if use_field not in dfp.columns:
            if field == "Adj Close" and "Close" in dfp.columns and auto_adjust_flag:
                use_field = "Close"
            else:
                failed.append(sym)
                continue

        s = dfp[use_field].copy()
        s.name = sym
        s = s.sort_index()
        s = s[~s.index.duplicated(keep="first")]
        ok[sym] = s

    if failed:
        st.warning(
            f"Price download skipped/failed for {len(failed)} tickers: "
            + ", ".join(failed[:12]) + ("…" if len(failed) > 12 else "")
        )

    if not ok:
        raise RuntimeError(
            f"No price data downloaded. start={start_date}, end={end_date}, field={field}, "
            f"auto_adjust={auto_adjust_flag}, symbols={symbols[:10]}{'…' if len(symbols)>10 else ''}"
        )
    return ok


def build_business_index(price_series, end_date, use_bdays: bool):
    if use_bdays:
        union_start = min(s.index.min().date() for s in price_series.values())
        union_end   = datetime.fromisoformat(end_date).date()
        return pd.bdate_range(union_start, union_end, freq="C")
    idx = pd.DatetimeIndex(sorted(set().union(*[s.index for s in price_series.values()])))
    return idx

def entry_index(date_index: pd.DatetimeIndex, target_dt: pd.Timestamp) -> pd.Timestamp | None:
    pos = date_index.searchsorted(target_dt.normalize())
    if pos >= len(date_index):
        return None
    return date_index[pos]

def aggregate_from_values_matrix(values_wide: pd.DataFrame, date_col_start_idx: int = 3) -> pd.DataFrame:
    value_cols = values_wide.columns[date_col_start_idx:]
    total_value = values_wide[value_cols].sum(axis=0)
    total_value.index = pd.to_datetime(total_value.index)
    total_value.name = "total_value"
    active_buys = (values_wide[value_cols] > 0).sum(axis=0)
    active_buys.index = total_value.index
    active_buys.name = "active_buys"
    cumulative_profit = (total_value - active_buys).rename("cumulative_profit")
    roi = (cumulative_profit / active_buys.replace(0, np.nan)).rename("roi")
    out = pd.concat([total_value, active_buys, cumulative_profit, roi], axis=1).reset_index(names="date")
    out["date"] = pd.to_datetime(out["date"])
    return out

def build_chart(benchmarks_df: pd.DataFrame):
    fig = go.Figure()
    for series_name, df in benchmarks_df.groupby("series"):
        df = df.sort_values("date")
        customdata = np.stack(
            [
                df["total_value"].to_numpy(),
                df["active_buys"].fillna(0).astype(int).to_numpy(),
                df["cumulative_profit"].to_numpy(),
                df["roi"].to_numpy(),
            ],
            axis=-1
        )
        fig.add_trace(
            go.Scatter(
                x=df["date"], y=df["roi"], mode="lines",
                name=series_name, line=dict(width=2),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{x|%Y-%m-%d}</b><br>"
                    + series_name + ": ROI %{y:.4f}<br>"
                    "Cumulative Profit: %{customdata[2]:.4f}<br>"
                    "Total Value: %{customdata[0]:.4f}<br>"
                    "Active Buys: %{customdata[1]:d}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title={
            "text": "MM Top 20 vs Competitors<br>"
                    "<sup style='color:gray'>Using level stake synchronised entry buys for competitors for each MM Top 20 buy</sup>",
            "x": 0.5, "y": 0.98,
        },
        xaxis=dict(title="Date", rangeslider=dict(visible=True), type="date"),
        yaxis=dict(title="ROI", rangemode="tozero"),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=0.99, yanchor="top", x=0.5, xanchor="center",
            bgcolor="rgba(255,255,255,0.7)", font=dict(size=11), itemwidth=70
        ),
        margin=dict(l=60, r=60, t=110, b=90),
    )
    return fig

# ---------- Main run ----------
try:
    buys = load_buys_path(MM_PATH)
    tmap = load_map_path(MAP_PATH)
except Exception as e:
    st.error(f"Failed to read CSVs: {e}")
    st.stop()

mapped = buys.merge(tmap[["user_ticker","yf_ticker"]], on="user_ticker", how="left")
mapped["yf_ticker"] = np.where(
    mapped["yf_ticker"].isna() | (mapped["yf_ticker"].str.len() == 0),
    mapped["user_ticker"].str.upper(),
    mapped["yf_ticker"]
)

END_DATE = (pd.Timestamp.utcnow().normalize() + pd.Timedelta(days=1)).date().isoformat()

start_date = (mapped["buy_date"].min() - pd.Timedelta(days=int(start_buffer_days))).date().isoformat()
all_syms = sorted(set(mapped["yf_ticker"].dropna().unique().tolist() + competitors))

# cache tag switches when user clicks refresh OR settings change
cache_tag = f"{price_field}-{auto_adjust}-{business_days}-{start_date}-{END_DATE}-{sorted(competitors)}-{refresh}-{cache_ttl}"

with st.spinner("Downloading prices from Yahoo Finance…"):
    # override default TTL with user choice
    download_prices.clear()  # ensure the custom ttl below applies fresh
    download_prices.ttl = 60 * int(cache_ttl)
    price_series = download_prices(all_syms, start_date, END_DATE, price_field, auto_adjust, cache_tag)

# Build date index & forward-fill
date_index = build_business_index(price_series, END_DATE, business_days)
prices = pd.DataFrame(index=date_index)
for sym, s in price_series.items():
    prices[sym] = s.reindex(date_index).ffill()

# Portfolio per-purchase values
per_purchase_values, row_keys = [], []
for _, row in mapped.iterrows():
    sym, bdt = row["yf_ticker"], row["buy_date"]
    if sym not in prices.columns:
        continue
    ent = entry_index(date_index, bdt)
    if ent is None:
        continue
    p0 = prices.loc[ent, sym]
    if pd.isna(p0) or p0 == 0:
        continue
    s_val = (prices[sym] / p0).where(prices.index >= ent, 0.0)
    per_purchase_values.append(s_val)
    row_keys.append((row["user_ticker"], row["buy_date"].date().isoformat(), sym))

if not per_purchase_values:
    st.error("No valid portfolio purchases after mapping/downloading.")
    st.stop()

per_purchase_matrix = pd.DataFrame(per_purchase_values)
per_purchase_matrix.columns = per_purchase_matrix.columns.strftime("%Y-%m-%d")
per_purchase_matrix.insert(0, "Buy Date", [k[1] for k in row_keys])
per_purchase_matrix.insert(0, "Ticker",   [k[0] for k in row_keys])
per_purchase_matrix.insert(2, "Yahoo Symbol", [k[2] for k in row_keys])

# Portfolio aggregates
portfolio_df = aggregate_from_values_matrix(per_purchase_matrix)
portfolio_df["series"] = "MM Top 20"

# Synchronized competitor entries (use actual entry dates of portfolio)
value_cols = per_purchase_matrix.columns[3:]
value_dt_index = pd.to_datetime(value_cols)
entry_dates = []
for _, r in per_purchase_matrix.iterrows():
    vals = r[value_cols].astype(float).to_numpy()
    nz = np.flatnonzero(vals > 0)
    if nz.size == 0: continue
    entry_dates.append(value_dt_index[nz[0]])

def competitor_timeseries_synced(sym: str) -> pd.DataFrame:
    if sym not in prices.columns: return pd.DataFrame()
    per_list = []
    for ent in entry_dates:
        pos = date_index.searchsorted(ent.normalize())
        if pos >= len(date_index): continue
        ent2 = date_index[pos]
        p0 = prices.loc[ent2, sym]
        if pd.isna(p0) or p0 == 0:
            pos2 = pos + 1
            if pos2 >= len(date_index): continue
            ent2 = date_index[pos2]
            p0 = prices.loc[ent2, sym]
            if pd.isna(p0) or p0 == 0: continue
        s_val = (prices[sym] / p0).where(prices.index >= ent2, 0.0)
        per_list.append(s_val)
    if not per_list: return pd.DataFrame()
    mat = pd.DataFrame(per_list)
    mat.columns = mat.columns.strftime("%Y-%m-%d")
    mat.insert(0, "Buy Date", [""] * len(mat))
    mat.insert(0, "Ticker",   [""] * len(mat))
    mat.insert(2, "Yahoo Symbol", [sym] * len(mat))
    ts = aggregate_from_values_matrix(mat)
    ts["series"] = sym
    return ts

bench_long = [portfolio_df]
for comp in competitors:
    cdf = competitor_timeseries_synced(comp)
    if not cdf.empty: bench_long.append(cdf)
benchmarks_df = pd.concat(bench_long, ignore_index=True)

# Plot
fig = build_chart(benchmarks_df)
st.plotly_chart(fig, use_container_width=True)

# Show last price date
last_date = benchmarks_df["date"].max().date()
st.caption(f"Last price date: **{last_date.isoformat()}** (auto-refresh every ~{cache_ttl} minutes or on demand via the button).")

# Download chart as self-contained HTML
html_bytes = fig.to_html(full_html=True, include_plotlyjs="inline").encode("utf-8")
st.download_button(
    label="Download chart as HTML",
    file_name=f"mm_top20_vs_competitors_{datetime.utcnow().date().isoformat()}.html",
    data=html_bytes,
    mime="text/html"
)
