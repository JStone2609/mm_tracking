# app.py — Streamlit Cloud, parquet-only reader (no live Yahoo calls)

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------- Page chrome ----------------
st.set_page_config(page_title="MM Top 20 vs Competitors — ROI", layout="wide")
st.title("MM Top 20 vs Competitors — ROI")
st.caption(
    "Level stakes; competitors buy on the same entry dates as MM Top 20 "
    "(rolled forward to next available price date)."
)

# ---------------- Files in repo ----------------
MM_PATH = Path("mm_top_20.csv")           # cols: Ticker, Date
MAP_PATH = Path("ticker_map.csv")         # cols: User Ticker, Resolved Ticker [, Currency]
PARQUET_PATH = Path("prices_cache.parquet")  # written by your GitHub Action

DEFAULT_COMPETITORS = [
    "SPY", "QQQ", "VTI", "RSP", "QQQE",
    "IWM", "DIA", "ACWI", "EFA", "EEM",
    "XLK", "SMH", "XLE", "XLF", "XLV",
    "VNQ", "GLD", "TLT", "BTC-USD",
]

# ---------------- Sidebar (minimal) ----------------
with st.sidebar:
    st.header("Settings")
    competitors = st.multiselect(
        "Competitors",
        options=DEFAULT_COMPETITORS,
        default=["SPY", "QQQ"],
        help="Comparison series built using the same entry dates as the portfolio.",
    )
    if st.button("Reload data", use_container_width=True):
        st.cache_data.clear()

# ---------------- Helpers ----------------
def exchsym_to_yahoo(resolved: str) -> str | None:
    """Turn 'EXCH:SYM' into a Yahoo-compatible symbol (suffix mapping)."""
    if not isinstance(resolved, str) or ":" not in resolved:
        return None
    exch, sym = resolved.split(":", 1)
    exch = exch.strip().upper()
    sym = sym.strip().upper()
    suffix_map = {
        "NASDAQ": "", "NYSE": "", "AMEX": "", "NYSEARCA": "",
        "LON": ".L", "LSE": ".L", "AMS": ".AS",
    }
    return sym + suffix_map.get(exch, "")

def load_buys(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"Ticker", "Date"}.issubset(df.columns):
        raise ValueError("mm_top_20.csv must have columns: Ticker, Date")
    out = df[["Ticker", "Date"]].rename(columns={"Ticker": "user_ticker", "Date": "buy_date"})
    out["user_ticker"] = out["user_ticker"].astype(str).str.strip()
    out["buy_date"] = pd.to_datetime(out["buy_date"], errors="coerce")
    if out["buy_date"].isna().any():
        bad = out[out["buy_date"].isna()].head(5)
        raise ValueError(f"Some buy dates could not be parsed. Example rows:\n{bad}")
    return out.dropna().reset_index(drop=True)

def load_map(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"User Ticker", "Resolved Ticker"}.issubset(df.columns):
        raise ValueError("ticker_map.csv must contain: User Ticker, Resolved Ticker")
    tmap = df[["User Ticker", "Resolved Ticker"]].rename(
        columns={"User Ticker": "user_ticker", "Resolved Ticker": "resolved"}
    )
    tmap["user_ticker"] = tmap["user_ticker"].astype(str).str.strip()
    tmap["resolved"] = tmap["resolved"].astype(str).str.strip()
    tmap["yf_ticker"] = tmap["resolved"].apply(exchsym_to_yahoo)
    return tmap

@st.cache_data(show_spinner=False)
def load_prices_parquet(path: Path) -> pd.DataFrame:
    """
    Read the parquet your GitHub Action writes.
    Expected shape: index = DatetimeIndex; columns = Yahoo symbols; values = Close (or Adj Close).
    """
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError("prices_cache.parquet is empty.")
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    # drop all-null columns defensively
    df = df.loc[:, df.notna().any(axis=0)]
    return df

def entry_index(date_index: pd.DatetimeIndex, target_dt: pd.Timestamp) -> pd.Timestamp | None:
    """First available price date on/after the target date."""
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

def build_chart(benchmarks_df: pd.DataFrame) -> go.Figure:
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

    # No in-figure title; legend parked above axes and below Streamlit title
    fig.update_layout(
        xaxis=dict(title="Date", rangeslider=dict(visible=True), type="date"),
        yaxis=dict(title="ROI", rangemode="tozero"),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=1.02, yanchor="bottom",   # put legend just above plotting area
            x=0.5, xanchor="center",
            bgcolor="rgba(255,255,255,0.65)",
            font=dict(size=11),
            itemwidth=70,
        ),
        margin=dict(l=60, r=60, t=60, b=80),
    )
    return fig

# ---------------- Main ----------------
# Load inputs
try:
    buys = load_buys(MM_PATH)
    tmap = load_map(MAP_PATH)
except Exception as e:
    st.error(f"Failed to read CSVs: {e}")
    st.stop()

if not PARQUET_PATH.exists():
    st.error(
        "Missing prices_cache.parquet.\n\n"
        "Make sure your GitHub Action has run and committed the file."
    )
    st.stop()

try:
    prices = load_prices_parquet(PARQUET_PATH)   # index: dates, columns: symbols
except Exception as e:
    st.error(f"Failed to load prices_cache.parquet: {e}")
    st.stop()

# map user tickers -> yahoo symbols
mapped = buys.merge(tmap[["user_ticker", "yf_ticker"]], on="user_ticker", how="left")
mapped["yf_ticker"] = np.where(
    mapped["yf_ticker"].isna() | (mapped["yf_ticker"].astype(str).str.len() == 0),
    mapped["user_ticker"].astype(str).str.upper(),
    mapped["yf_ticker"]
)

# Keep only symbols we actually have in the parquet
available_syms = set(prices.columns.astype(str))
missing_pf = sorted(set(mapped["yf_ticker"]) - available_syms)
if missing_pf:
    st.warning(
        "Portfolio symbols missing from parquet (skipped): "
        + ", ".join(missing_pf[:12]) + ("…" if len(missing_pf) > 12 else "")
    )
mapped = mapped[mapped["yf_ticker"].isin(available_syms)].reset_index(drop=True)
if mapped.empty:
    st.error("None of the mapped portfolio symbols are present in the parquet.")
    st.stop()

# Competitors present check
competitors_present = [c for c in competitors if c in available_syms]
missing_comp = sorted(set(competitors) - set(competitors_present))
if missing_comp:
    st.info("Missing competitors (not in parquet): " + ", ".join(missing_comp))

date_index = prices.index  # use the exact dates in parquet (no forward-fill)

# Portfolio per-purchase values (level stake, 1 unit each)
per_purchase_values, row_keys = [], []
for _, row in mapped.iterrows():
    sym, bdt = row["yf_ticker"], row["buy_date"]
    if sym not in prices.columns:
        continue
    ent = entry_index(date_index, bdt)
    if ent is None:
        continue
    p0 = prices.at[ent, sym]
    if pd.isna(p0) or p0 == 0:
        continue
    s_val = (prices[sym] / p0).where(date_index >= ent, 0.0)
    per_purchase_values.append(s_val)
    row_keys.append((row["user_ticker"], row["buy_date"].date().isoformat(), sym))

# Build the per-purchase matrix
per_purchase_matrix = pd.DataFrame(per_purchase_values)

# Ensure the date columns are real datetimes, then format to YYYY-MM-DD strings
if not isinstance(per_purchase_matrix.columns, pd.DatetimeIndex):
    try:
        per_purchase_matrix.columns = pd.to_datetime(per_purchase_matrix.columns)
    except Exception:
        # If they were already strings or mixed, leave as-is; we’ll stringify below
        pass

if isinstance(per_purchase_matrix.columns, pd.DatetimeIndex):
    per_purchase_matrix.columns = per_purchase_matrix.columns.strftime("%Y-%m-%d")
else:
    per_purchase_matrix.columns = [str(c) for c in per_purchase_matrix.columns]

# Add the descriptive columns
per_purchase_matrix.insert(0, "Buy Date", [k[1] for k in row_keys])
per_purchase_matrix.insert(0, "Ticker",   [k[0] for k in row_keys])
per_purchase_matrix.insert(2, "Yahoo Symbol", [k[2] for k in row_keys])


# Portfolio aggregate series
portfolio_df = aggregate_from_values_matrix(per_purchase_matrix)
portfolio_df["series"] = "MM Top 20"

# Build synchronized competitor series
value_cols = per_purchase_matrix.columns[3:]
value_dt_index = pd.to_datetime(value_cols)
entry_dates = []
for _, r in per_purchase_matrix.iterrows():
    vals = r[value_cols].astype(float).to_numpy()
    nz = np.flatnonzero(vals > 0)
    if nz.size:
        entry_dates.append(value_dt_index[nz[0]])

def competitor_timeseries_synced(sym: str) -> pd.DataFrame:
    if sym not in prices.columns:
        return pd.DataFrame()
    per_list = []
    for ent in entry_dates:
        ent2 = entry_index(date_index, ent)
        if ent2 is None:
            continue
        p0 = prices.at[ent2, sym]
        if pd.isna(p0) or p0 == 0:
            continue
        s_val = (prices[sym] / p0).where(date_index >= ent2, 0.0)
        per_list.append(s_val)
    if not per_list:
        return pd.DataFrame()
    mat = pd.DataFrame(per_list)
    mat.columns = mat.index.strftime("%Y-%m-%d")
    mat.insert(0, "Buy Date", [""] * len(mat))
    mat.insert(0, "Ticker", [""] * len(mat))
    mat.insert(2, "Yahoo Symbol", [sym] * len(mat))
    ts = aggregate_from_values_matrix(mat)
    ts["series"] = sym
    return ts

bench_long = [portfolio_df]
for comp in competitors_present:
    cdf = competitor_timeseries_synced(comp)
    if not cdf.empty:
        bench_long.append(cdf)

benchmarks_df = pd.concat(bench_long, ignore_index=True)

# Plot
fig = build_chart(benchmarks_df)
st.plotly_chart(fig, use_container_width=True)

# Footer info
last_date = pd.to_datetime(prices.index.max()).date()
st.caption(
    f"Last price date in cache: **{last_date.isoformat()}**. "
    "This app reads pre-fetched prices only; updates occur when your GitHub Action commits a new parquet."
)

# Optional: self-contained HTML download
html_bytes = fig.to_html(full_html=True, include_plotlyjs="inline").encode("utf-8")
st.download_button(
    label="Download chart as HTML",
    file_name=f"mm_top20_vs_competitors_{datetime.utcnow().date().isoformat()}.html",
    data=html_bytes,
    mime="text/html",
)
