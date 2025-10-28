# exodus_app.py — Plus500 Bankroll Tracker + SPY/QQQ Benchmarks
# --------------------------------------------------------------
# - Loads plus500_demo.csv from the repo (no upload UI)
# - Normalizes CR/LF before parsing
# - Bankroll = 40,000 + cumulative NetPL booked on CloseDate only
# - Competitors (SPY, QQQ): invest 40,000 on first trade's OpenDate (first valid price on/after),
#   then hold with daily value updates from prices_cache.parquet
# - Plotly chart with hover (bankroll, daily PnL, open trades, SPY/QQQ values)
# - Download chart as HTML
#
# Requirements:
#   pip install streamlit pandas plotly pyarrow

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

CSV_PATH = "plus500_demo.csv"
PARQUET_PATH = Path("prices_cache.parquet")
COMPETITORS = ["SPY", "QQQ"]
STARTING_BANKROLL = 40_000.0
TZ = ZoneInfo("Europe/London")  # only used to decide "today" (naive date)

# ---------- Page ----------
st.set_page_config(page_title="Plus500 Bankroll vs SPY & QQQ", layout="wide")
st.title("Plus500 Bankroll — vs SPY & QQQ")

st.caption(
    """
Bankroll books **NetPLInUserCurrency** on the **date of CloseTime** (when a trade closes).
Benchmarks invest **$40,000** on the **date of the first Plus500 trade open** (first valid market day),
then **hold** through the end of the chart.
"""
)

# ---------- Load CSV (repo file) & normalize CR/LF ----------
try:
    with open(CSV_PATH, "rb") as f:
        b = f.read()
    b = b.replace(b"\r\r\n", b"\n").replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    with open(CSV_PATH, "wb") as f:
        f.write(b)
except FileNotFoundError:
    st.error(f"CSV not found: {CSV_PATH}")
    st.stop()
except Exception as e:
    st.error(f"Failed to read/normalize CSV: {e}")
    st.stop()

try:
    pdf = pd.read_csv(CSV_PATH)
except Exception as e:
    st.error(f"Failed to parse CSV: {e}")
    st.stop()

# ---------- Parse & normalize ----------
# CSV is dd/mm/YYYY HH:MM:SS
pdf["OpenTime"]  = pd.to_datetime(pdf.get("OpenTime"),  errors="coerce", dayfirst=True)
pdf["CloseTime"] = pd.to_datetime(pdf.get("CloseTime"), errors="coerce", dayfirst=True)

pdf["OpenDate"]  = pdf["OpenTime"].dt.normalize()
pdf["CloseDate"] = pdf["CloseTime"].dt.normalize()

pdf["NetPLInUserCurrency"] = pd.to_numeric(pdf.get("NetPLInUserCurrency"), errors="coerce")

if not pdf["OpenDate"].notna().any():
    st.error("No valid OpenDate values after parsing. Check the CSV timestamps.")
    st.stop()

# ---------- Calendar index ----------
start_ts = pdf["OpenDate"].min()                  # earliest open date (tz-naive Timestamp)
today_date = datetime.now(TZ).date()              # 'today' in Europe/London, as date
today_ts = pd.Timestamp(today_date)               # tz-naive midnight

if start_ts > today_ts:
    start_ts = today_ts

date_index = pd.date_range(start=start_ts, end=today_ts, freq="D")
date_df = pd.DataFrame({"date": date_index})

# ---------- Daily PnL on CloseDate ----------
daily_pnl = (
    pdf.groupby("CloseDate", dropna=False)["NetPLInUserCurrency"]
      .sum()
      .rename("daily_pnl")
      .reset_index()
      .rename(columns={"CloseDate": "date"})
)
daily_pnl["date"] = pd.to_datetime(daily_pnl["date"])

daily = (
    date_df.merge(daily_pnl, on="date", how="left")
           .assign(daily_pnl=lambda d: d["daily_pnl"].fillna(0.0))
           .sort_values("date", ignore_index=True)
)

# ---------- Open-trade count per day (sweep-line) ----------
events_up = (
    pdf.loc[pdf["OpenDate"].notna(), ["OpenDate"]]
       .rename(columns={"OpenDate": "date"})
       .assign(delta=1)
)
events_down = (
    pdf.loc[pdf["CloseDate"].notna(), ["CloseDate"]]
       .rename(columns={"CloseDate": "date"})
       .assign(date=lambda d: d["date"] + pd.Timedelta(days=1), delta=-1)
)
events = pd.concat([events_up, events_down], ignore_index=True)

delta_series = (
    events.groupby("date")["delta"].sum()
          .reindex(date_index, fill_value=0)
)
open_trades_series = delta_series.cumsum().astype(int)
open_trades = (
    open_trades_series.rename("open_trades")
    .rename_axis("date")
    .reset_index()
)

# ---------- Bankroll series ----------
daily = daily.merge(open_trades, on="date", how="left")
daily["open_trades"] = daily["open_trades"].fillna(0).astype(int)
daily["bankroll"] = STARTING_BANKROLL + daily["daily_pnl"].cumsum()

# ---------- Competitors: load prices cache and build $40k hold series ----------
def load_prices(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index, errors="coerce").tz_localize(None)
        df = df[~df.index.isna()].sort_index()
        return df
    except Exception as e:
        st.warning(f"Could not read {path}: {e}")
        return None

prices = load_prices(PARQUET_PATH)
competitor_values = {}

def competitor_hold_value(sym: str) -> pd.Series | None:
    if prices is None or sym not in prices.columns:
        return None
    s = prices[sym].copy()           # Close
    s = s.dropna()
    if s.empty:
        return None
    # Normalize to naive dates (00:00)
    s.index = pd.to_datetime(s.index).normalize()
    s = s[~s.index.duplicated()].sort_index()

    # Reindex to chart calendar with ffill so we have a continuous series
    s = s.reindex(date_index).ffill()

    # Entry: first valid on/after start_ts
    entry_idx = s.loc[s.index >= start_ts].first_valid_index()
    if entry_idx is None or pd.isna(s.loc[entry_idx]):
        return None

    shares = STARTING_BANKROLL / float(s.loc[entry_idx])
    val = s * shares
    # zero/NaN before entry for a clean plot
    val = val.where(val.index >= entry_idx, np.nan)
    return val

for sym in COMPETITORS:
    ser = competitor_hold_value(sym)
    if ser is not None:
        competitor_values[sym] = ser

# ---------- Plotly chart ----------
fig = go.Figure()

# Bankroll
bankroll_custom = np.stack(
    [
        daily["bankroll"].to_numpy(),
        daily["daily_pnl"].to_numpy(),
        daily["open_trades"].to_numpy(),
    ],
    axis=-1,
)
fig.add_trace(
    go.Scatter(
        x=daily["date"],
        y=daily["bankroll"],
        mode="lines",
        name="Bankroll (Plus500)",
        line=dict(width=3),
        customdata=bankroll_custom,
        hovertemplate=(
            "<b>%{x|%Y-%m-%d}</b><br>"
            "Bankroll: %{y:.2f}<br>"
            "Daily PnL: %{customdata[1]:.2f}<br>"
            "Open Trades: %{customdata[2]:d}<extra></extra>"
        ),
    )
)

# Competitors
for sym, ser in competitor_values.items():
    fig.add_trace(
        go.Scatter(
            x=ser.index,
            y=ser.values,
            mode="lines",
            name=f"{sym} (40k hold)",
            line=dict(width=2),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>" + f"{sym} value: " + "%{y:.2f}<extra></extra>",
        )
    )

fig.update_layout(
    template="plotly_white",
    xaxis=dict(title="Date", type="date"),
    yaxis=dict(title="Value (USD)", rangemode="tozero"),
    hovermode="x unified",
    legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0.5, xanchor="center"),
    margin=dict(l=60, r=60, t=60, b=60),
)

st.plotly_chart(fig, use_container_width=True)

# ---------- Footer & download ----------
last_close = pdf["CloseDate"].max()
last_close_str = "N/A" if pd.isna(last_close) else pd.to_datetime(last_close).date().isoformat()

if prices is None:
    price_note = "Prices cache not found — competitor lines omitted."
else:
    last_price_date = pd.to_datetime(prices.index.max()).date()
    price_note = f"Prices cache last date: **{last_price_date.isoformat()}**."

st.caption(
    f"Bankroll window: **{start_ts.date().isoformat()} → {today_ts.date().isoformat()}** · "
    f"Last CloseDate in data: **{last_close_str}** · {price_note}"
)

html_bytes = fig.to_html(full_html=True, include_plotlyjs="inline").encode("utf-8")
st.download_button(
    label="Download chart as HTML",
    file_name=f"plus500_vs_spy_qqq_{pd.Timestamp.utcnow().date().isoformat()}.html",
    data=html_bytes,
    mime="text/html",
)
