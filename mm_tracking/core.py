from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    import streamlit as st
except ModuleNotFoundError:
    class _StreamlitShim:
        @staticmethod
        def cache_data(*_args, **_kwargs):
            def decorator(func):
                return func

            return decorator

    st = _StreamlitShim()


YAHOO_SUFFIXES = {
    "NASDAQ": "",
    "NYSE": "",
    "AMEX": "",
    "NYSEARCA": "",
    "LON": ".L",
    "LSE": ".L",
    "AMS": ".AS",
    "BME": ".MC",
    "MCE": ".MC",
}


@dataclass(frozen=True)
class EqualWeightAppConfig:
    page_title: str
    title: str
    caption: str
    portfolio_label: str
    buys_path: Path
    map_path: Path
    parquet_path: Path
    competitors: list[str] | list[tuple[str, str]]
    download_prefix: str
    price_caption_prefix: str
    expected_update_utc: str | None = None
    data_kind: str = "equity"
    missing_cache_message: str = "Missing prices cache. The GitHub Action must write it first."


@dataclass(frozen=True)
class TradeAppConfig:
    page_title: str
    title: str
    caption: str
    portfolio_label: str
    trades_path: Path
    map_path: Path
    parquet_path: Path
    competitors: list[str]
    download_prefix: str
    expected_update_utc: str | None = None
    missing_cache_message: str = "Missing prices cache. The GitHub Action must write it first."


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
    suffix = YAHOO_SUFFIXES.get(exch, "")
    return sym + suffix


def load_equal_weight_buys(path: Path, ticker_column: str = "Ticker") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={ticker_column: "user_ticker", "Date": "buy_date"})
    df["user_ticker"] = df["user_ticker"].astype(str).str.strip().str.upper()
    df["buy_date"] = pd.to_datetime(df["buy_date"], errors="coerce").dt.normalize()
    return df.dropna(subset=["user_ticker", "buy_date"]).reset_index(drop=True)


def load_trade_book(path: Path) -> pd.DataFrame:
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


def load_equity_map(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"User Ticker": "user_ticker", "Resolved Ticker": "resolved"})
    df["user_ticker"] = df["user_ticker"].astype(str).str.strip().str.upper()
    df["resolved"] = df["resolved"].astype(str).str.strip()
    df["yf_ticker"] = df["resolved"].apply(exchsym_to_yahoo)
    return df[["user_ticker", "yf_ticker"]]


def load_crypto_map(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"User Symbol": "user_ticker", "CoinGecko ID": "cg_id", "Currency": "currency"})
    df["user_ticker"] = df["user_ticker"].astype(str).str.strip().str.upper()
    df["cg_id"] = df["cg_id"].astype(str).str.strip().str.lower()
    return df[["user_ticker", "cg_id"]]


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
    return df.loc[:, df.notna().any(axis=0)]


def read_prices_for_app(path: Path) -> pd.DataFrame:
    version = path.stat().st_mtime_ns if path.exists() else 0
    return load_prices_parquet(path, version)


def first_valid_on_or_after(series: pd.Series, when: pd.Timestamp) -> pd.Timestamp | None:
    subset = series.loc[series.index >= when]
    subset = subset[subset.notna()]
    if subset.empty:
        return None
    return subset.index[0]


def map_equity_symbols(positions: pd.DataFrame, tmap: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    mapped = positions.merge(tmap, on="user_ticker", how="left")
    mapped["yf_ticker"] = mapped["yf_ticker"].where(
        mapped["yf_ticker"].notna() & (mapped["yf_ticker"].astype(str).str.len() > 0),
        mapped["user_ticker"].astype(str).str.upper(),
    )
    available = set(prices.columns.astype(str))
    return mapped[mapped["yf_ticker"].isin(available)].reset_index(drop=True)


def map_crypto_symbols(positions: pd.DataFrame, tmap: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    mapped = positions.merge(tmap, on="user_ticker", how="left")
    mapped["cg_id"] = mapped["cg_id"].fillna("").astype(str)
    available = set(prices.columns.astype(str))
    return mapped[mapped["cg_id"].isin(available)].reset_index(drop=True)


def build_equal_weight_series(
    mapped: pd.DataFrame,
    prices: pd.DataFrame,
    id_column: str,
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    first_buy_date = pd.to_datetime(mapped["buy_date"].min()).normalize()
    date_index = prices.index[prices.index >= first_buy_date]
    prices = prices.loc[date_index]

    per_purchase_values: list[pd.Series] = []
    row_keys: list[tuple[str, str, str]] = []

    for _, row in mapped.iterrows():
        symbol = row[id_column]
        buy_date = row["buy_date"]
        series = prices[symbol]
        entry = first_valid_on_or_after(series, buy_date)
        if entry is None:
            continue
        p0 = series.at[entry]
        if pd.isna(p0) or p0 == 0:
            continue
        rel = (series / p0).where(date_index >= entry, 0.0)
        per_purchase_values.append(rel)
        row_keys.append((row["user_ticker"], entry.date().isoformat(), symbol))

    if not per_purchase_values:
        return pd.DataFrame(), []

    permat = pd.DataFrame(per_purchase_values)
    permat.columns = permat.columns.strftime("%Y-%m-%d")
    permat.insert(0, "Buy Date", [k[1] for k in row_keys])
    permat.insert(0, "Ticker", [k[0] for k in row_keys])
    permat.insert(2, "Symbol", [k[2] for k in row_keys])

    value_cols = permat.columns[3:]
    value_dt_index = pd.to_datetime(value_cols)
    entry_dates: list[pd.Timestamp] = []
    for _, row in permat.iterrows():
        vals = row[value_cols].astype(float).to_numpy()
        non_zero = np.flatnonzero(vals > 0)
        if non_zero.size:
            entry_dates.append(value_dt_index[non_zero[0]])

    return permat, entry_dates


def aggregate_equal_weight_matrix(values_wide: pd.DataFrame, date_col_start_idx: int = 3) -> pd.DataFrame:
    dates = pd.to_datetime(values_wide.columns[date_col_start_idx:])
    values = values_wide.iloc[:, date_col_start_idx:].to_numpy(dtype=float)
    total_value = pd.Series(values.sum(axis=0), index=dates, name="total_value")
    active_buys = pd.Series((values > 0).sum(axis=0), index=dates, name="active_buys")
    cumulative_profit = (total_value - active_buys).rename("cumulative_profit")
    roi = (cumulative_profit / active_buys.replace(0, np.nan)).rename("roi")
    out = pd.concat([total_value, active_buys, cumulative_profit, roi], axis=1).reset_index(names="date")
    out["date"] = pd.to_datetime(out["date"])
    return out


def build_equal_weight_competitor_series(
    symbol: str,
    label: str,
    prices: pd.DataFrame,
    date_index: pd.DatetimeIndex,
    entry_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    if symbol not in prices.columns:
        return pd.DataFrame()

    series = prices[symbol]
    per_list = []
    for entry in entry_dates:
        competitor_entry = first_valid_on_or_after(series, entry)
        if competitor_entry is None:
            continue
        p0 = series.at[competitor_entry]
        if pd.isna(p0) or p0 == 0:
            continue
        rel = (series / p0).where(date_index >= competitor_entry, 0.0)
        per_list.append(rel)

    if not per_list:
        return pd.DataFrame()

    mat = pd.DataFrame(per_list)
    mat.columns = mat.columns.strftime("%Y-%m-%d")
    mat.insert(0, "Buy Date", [""] * len(mat))
    mat.insert(0, "Ticker", [""] * len(mat))
    mat.insert(2, "Symbol", [symbol] * len(mat))
    ts = aggregate_equal_weight_matrix(mat)
    ts["series"] = label
    return ts


def build_position_series(
    price_series: pd.Series,
    date_index: pd.DatetimeIndex,
    entry_date: pd.Timestamp,
    amount: float,
    action: str,
) -> tuple[pd.Series | None, pd.Series | None, pd.Timestamp | None]:
    entry = first_valid_on_or_after(price_series, entry_date)
    if entry is None:
        return None, None, None

    p0 = price_series.at[entry]
    if pd.isna(p0) or p0 == 0:
        return None, None, None

    ratio = price_series / p0
    value = (2.0 - ratio) * amount if action == "SELL" else ratio * amount
    value = value.where(date_index >= entry, 0.0)
    invested = pd.Series(np.where(date_index >= entry, amount, 0.0), index=date_index, dtype=float)
    return value, invested, entry


def aggregate_trade_matrices(value_wide: pd.DataFrame, invested_wide: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(value_wide.columns)
    values = value_wide.to_numpy(dtype=float)
    invested = invested_wide.to_numpy(dtype=float)

    total_value = pd.Series(values.sum(axis=0), index=dates, name="total_value")
    invested_amount = pd.Series(invested.sum(axis=0), index=dates, name="invested_amount")
    active_trades = pd.Series((invested > 0).sum(axis=0), index=dates, name="active_trades")
    cumulative_profit = (total_value - invested_amount).rename("cumulative_profit")
    roi = (cumulative_profit / invested_amount.replace(0, np.nan)).rename("roi")

    out = pd.concat(
        [total_value, invested_amount, active_trades, cumulative_profit, roi],
        axis=1,
    ).reset_index(names="date")
    out["date"] = pd.to_datetime(out["date"])
    return out


def filter_valid_roi_dates(benchmarks_df: pd.DataFrame) -> pd.DataFrame:
    valid_dates = benchmarks_df.groupby("date")["roi"].apply(lambda x: x.notna().any())
    return benchmarks_df[benchmarks_df["date"].isin(valid_dates[valid_dates].index)].reset_index(drop=True)


def build_buy_chart(benchmarks_df: pd.DataFrame, start_date: pd.Timestamp, primary_series: str) -> go.Figure:
    fig = go.Figure()
    groups = list(benchmarks_df.groupby("series"))
    primary_groups = [group for group in groups if group[0] == primary_series]
    other_groups = [group for group in groups if group[0] != primary_series]

    for name, df in primary_groups + other_groups:
        df = df.sort_values("date")
        custom = np.stack(
            [
                df["total_value"].to_numpy(),
                df["active_buys"].fillna(0).astype(int).to_numpy(),
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
                    + name
                    + ": ROI %{y:.2%}<br>"
                    "Cumulative Profit: %{customdata[2]:.4f}<br>"
                    "Total Value: %{customdata[0]:.4f}<br>"
                    "Active Buys: %{customdata[1]:d}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        template="plotly_white",
        xaxis=dict(title="Date", type="date", range=[pd.to_datetime(start_date), None], rangeslider=dict(visible=False)),
        yaxis=dict(title="ROI", rangemode="tozero", tickformat=".0%"),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0.5, xanchor="center"),
        margin=dict(l=60, r=60, t=60, b=80),
    )
    return fig


def build_trade_chart(benchmarks_df: pd.DataFrame, start_date: pd.Timestamp, primary_series: str) -> go.Figure:
    fig = go.Figure()
    groups = list(benchmarks_df.groupby("series"))
    primary_groups = [group for group in groups if group[0] == primary_series]
    other_groups = [group for group in groups if group[0] != primary_series]

    for name, df in primary_groups + other_groups:
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
                    + name
                    + ": ROI %{y:.2%}<br>"
                    "Cumulative Profit: %{customdata[3]:.4f}<br>"
                    "Total Value: %{customdata[0]:.4f}<br>"
                    "Invested Amount: %{customdata[1]:.4f}<br>"
                    "Active Trades: %{customdata[2]:d}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        template="plotly_white",
        xaxis=dict(title="Date", type="date", range=[pd.to_datetime(start_date), None], rangeslider=dict(visible=False)),
        yaxis=dict(title="ROI", rangemode="tozero", tickformat=".0%"),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0.5, xanchor="center"),
        margin=dict(l=60, r=60, t=60, b=80),
    )
    return fig


def render_equal_weight_app(config: EqualWeightAppConfig) -> None:
    st.set_page_config(page_title=config.page_title, layout="wide")
    st.title(config.title)
    st.caption(config.caption)

    try:
        buys = load_equal_weight_buys(config.buys_path)
        tmap = load_crypto_map(config.map_path) if config.data_kind == "crypto" else load_equity_map(config.map_path)
    except Exception as exc:
        st.error(f"Failed to read CSVs: {exc}")
        st.stop()

    if not config.parquet_path.exists():
        st.error(config.missing_cache_message)
        st.stop()

    try:
        prices = read_prices_for_app(config.parquet_path)
    except Exception as exc:
        st.error(f"Failed to load {config.parquet_path.name}: {exc}")
        st.stop()

    mapped = map_crypto_symbols(buys, tmap, prices) if config.data_kind == "crypto" else map_equity_symbols(buys, tmap, prices)
    if mapped.empty:
        missing_target = "CoinGecko IDs" if config.data_kind == "crypto" else "symbols"
        st.error(f"No portfolio {missing_target} are present in the price cache.")
        st.stop()

    first_buy_date = pd.to_datetime(buys["buy_date"].min()).normalize()
    date_index = prices.index[prices.index >= first_buy_date]
    prices = prices.loc[date_index]

    id_column = "cg_id" if config.data_kind == "crypto" else "yf_ticker"
    permat, entry_dates = build_equal_weight_series(mapped, prices, id_column=id_column)
    if permat.empty:
        st.error("No valid portfolio entries after symbol/date alignment.")
        st.stop()

    portfolio_df = aggregate_equal_weight_matrix(permat)
    portfolio_df["series"] = config.portfolio_label

    bench_long = [portfolio_df]
    competitor_pairs = config.competitors if config.data_kind == "crypto" else [(sym, sym) for sym in config.competitors]
    for competitor_symbol, label in competitor_pairs:
        competitor_df = build_equal_weight_competitor_series(
            symbol=competitor_symbol,
            label=label,
            prices=prices,
            date_index=date_index,
            entry_dates=entry_dates,
        )
        if not competitor_df.empty:
            bench_long.append(competitor_df)

    benchmarks_df = pd.concat(bench_long, ignore_index=True)
    benchmarks_df = benchmarks_df[benchmarks_df["date"] >= first_buy_date].reset_index(drop=True)
    benchmarks_df = filter_valid_roi_dates(benchmarks_df)

    fig = build_buy_chart(benchmarks_df, start_date=first_buy_date, primary_series=config.portfolio_label)
    st.plotly_chart(fig, use_container_width=True)

    last_date = pd.to_datetime(prices.index.max()).date()
    if config.expected_update_utc:
        st.caption(
            f"{config.price_caption_prefix} **{last_date.isoformat()}**. "
            f"Updates happen once per trading day via GitHub Actions (≈{config.expected_update_utc}, after U.S. market close)."
        )
    else:
        st.caption(f"{config.price_caption_prefix} **{last_date.isoformat()}**.")

    html_bytes = fig.to_html(full_html=True, include_plotlyjs="inline").encode("utf-8")
    st.download_button(
        label="Download chart as HTML",
        file_name=f"{config.download_prefix}_{pd.Timestamp.utcnow().date().isoformat()}.html",
        data=html_bytes,
        mime="text/html",
    )


def render_trade_app(config: TradeAppConfig) -> None:
    st.set_page_config(page_title=config.page_title, layout="wide")
    st.title(config.title)
    st.caption(config.caption)

    try:
        trades = load_trade_book(config.trades_path)
        tmap = load_equity_map(config.map_path)
    except Exception as exc:
        st.error(f"Failed to read CSVs: {exc}")
        st.stop()

    if not config.parquet_path.exists():
        st.error(config.missing_cache_message)
        st.stop()

    try:
        prices = read_prices_for_app(config.parquet_path)
    except Exception as exc:
        st.error(f"Failed to load {config.parquet_path.name}: {exc}")
        st.stop()

    mapped = map_equity_symbols(trades, tmap, prices)
    if mapped.empty:
        st.error("No portfolio symbols are present in the price cache.")
        st.stop()

    first_trade_date = pd.to_datetime(trades["trade_date"].min()).normalize()
    date_index = prices.index[prices.index >= first_trade_date]
    prices = prices.loc[date_index]

    portfolio_values: list[pd.Series] = []
    portfolio_invested: list[pd.Series] = []
    entry_records: list[dict[str, object]] = []

    for _, row in mapped.iterrows():
        value_s, invested_s, entry = build_position_series(
            price_series=prices[row["yf_ticker"]],
            date_index=date_index,
            entry_date=row["trade_date"],
            amount=float(row["amount"]),
            action=row["action"],
        )
        if value_s is None or invested_s is None or entry is None:
            continue
        portfolio_values.append(value_s)
        portfolio_invested.append(invested_s)
        entry_records.append({"entry_date": entry, "amount": float(row["amount"]), "action": row["action"]})

    if not portfolio_values:
        st.error("No valid portfolio entries after symbol/date alignment.")
        st.stop()

    portfolio_value_mat = pd.DataFrame(portfolio_values)
    portfolio_invested_mat = pd.DataFrame(portfolio_invested)
    portfolio_value_mat.columns = portfolio_value_mat.columns.strftime("%Y-%m-%d")
    portfolio_invested_mat.columns = portfolio_invested_mat.columns.strftime("%Y-%m-%d")

    portfolio_df = aggregate_trade_matrices(portfolio_value_mat, portfolio_invested_mat)
    portfolio_df["series"] = config.portfolio_label

    bench_long = [portfolio_df]
    for competitor in config.competitors:
        if competitor not in prices.columns:
            continue
        value_list: list[pd.Series] = []
        invested_list: list[pd.Series] = []
        for record in entry_records:
            value_s, invested_s, _ = build_position_series(
                price_series=prices[competitor],
                date_index=date_index,
                entry_date=record["entry_date"],
                amount=float(record["amount"]),
                action=str(record["action"]),
            )
            if value_s is None or invested_s is None:
                continue
            value_list.append(value_s)
            invested_list.append(invested_s)

        if not value_list:
            continue

        value_mat = pd.DataFrame(value_list)
        invested_mat = pd.DataFrame(invested_list)
        value_mat.columns = value_mat.columns.strftime("%Y-%m-%d")
        invested_mat.columns = invested_mat.columns.strftime("%Y-%m-%d")

        competitor_df = aggregate_trade_matrices(value_mat, invested_mat)
        competitor_df["series"] = competitor
        bench_long.append(competitor_df)

    benchmarks_df = pd.concat(bench_long, ignore_index=True)
    benchmarks_df = benchmarks_df[benchmarks_df["date"] >= first_trade_date].reset_index(drop=True)
    benchmarks_df = filter_valid_roi_dates(benchmarks_df)

    fig = build_trade_chart(benchmarks_df, start_date=first_trade_date, primary_series=config.portfolio_label)
    st.plotly_chart(fig, use_container_width=True)

    last_date = pd.to_datetime(prices.index.max()).date()
    st.caption(
        f"Last price date in cache: **{last_date.isoformat()}**. "
        f"Updates happen once per trading day via GitHub Actions (≈{config.expected_update_utc}, after U.S. market close)."
    )

    html_bytes = fig.to_html(full_html=True, include_plotlyjs="inline").encode("utf-8")
    st.download_button(
        label="Download chart as HTML",
        file_name=f"{config.download_prefix}_{pd.Timestamp.utcnow().date().isoformat()}.html",
        data=html_bytes,
        mime="text/html",
    )
