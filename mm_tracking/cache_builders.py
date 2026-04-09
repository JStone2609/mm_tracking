from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from mm_tracking.core import load_crypto_map, load_equal_weight_buys, load_equity_map, load_trade_book


def collect_yahoo_symbols(dataset_path: Path, map_path: Path, mode: str, competitors: list[str]) -> tuple[list[str], str, str]:
    if mode == "trades":
        positions = load_trade_book(dataset_path).rename(columns={"trade_date": "event_date"})
    else:
        positions = load_equal_weight_buys(dataset_path).rename(columns={"buy_date": "event_date"})

    tmap = load_equity_map(map_path)
    mapped = positions.merge(tmap, on="user_ticker", how="left")
    mapped["yf_ticker"] = np.where(
        mapped["yf_ticker"].isna() | (mapped["yf_ticker"].astype(str).str.len() == 0),
        mapped["user_ticker"].astype(str).str.upper(),
        mapped["yf_ticker"],
    )
    symbols = sorted(set(mapped["yf_ticker"].dropna().astype(str)) | set(competitors))
    start = (mapped["event_date"].min() - pd.Timedelta(days=200)).date().isoformat()
    end = (pd.Timestamp.utcnow().normalize() + pd.Timedelta(days=1)).date().isoformat()
    return symbols, start, end


def fetch_yahoo_prices(symbols: list[str], start: str, end: str) -> dict[str, pd.Series]:
    import yfinance as yf

    print("Fetching", len(symbols), "symbols")
    print("Window:", start, "->", end)

    ok: dict[str, pd.Series] = {}
    chunk_size = 8

    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i : i + chunk_size]
        try:
            df = yf.download(
                tickers=chunk,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                progress=False,
                threads=False,
            )
        except Exception as exc:
            print("Batch failed for", chunk, "reason:", exc)
            df = None

        if df is None or df.empty:
            continue

        if isinstance(df.columns, pd.MultiIndex):
            roots = set(df.columns.get_level_values(0))
            for symbol in chunk:
                if symbol in roots and "Close" in df[symbol].columns:
                    series = df[symbol]["Close"].dropna()
                    if not series.empty:
                        ok[symbol] = series
        elif len(chunk) == 1 and "Close" in df.columns:
            series = df["Close"].dropna()
            if not series.empty:
                ok[chunk[0]] = series

    missing = [symbol for symbol in symbols if symbol not in ok]
    for symbol in missing:
        success = False
        for _ in range(3):
            try:
                history = yf.Ticker(symbol).history(start=start, end=end, interval="1d", auto_adjust=True)
                if history is not None and not history.empty and "Close" in history.columns:
                    series = history["Close"].dropna()
                    if not series.empty:
                        ok[symbol] = series
                        success = True
                        break
            except Exception:
                continue
        if not success:
            print("Fallback failed for", symbol)

    return ok


def build_yahoo_cache(dataset_path: Path, map_path: Path, parquet_path: Path, mode: str, competitors: list[str]) -> int:
    symbols, start, end = collect_yahoo_symbols(dataset_path, map_path, mode=mode, competitors=competitors)
    ok = fetch_yahoo_prices(symbols, start, end)

    if not ok:
        print(f"WARN: No data fetched - leaving existing {parquet_path.name} as-is (if any).")
        return 0

    out = pd.DataFrame(ok)
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()

    if not out.empty:
        full_date_range = pd.date_range(start=out.index.min(), end=out.index.max(), freq="D")
        out = out.reindex(full_date_range)
        out = out.ffill()
        out = out.dropna(how="all")

    out.to_parquet(parquet_path)
    print(f"Wrote {parquet_path.name} with columns:", list(out.columns))
    return 0


def fetch_crypto_market_chart_range(cg_id: str, currency: str, start_unix: int, end_unix: int) -> pd.DataFrame | None:
    import requests

    url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart/range"
    params = {"vs_currency": currency.lower(), "from": start_unix, "to": end_unix}
    delay = 0.5

    for _ in range(6):
        try:
            response = requests.get(url, params=params, timeout=40)
            if response.status_code == 200:
                data = response.json()
                prices = data.get("prices") or []
                if not prices:
                    return None
                df = pd.DataFrame(prices, columns=["ts", "price"])
                df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.normalize()
                return df.groupby("ts", as_index=True)["price"].last().rename(cg_id).to_frame()
            if response.status_code == 429:
                time.sleep(delay)
                delay *= 2
                continue
            time.sleep(1.0)
        except requests.RequestException:
            time.sleep(1.0)

    print("WARN", cg_id, "failed after retries")
    return None


def build_crypto_cache(
    dataset_path: Path,
    map_path: Path,
    parquet_path: Path,
    competitor_ids: set[str],
) -> int:
    buys = load_equal_weight_buys(dataset_path)
    cmap = load_crypto_map(map_path)
    mapped = buys.merge(cmap, on="user_ticker", how="left")

    ids = sorted(set(mapped["cg_id"].dropna().astype(str)) | competitor_ids)
    if not ids:
        print("No CoinGecko ids found from mapping; aborting.")
        return 0

    currency = "USD"
    try:
        raw_map = pd.read_csv(map_path)
        if "Currency" in raw_map.columns and not raw_map.empty:
            currency = str(raw_map["Currency"].mode().iat[0]).strip() or "USD"
    except Exception:
        pass

    start_dt = (buys["buy_date"].min() - pd.Timedelta(days=200)).to_pydatetime()
    end_dt = pd.Timestamp.utcnow().to_pydatetime() + pd.Timedelta(days=1)
    start_unix = int(start_dt.timestamp())
    end_unix = int(end_dt.timestamp())

    frames = []
    for cg_id in ids:
        df = fetch_crypto_market_chart_range(cg_id, currency, start_unix, end_unix)
        if df is not None and not df.empty:
            frames.append(df)
        time.sleep(1.0)

    if not frames:
        print(f"No crypto data fetched; leaving any existing {parquet_path.name} untouched.")
        return 0

    out = pd.concat(frames, axis=1).sort_index()
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"

    try:
        prev = pd.read_parquet(parquet_path)
        prev.index = pd.to_datetime(prev.index)
        out = prev.join(out, how="outer")
    except Exception:
        pass

    out = out.sort_index()
    out.to_parquet(parquet_path)
    print(f"Wrote {parquet_path.name} with columns:", list(out.columns))
    return 0
