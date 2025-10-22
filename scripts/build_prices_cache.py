# scripts/build_prices_cache.py
# Build prices_cache.parquet from CSVs. No plotting, no Streamlit—just data.
from datetime import timedelta
import numpy as np
import pandas as pd
import yfinance as yf

# ---- repo files ----
MM_PATH = "mm_top_20.csv"       # columns: Ticker, Date
MAP_PATH = "ticker_map.csv"     # columns: User Ticker, Resolved Ticker [, Currency]
OUT_PATH = "prices_cache.parquet"

DEFAULT_COMPETITORS = [
    "SPY", "QQQ", "VTI", "RSP", "QQQE",
    "IWM", "DIA", "ACWI", "EFA", "EEM",
    "XLK", "SMH", "XLE", "XLF", "XLV",
    "VNQ", "GLD", "TLT", "BTC-USD",
]

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

def main():
    # Load CSVs
    buys = pd.read_csv(MM_PATH)
    tmap = pd.read_csv(MAP_PATH)

    buys = buys.rename(columns={"Ticker": "user_ticker", "Date": "buy_date"})
    buys["user_ticker"] = buys["user_ticker"].astype(str).str.strip()
    buys["buy_date"] = pd.to_datetime(buys["buy_date"], errors="coerce")
    buys = buys.dropna(subset=["user_ticker", "buy_date"])

    tmap = tmap.rename(columns={"User Ticker": "user_ticker", "Resolved Ticker": "resolved"})
    tmap["user_ticker"] = tmap["user_ticker"].astype(str).str.strip()
    tmap["resolved"] = tmap["resolved"].astype(str).str.strip()
    tmap["yf_ticker"] = tmap["resolved"].apply(exchsym_to_yahoo)

    mapped = buys.merge(tmap[["user_ticker", "yf_ticker"]], on="user_ticker", how="left")
    mapped["yf_ticker"] = np.where(
        mapped["yf_ticker"].isna() | (mapped["yf_ticker"].astype(str).str.len() == 0),
        mapped["user_ticker"].astype(str).str.upper(),
        mapped["yf_ticker"]
    )

    # Date window
    start = (mapped["buy_date"].min() - pd.Timedelta(days=5)).date().isoformat()
    end = (pd.Timestamp.utcnow().normalize() + pd.Timedelta(days=1)).date().isoformat()  # Yahoo end is exclusive

    # Symbols: portfolio + competitors
    symbols = sorted(set(mapped["yf_ticker"].dropna().unique().tolist() + DEFAULT_COMPETITORS))

    # Use curl_cffi session if available (yfinance[fast])
    try:
        from curl_cffi import requests as cf_requests
        sess = cf_requests.Session(impersonate="chrome")
    except Exception:
        sess = None

    # Fetch chunked (reliable for CI)
    CHUNK = 10
    cols = {}
    for i in range(0, len(symbols), CHUNK):
        chunk = symbols[i:i+CHUNK]
        try:
            df = yf.download(
                tickers=chunk, start=start, end=end,
                interval="1d", auto_adjust=True,
                group_by="ticker", progress=False, threads=False,
                session=sess
            )
        except Exception:
            df = None

        if df is None or df.empty:
            continue

        if isinstance(df.columns, pd.MultiIndex):
            # Multi-ticker DataFrame: (symbol, field)
            root_syms = set(df.columns.get_level_values(0))
            for sym in chunk:
                if sym in root_syms and "Close" in df[sym].columns:
                    s = df[sym]["Close"].dropna()
                    if not s.empty:
                        cols[sym] = s
        else:
            # Single-ticker; map if possible
            if len(chunk) == 1 and "Close" in df.columns:
                cols[chunk[0]] = df["Close"].dropna()

    if not cols:
        raise SystemExit("No data fetched. Check mapping/symbols or network.")

    out = pd.DataFrame(cols)
    out.index = pd.to_datetime(out.index)
    out = out[~out.index.duplicated(keep="first")].sort_index()

    # Write parquet
    out.to_parquet(OUT_PATH)
    print(f"Wrote {OUT_PATH} with {out.shape[0]} rows and {out.shape[1]} columns.")
    print(f"Last date in cache: {out.index.max().date().isoformat()}")

if __name__ == "__main__":
    main()
