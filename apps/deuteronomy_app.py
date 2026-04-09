from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mm_tracking.core import TradeAppConfig, render_trade_app


render_trade_app(
    TradeAppConfig(
        page_title="MM Deuteronomy Algo vs Competitors — Live Tracking",
        title="MM Deuteronomy Algo vs Competitors — Live Tracking",
        caption="""
- **What we do:** The **MM Deuteronomy Algo** tracks trades and compares their performance to selected competitors.
- **Fair benchmark:** For each Deuteronomy trade, the same **amount** is applied to **SPY** and **QQQ** on the same dates, rolling forward to the next valid price date if the market was closed.
        - **Action handling:** **BUY** is treated as long exposure; **SELL** is treated as short exposure.
        - **ROI metric:** **(Portfolio value − invested amount) ÷ invested amount**.
        - **Breakdown:** Hover over any point on the chart to see ROI, cumulative profit, total value, invested amount, and active trade count for that date.
""",
        portfolio_label="MM Deuteronomy",
        trades_path=Path("data/deuteronomy/deuteronomy_trades.csv"),
        map_path=Path("data/deuteronomy/deuteronomy_ticker_map.csv"),
        parquet_path=Path("caches/deuteronomy_prices_cache.parquet"),
        competitors=["SPY", "QQQ"],
        download_prefix="mm_deuteronomy_vs_competitors",
        expected_update_utc="22:25 UTC",
        missing_cache_message="Missing caches/deuteronomy_prices_cache.parquet. The GitHub Action must write it first.",
    )
)
