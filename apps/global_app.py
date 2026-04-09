from pathlib import Path
from mm_tracking.core import EqualWeightAppConfig, render_equal_weight_app


render_equal_weight_app(
    EqualWeightAppConfig(
        page_title="MM Global Momentum Algorithm vs SPY & QQQ — ROI",
        title="MM Global Momentum Algorithm vs SPY & QQQ — Live Tracking",
        caption="""
- **What we do:** The **MM Global Momentum Algorithm** selects 20 stocks each month and dollar-cost averages **1 unit** into each position on its specified buy date.
- **Fair benchmark:** **SPY** and **QQQ** also invest **1 unit** on those same buy dates, rolling forward to the next date with a valid price if the market was closed.
- **ROI metric:** (Portfolio value − cost) ÷ cost (i.e., percent return on invested units).
        - **Breakdown:** Hover over any point on the chart to see ROI, cumulative profit, total value, and active buy count for any specific date.
        - **Note:** Only **18** stocks were sent for **October 2025**.
""",
        portfolio_label="MM Global",
        buys_path=Path("data/global/global_top_20.csv"),
        map_path=Path("data/global/global_ticker_map.csv"),
        parquet_path=Path("caches/global_prices_cache.parquet"),
        competitors=["SPY", "QQQ"],
        download_prefix="mm_top20_vs_competitors",
        price_caption_prefix="Last price date in cache:",
        expected_update_utc="22:10 UTC",
    )
)
