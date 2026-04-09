from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mm_tracking.core import EqualWeightAppConfig, render_equal_weight_app


render_equal_weight_app(
    EqualWeightAppConfig(
        page_title="MM Crypto Momentum Algorithm vs Competitors — ROI",
        title="MM Crypto Momentum Algorithm vs Competitors — Live Tracking",
        caption="""
- **What we do:** The **MM Crypto Momentum Algorithm** selects a set of crypto assets each month and dollar-cost averages **1 unit** into each on its specified buy date.
        - **Benchmarks:** **BTC and ETH** also invest **1 unit** on those same buy dates (first valid day with a price), holding thereafter.
        - **ROI metric:** (Portfolio value − cost) ÷ cost.
        - **Breakdown:** Hover a point to see ROI, cumulative profit, total value, and active buy count.
""",
        portfolio_label="MM Crypto",
        buys_path=Path("data/crypto/crypto_top_20.csv"),
        map_path=Path("data/crypto/crypto_map.csv"),
        parquet_path=Path("caches/crypto_prices_cache.parquet"),
        competitors=[("bitcoin", "BTC"), ("ethereum", "ETH")],
        download_prefix="mm_crypto_vs_competitors",
        price_caption_prefix="Last price date in crypto cache:",
        data_kind="crypto",
        missing_cache_message="Missing caches/crypto_prices_cache.parquet. The GitHub Action must write it first.",
    )
)
