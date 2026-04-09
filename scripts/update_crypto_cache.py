from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mm_tracking.cache_builders import build_crypto_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the crypto price cache parquet from repo CSV inputs.")
    parser.add_argument("--dataset", required=True, help="Path to the crypto buy CSV")
    parser.add_argument("--map", required=True, help="Path to the crypto map CSV")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--competitor-id", action="append", default=[], help="CoinGecko competitor id to include")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return build_crypto_cache(
        dataset_path=Path(args.dataset),
        map_path=Path(args.map),
        parquet_path=Path(args.output),
        competitor_ids=set(args.competitor_id),
    )


if __name__ == "__main__":
    raise SystemExit(main())
