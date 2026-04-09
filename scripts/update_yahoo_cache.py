from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mm_tracking.cache_builders import build_yahoo_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an equity price cache parquet from repo CSV inputs.")
    parser.add_argument("--dataset", required=True, help="Path to the buy/trade CSV")
    parser.add_argument("--map", required=True, help="Path to the ticker map CSV")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--mode", choices=["buys", "trades"], required=True, help="Input dataset mode")
    parser.add_argument("--competitor", action="append", default=[], help="Competitor ticker to include")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return build_yahoo_cache(
        dataset_path=Path(args.dataset),
        map_path=Path(args.map),
        parquet_path=Path(args.output),
        mode=args.mode,
        competitors=args.competitor,
    )


if __name__ == "__main__":
    raise SystemExit(main())
