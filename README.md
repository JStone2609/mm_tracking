# mm_tracking

This repo contains four Streamlit-facing trackers built from CSV inputs and cached price parquet files:

- `apps/global_app.py`: MM Global Momentum
- `apps/crypto_app.py`: MM Crypto Momentum
- `apps/exodus_app.py`: MM Exodus
- `apps/deuteronomy_app.py`: MM Deuteronomy

## Structure

- `apps/`: Streamlit entrypoints only
- `data/`: monthly editable CSV inputs grouped by strategy
- `caches/`: generated parquet price caches grouped in one place
- `mm_tracking/core.py`: shared app logic for CSV loading, mapping, price-cache reading, ROI calculations, and chart rendering
- `mm_tracking/cache_builders.py`: shared cache-building logic used by GitHub Actions
- `scripts/update_yahoo_cache.py`: CLI wrapper for the equity workflows
- `scripts/update_crypto_cache.py`: CLI wrapper for the crypto workflow
- `*.csv`: editable monthly inputs and ticker maps
- `.github/workflows/*.yml`: thin workflow wrappers that call the shared scripts

## Monthly updates

For normal monthly maintenance, you usually only need to edit:

- `data/global/global_top_20.csv` and `data/global/global_ticker_map.csv`
- `data/crypto/crypto_top_20.csv` and `data/crypto/crypto_map.csv`
- `data/exodus/exodus_trades.csv` and `data/exodus/exodus_ticker_map.csv`
- `data/deuteronomy/deuteronomy_trades.csv` and `data/deuteronomy/deuteronomy_ticker_map.csv`

Each related workflow rebuilds its parquet cache on push.

## Notes

- Streamlit entrypoint filenames changed, so Streamlit should now point to the files under `apps/`.
- Shared logic now lives in one place, so fixes to mapping, charting, or cache generation can be made once and reused across apps.
- Old `plus500` tracker files and generated `__pycache__` artifacts were removed to keep the repo focused on the four maintained apps.
