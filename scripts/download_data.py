#!/usr/bin/env python3
"""Download Binance USD-M futures data (1m klines + funding + symbol meta).

usage:
  python scripts/download_data.py --symbol ZHONGJIUSDT --start 2026-08-14 --end 2026-09-16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import download  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--no-synthetic-fallback", action="store_true")
    args = p.parse_args()

    meta = download(args.symbol, args.start, args.end, Path(args.data_dir),
                    allow_synthetic=not args.no_synthetic_fallback)
    print(f"[OK] {args.symbol}: source={meta['data_source']} "
          f"tick={meta.get('tick_size')} step={meta.get('step_size')} "
          f"-> {args.data_dir}/")


if __name__ == "__main__":
    main()
