#!/usr/bin/env python3
"""Execution stress test for the frozen ZHONGJI lunch ATR candidate.

This script does not re-select a strategy.  It takes the candidate selected from
the pre-2026-09-01 development period by run_zhongji_atr_lab.py and applies
predefined execution stresses to historical-audit lunch windows.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_grid_lab import load_cached, windows_for
from scripts.run_zhongji_atr_lab import CUTOFF, prepare_atr, summary
from src.atr_dynamic import candidate_registry, run_array


FROZEN_LUNCH = "atr_refresh_aewm_m2_n4_r5_c3"
SCENARIOS = {
    "base": {},
    "fees_slippage_2x": dict(maker=.0004, taker=.0010, slip_bps=10),
    "fill_drop25": dict(fill_drop=.25),
    "volume_cap_0.1pct": dict(participation=.001),
    "buy_first": dict(path_mode=1),
    "sell_first": dict(path_mode=2),
    "combined": dict(maker=.0004, taker=.0010, slip_bps=10,
                     fill_drop=.25, participation=.001),
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config/default.yaml").read_text())
    outdir = root / "outputs/zhongji_atr_dynamic"
    outdir.mkdir(parents=True, exist_ok=True)

    k, funding, meta, _ = load_cached("ZHONGJIUSDT", root / "data")
    matrix = prepare_atr(k, funding)
    windows = windows_for(k, cfg, "ZHONGJIUSDT")
    windows = windows[
        windows.usable
        & (windows.start_idx >= 121)
        & (windows.window_start >= CUTOFF)
        & (windows.window_type == "lunch_break")
    ].reset_index(drop=True)

    candidates = {c.name: c for c in candidate_registry()}
    candidate = candidates[FROZEN_LUNCH]
    base_kw = dict(tick=meta["tick_size"], step=meta["step_size"])

    rows = []
    for scenario, overrides in SCENARIOS.items():
        per_window = []
        for w in windows.itertuples(index=False):
            rec, _ = run_array(
                matrix[w.start_idx:w.end_idx], candidate,
                **base_kw, **overrides,
            )
            rec.update(window_start=w.window_start, window_type=w.window_type)
            per_window.append(rec)
        s = summary(pd.DataFrame(per_window))
        rows.append({"candidate": FROZEN_LUNCH, "scenario": scenario, **s})

    result = pd.DataFrame(rows)
    result.to_csv(outdir / "lunch_candidate_stress.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
