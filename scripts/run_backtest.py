#!/usr/bin/env python3
"""Run the closed-market grid experiment for one symbol.

usage:
  python scripts/run_backtest.py --symbol ZHONGJIUSDT --config config/default.yaml

Outputs under outputs/<symbol>/: windows.parquet, parameter_summary.csv,
best_dev_parameters.json, evaluation_summary.json, robustness_summary.csv,
equity_curve.png, parameter_surface.png, pnl_vs_volatility.png,
pnl_vs_window_return.png, report.md
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.backtest import run_session  # noqa: E402
from src.calendar import closed_market_windows, spec_from_config  # noqa: E402
from src.data import download, load  # noqa: E402
from src.metrics import dev_eval_split, parameter_surface, summarize_windows  # noqa: E402
from src.plots import equity_curve, parameter_surface as plot_surface, scatter  # noqa: E402
from src.report import render_report  # noqa: E402


def run_params(klines, funding, windows, cfg, meta, spacing, range_pct,
               levels, max_inv, mode, slip_bps, drop_prob, maker_fee, label):
    recs, curves = [], []
    for w in windows.itertuples(index=False):
        res = run_session(
            klines, funding, w.window_start, w.window_end,
            symbol=meta["symbol"], initial_capital=cfg["account"]["initial_capital"],
            spacing=spacing, range_pct=range_pct, levels_each_side=levels,
            max_inventory_pct=max_inv, maker_fee=maker_fee,
            taker_fee=cfg["fees"]["taker_fee"], exit_slippage_bps=slip_bps,
            tick_size=meta["tick_size"], step_size=meta["step_size"],
            mode=mode, drop_prob=drop_prob, seed=cfg["execution"]["seed"],
            label=label, window_type=w.window_type, complete=bool(w.complete))
        if res is not None:
            recs.append(res.record)
            curves.append(res.equity_curve)
    return pd.DataFrame(recs), curves


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--start", default="2026-08-14")
    p.add_argument("--end", default="2026-09-16")
    p.add_argument("--quick", action="store_true", help="single param combo only")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_dir = Path("data")
    out_dir = Path("outputs") / args.symbol
    out_dir.mkdir(parents=True, exist_ok=True)

    kf = data_dir / f"{args.symbol}_1m.parquet"
    if not kf.exists():
        print(f"[data] {args.symbol} not cached, downloading...")
        download(args.symbol, args.start, args.end, data_dir)
    klines, funding, meta = load(args.symbol, data_dir)
    print(f"[data] {len(klines)} 1m bars, {len(funding)} funding events, "
          f"source={meta['data_source']}")

    spec = spec_from_config(cfg["symbols"][args.symbol])
    start, end = klines.open_time.min(), klines.open_time.max()
    windows = closed_market_windows(spec, start, end)
    print(f"[cal] {len(windows)} closed windows: "
          f"{windows.window_type.value_counts().to_dict()}")

    # keep only windows with enough bar coverage
    n_bars = []
    for w in windows.itertuples(index=False):
        n = ((klines.open_time >= w.window_start) & (klines.open_time < w.window_end)).sum()
        n_bars.append(int(n))
    windows["n_bars"] = n_bars
    windows = windows[windows.n_bars >= cfg["experiment"]["min_bars_per_window"]] \
        .reset_index(drop=True)

    g = cfg["grid"]
    ex = cfg["execution"]
    base = dict(mode=ex["mode"], slip_bps=ex["exit_slippage_bps"],
                drop_prob=ex["fill_drop_prob"], maker_fee=cfg["fees"]["maker_fee"])

    combos = list(itertools.product(g["grid_spacing_pct"], g["range_pct"],
                                    g["levels_each_side"], g["max_inventory_pct"]))
    if args.quick:
        combos = combos[:1]

    # ---- parameter sweep on strategy B (all closed windows) ----
    all_recs, ref_curves = [], []
    for spacing, rng_pct, lvls, mx in combos:
        recs, curves = run_params(klines, funding, windows, cfg, meta,
                                  spacing, rng_pct, lvls, mx, label="B_closed", **base)
        all_recs.append(recs)
        if (spacing, rng_pct, lvls, mx) == combos[0]:
            ref_curves = curves
    results = pd.concat(all_recs, ignore_index=True)
    results.to_parquet(out_dir / "windows.parquet", index=False)
    print(f"[sweep] {len(combos)} param combos x {len(windows)} windows "
          f"= {len(results)} sessions")

    # ---- dev / eval split (chronological) ----
    dev_idx, eval_idx = dev_eval_split(windows, cfg["experiment"]["dev_fraction"])
    surf_dev = parameter_surface(results, dev_idx)
    surf_eval = parameter_surface(results, eval_idx)
    surf_dev["split"], surf_eval["split"] = "dev", "eval"
    pd.concat([surf_dev, surf_eval]).to_csv(out_dir / "parameter_summary.csv", index=False)

    best_row = surf_dev.sort_values("mean_return_pct", ascending=False).iloc[0]
    best = {k: (float(best_row[k]) if k in ("spacing", "range_pct", "max_inventory_pct")
                else int(best_row[k]))
            for k in ("spacing", "range_pct", "levels", "max_inventory_pct")}
    (out_dir / "best_dev_parameters.json").write_text(json.dumps(best, indent=2))
    print(f"[dev] best params: {best}")

    dev_res = results[results.window_start.isin(dev_idx) &
                      (results.spacing == best["spacing"]) &
                      (results.range_pct == best["range_pct"]) &
                      (results.levels == best["levels"]) &
                      (results.max_inventory_pct == best["max_inventory_pct"])]
    eval_res = results[results.window_start.isin(eval_idx) &
                       (results.spacing == best["spacing"]) &
                       (results.range_pct == best["range_pct"]) &
                       (results.levels == best["levels"]) &
                       (results.max_inventory_pct == best["max_inventory_pct"])]
    dev_summary = summarize_windows(dev_res, cfg["account"]["initial_capital"],
                                    cfg["experiment"]["bootstrap_iters"])
    eval_summary = summarize_windows(eval_res, cfg["account"]["initial_capital"],
                                     cfg["experiment"]["bootstrap_iters"])
    eval_summary["best_dev_parameters"] = best
    (out_dir / "evaluation_summary.json").write_text(
        json.dumps({"development": dev_summary, "evaluation": eval_summary},
                   indent=2, default=str))

    # ---- strategy benchmarks at best params ----
    bp = best
    strat_windows = {
        "B_closed": windows,
        "C_overnight": windows[windows.window_type == "weekday_overnight"],
        "D_weekend": windows[windows.window_type == "weekend"],
    }
    strategy_summaries = {}
    for name, wdf in strat_windows.items():
        recs, _ = run_params(klines, funding, wdf, cfg, meta, bp["spacing"],
                             bp["range_pct"], bp["levels"], bp["max_inventory_pct"],
                             label=name, **base)
        strategy_summaries[name] = summarize_windows(
            recs, cfg["account"]["initial_capital"], 500)
    # A: 24/7 continuous grid over the whole data range
    a = run_session(klines, funding, start, end, symbol=meta["symbol"],
                    initial_capital=cfg["account"]["initial_capital"],
                    spacing=bp["spacing"], range_pct=bp["range_pct"],
                    levels_each_side=bp["levels"], max_inventory_pct=bp["max_inventory_pct"],
                    maker_fee=cfg["fees"]["maker_fee"], taker_fee=cfg["fees"]["taker_fee"],
                    exit_slippage_bps=ex["exit_slippage_bps"],
                    tick_size=meta["tick_size"], step_size=meta["step_size"],
                    mode=ex["mode"], drop_prob=ex["fill_drop_prob"],
                    seed=ex["seed"], label="A_24x7", window_type="continuous")
    strategy_summaries["A_24x7"] = {
        "n_windows": 1, "total_net_pnl": a.record["net_pnl"],
        "mean_return_pct": a.record["return_pct"],
        "win_rate": float(a.record["net_pnl"] > 0)} if a else {"n_windows": 0}

    # ---- robustness on eval windows at best params ----
    robust_rows = []
    eval_windows = windows[windows.window_start.isin(eval_idx)]
    tests = [("exec_model", "optimistic", dict(mode="optimistic")),
             ("exec_model", "conservative", dict(mode="conservative"))]
    for bps in (0, 2, 5, 10):
        tests.append(("slippage", f"{bps}bps", dict(slip_bps=bps)))
    for mf in (0.0, 0.0001, 0.0002):
        tests.append(("maker_fee", f"{mf*1e4:.0f}bp", dict(maker_fee=mf)))
    for dp in (0.0, 0.10, 0.25):
        tests.append(("fill_drop", f"{dp:.0%}", dict(drop_prob=dp)))
    for tag, setting, over in tests:
        kw = {**base, **over}
        recs, _ = run_params(klines, funding, eval_windows, cfg, meta, bp["spacing"],
                             bp["range_pct"], bp["levels"], bp["max_inventory_pct"],
                             label=f"robust_{tag}_{setting}", **kw)
        s = summarize_windows(recs, cfg["account"]["initial_capital"], 500)
        robust_rows.append({"tag": tag, "setting": setting,
                            "exit_slippage_bps": kw["slip_bps"],
                            "n_windows": s.get("n_windows", 0),
                            "mean_return_pct": s.get("mean_return_pct", np.nan),
                            "total_net_pnl": s.get("total_net_pnl", np.nan),
                            "win_rate": s.get("win_rate", np.nan)})
    robustness = pd.DataFrame(robust_rows)
    robustness.to_csv(out_dir / "robustness_summary.csv", index=False)

    # ---- plots + report ----
    if ref_curves:
        eq = pd.concat(ref_curves).sort_index()
        equity_curve(eq, out_dir / "equity_curve.png",
                     f"{args.symbol} grid equity (combo0, all windows)")
    plot_surface(surf_dev, out_dir / "parameter_surface.png",
                 f"{args.symbol} dev parameter surface")
    main_res = results[(results.spacing == best["spacing"]) &
                       (results.range_pct == best["range_pct"]) &
                       (results.levels == best["levels"]) &
                       (results.max_inventory_pct == best["max_inventory_pct"])]
    scatter(main_res.realized_vol, main_res.net_pnl, out_dir / "pnl_vs_volatility.png",
            "realized vol (std of 1m ret)", "net pnl", f"{args.symbol} pnl vs volatility")
    scatter(main_res.window_return.abs(), main_res.net_pnl,
            out_dir / "pnl_vs_window_return.png", "|window return|", "net pnl",
            f"{args.symbol} pnl vs |window return|")

    report = render_report(args.symbol, meta, spec.name, main_res, dev_summary,
                           eval_summary, best, robustness, strategy_summaries,
                           strat_windows)
    (out_dir / "report.md").write_text(report)
    print(f"[done] outputs -> {out_dir}/")
    print(f"[result] eval net={eval_summary.get('total_net_pnl', 0):+.2f} USDT "
          f"over {eval_summary.get('n_windows', 0)} windows")


if __name__ == "__main__":
    main()
