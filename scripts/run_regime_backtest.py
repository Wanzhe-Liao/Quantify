#!/usr/bin/env python3
"""Causal low-volatility regime experiment for closed-market grids.

This script freezes the grid parameters selected by the existing development
sweep, observes the first N minutes of each *complete* closed-market window,
and then compares a delayed-all baseline against pre-specified low-volatility
regime gates using only development-fitted thresholds.

No gate is selected on evaluation performance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.backtest import run_session  # noqa: E402
from src.calendar import closed_market_windows, spec_from_config  # noqa: E402
from src.data import download, load  # noqa: E402
from src.metrics import dev_eval_split, summarize_windows  # noqa: E402
from src.regime import build_observation_table, fit_gate_thresholds, gate_mask  # noqa: E402


def _run_feature_rows(features, klines, funding, cfg, meta, best, gate_name):
    ex = cfg["execution"]
    rows = []
    for row in features.itertuples(index=False):
        res = run_session(
            klines, funding, row.trade_start, row.window_end,
            symbol=meta["symbol"],
            initial_capital=cfg["account"]["initial_capital"],
            spacing=best["spacing"], range_pct=best["range_pct"],
            levels_each_side=best["levels"],
            max_inventory_pct=best["max_inventory_pct"],
            maker_fee=cfg["fees"]["maker_fee"],
            taker_fee=cfg["fees"]["taker_fee"],
            exit_slippage_bps=ex["exit_slippage_bps"],
            tick_size=meta["tick_size"], step_size=meta["step_size"],
            mode=ex["mode"], drop_prob=ex["fill_drop_prob"], seed=ex["seed"],
            label=gate_name, window_type=row.window_type, complete=True,
        )
        if res is None:
            continue
        rec = dict(res.record)
        rec["original_window_start"] = row.window_start
        rec["trade_start"] = row.trade_start
        rec["observation_minutes"] = row.observation_minutes
        rec["observation_vol"] = row.observation_vol
        rec["observation_return"] = row.observation_return
        rec["abs_observation_return"] = row.abs_observation_return
        rec["observation_range_pct"] = row.observation_range_pct
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--start", default="2026-08-14")
    p.add_argument("--end", default="2026-09-16")
    p.add_argument("--observation-minutes", type=int, default=30)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_dir = Path("data")
    out_dir = Path("outputs") / args.symbol / "regime_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    kf = data_dir / f"{args.symbol}_1m.parquet"
    if not kf.exists():
        download(args.symbol, args.start, args.end, data_dir)
    klines, funding, meta = load(args.symbol, data_dir)

    spec = spec_from_config(cfg["symbols"][args.symbol])
    start, end = klines.open_time.min(), klines.open_time.max()
    windows = closed_market_windows(spec, start, end)
    windows = windows[windows.complete].reset_index(drop=True)

    # Preserve the original chronological split at the closed-window level.
    dev_idx, eval_idx = dev_eval_split(windows, cfg["experiment"]["dev_fraction"])

    features = build_observation_table(
        klines, windows,
        observation_minutes=args.observation_minutes,
        min_observation_bars=max(10, args.observation_minutes // 2),
        min_trade_minutes=10,
    )
    features.to_parquet(out_dir / "observation_features.parquet", index=False)

    thresholds = fit_gate_thresholds(features, dev_idx)
    threshold_dict = {
        "observation_minutes": args.observation_minutes,
        "vol_q25": thresholds.vol_q25,
        "vol_q50": thresholds.vol_q50,
        "vol_q75": thresholds.vol_q75,
        "drift_q75": thresholds.drift_q75,
    }
    (out_dir / "gate_thresholds.json").write_text(json.dumps(threshold_dict, indent=2))

    best_path = Path("outputs") / args.symbol / "best_dev_parameters.json"
    if not best_path.exists():
        raise FileNotFoundError(
            f"{best_path} missing; run the original development sweep first"
        )
    best = json.loads(best_path.read_text())

    gate_names = [
        "delayed_all",
        "lowvol_q50",
        "lowvol_q50_driftguard",
        "lowvol_q25",
        "lowvol_q75",
    ]
    all_results = []
    summary_rows = []
    summary_json = {
        "symbol": args.symbol,
        "reference_market": spec.name,
        "frozen_grid_parameters": best,
        "thresholds_fit_on_development_only": threshold_dict,
        "gates": {},
    }

    for gate in gate_names:
        mask = gate_mask(features, thresholds, gate)
        selected = features[mask].copy()
        results = _run_feature_rows(selected, klines, funding, cfg, meta, best, gate)
        if not results.empty:
            all_results.append(results)

        dev = results[results.original_window_start.isin(dev_idx)] if not results.empty else results
        ev = results[results.original_window_start.isin(eval_idx)] if not results.empty else results
        dev_summary = summarize_windows(dev, cfg["account"]["initial_capital"],
                                        cfg["experiment"]["bootstrap_iters"])
        eval_summary = summarize_windows(ev, cfg["account"]["initial_capital"],
                                         cfg["experiment"]["bootstrap_iters"])
        summary_json["gates"][gate] = {
            "development": dev_summary,
            "evaluation": eval_summary,
        }
        for split, s in (("development", dev_summary), ("evaluation", eval_summary)):
            summary_rows.append({
                "gate": gate,
                "split": split,
                "n_windows": s.get("n_windows", 0),
                "total_net_pnl": s.get("total_net_pnl"),
                "mean_return_pct": s.get("mean_return_pct"),
                "win_rate": s.get("win_rate"),
                "profit_factor": s.get("profit_factor"),
                "max_drawdown_pct": s.get("max_drawdown_pct"),
                "cvar20_return_pct": s.get("cvar20_return_pct"),
                "ci95_lo": (s.get("bootstrap_ci95_mean_return") or [None, None])[0],
                "ci95_hi": (s.get("bootstrap_ci95_mean_return") or [None, None])[1],
            })

    if all_results:
        pd.concat(all_results, ignore_index=True).to_parquet(
            out_dir / "gated_windows.parquet", index=False
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "gate_summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary_json, indent=2, default=str))

    # Compact Markdown artifact focused on the primary test.
    lines = [
        f"# Regime-gated closed-market grid: {args.symbol}",
        "",
        f"Reference market: **{spec.name}**",
        f"Observation delay: **{args.observation_minutes} min**",
        f"Frozen grid params: `{best}`",
        "",
        "Thresholds are fit on development windows only. Evaluation windows are never used to choose a gate.",
        "Only complete closed-market windows are included; the prior unclosed tail is excluded.",
        "",
        "| gate | split | n | net USDT | mean/win | win rate | PF | max DD | CVaR20 | CI95 mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        def f(v, nd=4):
            return "-" if v is None or pd.isna(v) else f"{v:.{nd}f}"
        lines.append(
            f"| {row['gate']} | {row['split']} | {row['n_windows']} | "
            f"{f(row['total_net_pnl'], 2)} | {f(row['mean_return_pct'], 5)} | "
            f"{f(row['win_rate'], 3)} | {f(row['profit_factor'], 2)} | "
            f"{f(row['max_drawdown_pct'], 5)} | {f(row['cvar20_return_pct'], 5)} | "
            f"[{f(row['ci95_lo'], 5)}, {f(row['ci95_hi'], 5)}] |"
        )
    lines.extend([
        "",
        "## Interpretation rule",
        "",
        "Primary hypothesis: `lowvol_q50` should improve tail risk and mean net return versus `delayed_all` on evaluation windows.",
        "`lowvol_q25` and `lowvol_q75` are sensitivity analyses, not candidates for post-hoc selection.",
        "Because the sample is short, a positive point estimate with a confidence interval crossing zero remains inconclusive.",
    ])
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"[done] outputs -> {out_dir}")


if __name__ == "__main__":
    main()
