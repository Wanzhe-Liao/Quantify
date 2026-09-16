#!/usr/bin/env python3
"""Causal low-volatility regime experiment for closed-market grids.

The existing development sweep selects the grid geometry.  This experiment
freezes those parameters, observes the first N minutes of each *complete*
closed-market window, and asks whether a causal low-volatility gate improves
the strategy out of sample.

Important evaluation rule: a skipped window is a zero-return opportunity, not a
missing observation.  Gate-level headline metrics therefore include every
eligible closed-market opportunity; traded-window metrics are reported only as
secondary diagnostics.  Gate thresholds are fit on development windows only.
"""
from __future__ import annotations

import argparse
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
from src.metrics import dev_eval_split, summarize_windows  # noqa: E402
from src.regime import build_observation_table, fit_gate_thresholds, gate_mask  # noqa: E402


ZERO_WHEN_SKIPPED = (
    "net_pnl", "return_pct", "funding_pnl", "gross_grid_pnl",
    "inventory_liquidation_pnl", "maker_fees", "taker_fees",
    "slippage_cost", "trading_pnl", "turnover", "num_fills", "num_cycles",
    "max_long_inventory", "max_short_inventory",
)


def _run_feature_rows(features, klines, funding, cfg, meta, best):
    """Run the frozen grid once for every eligible delayed-start opportunity."""
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
            label="delayed_all", window_type=row.window_type, complete=True,
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


def _gate_views(base_results: pd.DataFrame, selected_starts: set,
                gate: str, initial_capital: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (active_trades, all_opportunities_with_skips_as_zero)."""
    selected = base_results.original_window_start.isin(selected_starts)
    active = base_results[selected].copy()
    active["label"] = gate
    active["gate_active"] = True

    opportunity = base_results.copy()
    opportunity["label"] = gate
    opportunity["gate_active"] = selected.to_numpy()
    skipped = ~selected.to_numpy()
    for col in ZERO_WHEN_SKIPPED:
        if col in opportunity.columns:
            opportunity.loc[skipped, col] = 0.0
    opportunity.loc[skipped, "final_equity"] = initial_capital
    opportunity.loc[skipped, "max_drawdown"] = 0.0
    return active, opportunity


def _paired_delta(gated: pd.DataFrame, baseline: pd.DataFrame,
                  starts: pd.Index, bootstrap_iters: int, seed: int = 7) -> dict:
    """Paired gate-minus-baseline return delta over the same opportunities."""
    g = gated[gated.original_window_start.isin(starts)].sort_values("original_window_start")
    b = baseline[baseline.original_window_start.isin(starts)].sort_values("original_window_start")
    merged = g[["original_window_start", "return_pct", "net_pnl"]].merge(
        b[["original_window_start", "return_pct", "net_pnl"]],
        on="original_window_start", suffixes=("_gate", "_base"), validate="one_to_one",
    )
    if merged.empty:
        return {"n_windows": 0}
    delta = (merged.return_pct_gate - merged.return_pct_base).to_numpy(dtype=float)
    net_delta = (merged.net_pnl_gate - merged.net_pnl_base).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    boots = rng.choice(delta, size=(bootstrap_iters, len(delta)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n_windows": int(len(delta)),
        "mean_return_delta": float(delta.mean()),
        "total_net_pnl_delta": float(net_delta.sum()),
        "bootstrap_ci95_mean_delta": [float(lo), float(hi)],
    }


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
    initial_capital = cfg["account"]["initial_capital"]
    bootstrap_iters = cfg["experiment"]["bootstrap_iters"]

    kf = data_dir / f"{args.symbol}_1m.parquet"
    if not kf.exists():
        download(args.symbol, args.start, args.end, data_dir)
    klines, funding, meta = load(args.symbol, data_dir)

    spec = spec_from_config(cfg["symbols"][args.symbol])
    start, end = klines.open_time.min(), klines.open_time.max()
    windows = closed_market_windows(spec, start, end)
    # An unclosed tail has no observed next-market-open liquidation endpoint.
    windows = windows[windows.complete].reset_index(drop=True)

    # Keep the original chronological split before any regime filtering.
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
        raise FileNotFoundError(f"{best_path} missing; run the original development sweep first")
    best = json.loads(best_path.read_text())

    base_results = _run_feature_rows(features, klines, funding, cfg, meta, best)
    if base_results.empty:
        raise RuntimeError("no eligible delayed-start sessions were produced")

    gate_names = [
        "delayed_all",
        "lowvol_q50",                 # primary hypothesis
        "lowvol_q50_driftguard",      # pre-specified secondary
        "lowvol_q25",                 # sensitivity only
        "lowvol_q75",                 # sensitivity only
    ]
    all_results = []
    opportunity_results = []
    summary_rows = []
    summary_json = {
        "symbol": args.symbol,
        "reference_market": spec.name,
        "frozen_grid_parameters": best,
        "thresholds_fit_on_development_only": threshold_dict,
        "evaluation_rule": "skipped windows count as zero-return opportunities",
        "gates": {},
    }

    # The matched delayed baseline uses exactly the same 30-min observation delay.
    _, baseline_opp = _gate_views(
        base_results, set(features.window_start), "delayed_all", initial_capital
    )

    for gate in gate_names:
        mask = gate_mask(features, thresholds, gate)
        selected_starts = set(features.loc[mask, "window_start"])
        active, opportunity = _gate_views(
            base_results, selected_starts, gate, initial_capital
        )
        if not active.empty:
            all_results.append(active)
        opportunity_results.append(opportunity)

        gate_block = {}
        for split, split_idx in (("development", dev_idx), ("evaluation", eval_idx)):
            active_split = active[active.original_window_start.isin(split_idx)]
            opp_split = opportunity[opportunity.original_window_start.isin(split_idx)]
            active_summary = summarize_windows(
                active_split, initial_capital, bootstrap_iters
            ) if len(active_split) else {"n_windows": 0}
            opp_summary = summarize_windows(opp_split, initial_capital, bootstrap_iters)
            delta = _paired_delta(
                opportunity, baseline_opp, split_idx, bootstrap_iters
            )
            coverage = len(active_split) / len(opp_split) if len(opp_split) else 0.0
            gate_block[split] = {
                "coverage": coverage,
                "active_trades": active_summary,
                "all_opportunities": opp_summary,
                "paired_vs_delayed_all": delta,
            }
            summary_rows.append({
                "gate": gate,
                "split": split,
                "eligible_windows": len(opp_split),
                "traded_windows": len(active_split),
                "coverage": coverage,
                "total_net_pnl": opp_summary.get("total_net_pnl"),
                "mean_return_per_opportunity": opp_summary.get("mean_return_pct"),
                "active_mean_return": active_summary.get("mean_return_pct"),
                "active_win_rate": active_summary.get("win_rate"),
                "max_drawdown_pct": opp_summary.get("max_drawdown_pct"),
                "cvar20_return_pct": opp_summary.get("cvar20_return_pct"),
                "delta_mean_vs_delayed_all": delta.get("mean_return_delta"),
                "delta_ci95_lo": (delta.get("bootstrap_ci95_mean_delta") or [None, None])[0],
                "delta_ci95_hi": (delta.get("bootstrap_ci95_mean_delta") or [None, None])[1],
            })
        summary_json["gates"][gate] = gate_block

    # Primary decision rule is fixed before reading evaluation results.
    b_eval = summary_json["gates"]["delayed_all"]["evaluation"]["all_opportunities"]
    p_eval = summary_json["gates"]["lowvol_q50"]["evaluation"]["all_opportunities"]
    p_delta = summary_json["gates"]["lowvol_q50"]["evaluation"]["paired_vs_delayed_all"]
    ci = p_delta.get("bootstrap_ci95_mean_delta", [None, None])
    point_estimate_pass = (
        p_eval.get("mean_return_pct", float("-inf")) > b_eval.get("mean_return_pct", float("-inf"))
        and p_eval.get("max_drawdown_pct", float("-inf")) > b_eval.get("max_drawdown_pct", float("-inf"))
        and p_eval.get("cvar20_return_pct", float("-inf")) > b_eval.get("cvar20_return_pct", float("-inf"))
    )
    statistically_supported = bool(
        point_estimate_pass and ci[0] is not None and ci[0] > 0
    )
    summary_json["primary_decision"] = {
        "gate": "lowvol_q50",
        "point_estimate_pass": bool(point_estimate_pass),
        "paired_delta_ci_excludes_zero": bool(ci[0] is not None and ci[0] > 0),
        "statistically_supported": statistically_supported,
        "rule": "eval opportunity mean, max drawdown and CVaR20 must all improve; paired mean-return delta CI must be > 0 for statistical support",
    }

    if all_results:
        pd.concat(all_results, ignore_index=True).to_parquet(
            out_dir / "active_trades.parquet", index=False
        )
    pd.concat(opportunity_results, ignore_index=True).to_parquet(
        out_dir / "opportunity_results.parquet", index=False
    )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "gate_summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary_json, indent=2, default=str))

    lines = [
        f"# Regime-gated closed-market grid: {args.symbol}",
        "",
        f"Reference market: **{spec.name}**",
        f"Observation delay: **{args.observation_minutes} min**",
        f"Frozen grid params: `{best}`",
        "",
        "Thresholds are fit on development windows only. Evaluation windows are never used to choose a gate.",
        "Only complete closed-market windows are included. Skipped windows count as zero-return opportunities.",
        "",
        "| gate | split | traded/eligible | coverage | net USDT | mean/oppty | mean/active | active win | max DD | CVaR20 | delta vs baseline (CI95) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(v, nd=5):
        return "-" if v is None or pd.isna(v) else f"{v:.{nd}f}"

    for row in summary_rows:
        lines.append(
            f"| {row['gate']} | {row['split']} | {row['traded_windows']}/{row['eligible_windows']} | "
            f"{fmt(row['coverage'], 3)} | {fmt(row['total_net_pnl'], 2)} | "
            f"{fmt(row['mean_return_per_opportunity'])} | {fmt(row['active_mean_return'])} | "
            f"{fmt(row['active_win_rate'], 3)} | {fmt(row['max_drawdown_pct'])} | "
            f"{fmt(row['cvar20_return_pct'])} | {fmt(row['delta_mean_vs_delayed_all'])} "
            f"([{fmt(row['delta_ci95_lo'])}, {fmt(row['delta_ci95_hi'])}]) |"
        )

    decision = summary_json["primary_decision"]
    lines.extend([
        "",
        "## Primary decision rule",
        "",
        "Primary gate: `lowvol_q50`. It must improve evaluation mean return per eligible opportunity, max drawdown, and CVaR20 versus `delayed_all`.",
        "Statistical support additionally requires the paired bootstrap CI for gate-minus-baseline mean return to lie entirely above zero.",
        "",
        f"- point-estimate pass: **{decision['point_estimate_pass']}**",
        f"- paired delta CI excludes zero: **{decision['paired_delta_ci_excludes_zero']}**",
        f"- statistically supported: **{decision['statistically_supported']}**",
        "",
        "`lowvol_q25` and `lowvol_q75` are sensitivity analyses only; they must not be selected post hoc from evaluation performance.",
    ])
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"[done] outputs -> {out_dir}")
    print(json.dumps(summary_json["primary_decision"], indent=2))


if __name__ == "__main__":
    main()
