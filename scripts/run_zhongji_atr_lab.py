#!/usr/bin/env python3
"""Targeted ZHONGJIUSDT ATR/dynamic-grid research.

The audited period has already been inspected in prior work, so every result in
this script is retrospective/exploratory.  No result is promoted to a fresh
holdout.  Selection logic is causal within the historical walk-forward.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.atr_dynamic import candidate_registry, run_array
from scripts.run_grid_lab import load_cached, windows_for, es20, bootstrap_mean

ROOT = Path(__file__).resolve().parents[1]
CUTOFF = pd.Timestamp("2026-09-01", tz="UTC")


def prepare_atr(k: pd.DataFrame, funding: pd.DataFrame) -> np.ndarray:
    c = k.close.astype(float)
    ret = c.pct_change(fill_method=None)
    tr = pd.concat([
        k.high - k.low,
        (k.high - c.shift()).abs(),
        (k.low - c.shift()).abs(),
    ], axis=1).max(axis=1)
    vw = ((c * k.volume).rolling(60, min_periods=30).sum() /
          k.volume.rolling(60, min_periods=30).sum().replace(0, np.nan))
    efficiency = ((c - c.shift(30)).abs() /
                  c.diff().abs().rolling(30).sum().replace(0, np.nan))
    feat = pd.DataFrame({
        "prev": c.shift(),
        "ema30": c.ewm(span=30, adjust=False).mean().shift(),
        "ema120": c.ewm(span=120, adjust=False).mean().shift(),
        "vwap60": vw.fillna(c).shift(),
        "atr10": tr.rolling(10, min_periods=10).mean().shift(),
        "atr30": tr.rolling(30, min_periods=30).mean().shift(),
        "atr60": tr.rolling(60, min_periods=60).mean().shift(),
        "atr120": tr.rolling(120, min_periods=120).mean().shift(),
        "atr_ewm30": tr.ewm(span=30, adjust=False).mean().shift(),
        "rv30": ret.rolling(30, min_periods=30).std(ddof=0).shift(),
        "ret10": (c / c.shift(10) - 1).shift(),
        "ret30": (c / c.shift(30) - 1).shift(),
        "eff30": efficiency.fillna(0).shift(),
    })
    ff = (funding.assign(minute=funding.funding_time.dt.floor("min"),
                         weighted=funding.funding_rate * funding.mark_price)
          .groupby("minute").weighted.sum())
    a = np.column_stack([
        k[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float),
        feat.to_numpy(dtype=float),
        k.open_time.map(ff).fillna(0).to_numpy(dtype=float),
    ])
    return a


def summary(df: pd.DataFrame, capital=1000.0) -> dict:
    if df.empty:
        return {"windows": 0}
    v = df.sort_values("window_start").net_pnl.to_numpy(dtype=float)
    eq = np.r_[capital, capital + np.cumsum(v)]
    return {
        "windows": int(len(v)), "net": float(v.sum()), "mean": float(v.mean()),
        "wins": int((v > 1e-8).sum()), "losses": int((v < -1e-8).sum()),
        "zeros": int((np.abs(v) <= 1e-8).sum()),
        "win_rate": float((v > 1e-8).mean()),
        "loss_rate": float((v < -1e-8).mean()),
        "filled_windows": int((df.fills > 0).sum()),
        "min_window": float(v.min()), "max_window": float(v.max()),
        "es20": es20(v),
        "window_end_dd": float(np.min(eq / np.maximum.accumulate(eq) - 1)),
        "worst_intrawindow_dd": float(df.max_drawdown.min()),
        "every_window_profitable": bool(np.all(v > 1e-8)),
    }


def choose_stability(train: pd.DataFrame, min_windows=8) -> str | None:
    """Select without peeking: prioritize positive-window coverage and downside."""
    ranked = []
    for name, g in train.groupby("candidate"):
        if len(g) < min_windows:
            continue
        fill_cov = (g.fills > 0).mean()
        if fill_cov < 0.50 or g.net_pnl.mean() <= 0:
            continue
        s = summary(g)
        key = (-s["win_rate"], s["loss_rate"], -s["es20"], -s["min_window"], -s["mean"], name)
        ranked.append((key, name))
    return sorted(ranked, key=lambda x: x[0])[0][1] if ranked else None


def cash_rows(w: pd.DataFrame, label="CASH") -> pd.DataFrame:
    out = w[["window_start", "window_end", "window_type"]].copy()
    for col in ("net_pnl", "return_pct", "fills", "cycles", "max_drawdown",
                "cycle_pnl", "inventory_pnl", "funding_pnl", "maker_fees",
                "taker_fees", "slippage_cost", "turnover"):
        out[col] = 0.0
    out["candidate"] = "CASH"
    out["family"] = label
    return out


def run_symbol(args) -> None:
    symbol = "ZHONGJIUSDT"
    cfg = yaml.safe_load(Path(args.config).read_text())
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    k, f, meta, hashes = load_cached(symbol, Path(args.data_dir))
    a = prepare_atr(k, f)
    wa = windows_for(k, cfg, symbol)
    wa["atr_usable"] = wa.usable & (wa.start_idx >= 121)
    wa.to_csv(outdir / "window_audit.csv", index=False)
    w = wa[wa.atr_usable].reset_index(drop=True)
    if w.empty:
        raise RuntimeError("no usable ZHONGJI windows")

    reg = candidate_registry()
    print(f"ZHONGJI ATR lab: {len(reg)} candidates x {len(w)} windows", flush=True)
    base_kw = dict(tick=meta["tick_size"], step=meta["step_size"])
    result_path = outdir / "all_windows.csv.gz"
    if args.reuse_results and result_path.exists():
        results = pd.read_csv(result_path)
        results["window_start"] = pd.to_datetime(results["window_start"], utc=True)
        results["window_end"] = pd.to_datetime(results["window_end"], utc=True)
        print(f"Reusing {len(results)} saved candidate-window results", flush=True)
    else:
        rows = []
        for ci, cand in enumerate(reg):
            for wr in w.itertuples():
                rec, _ = run_array(a[wr.start_idx:wr.end_idx], cand, **base_kw)
                rec.update(window_start=wr.window_start, window_end=wr.window_end,
                           window_type=wr.window_type, symbol=symbol)
                rows.append(rec)
            if (ci + 1) % 50 == 0 or ci + 1 == len(reg):
                print(f"  {ci+1}/{len(reg)} candidates", flush=True)
        results = pd.DataFrame(rows)
        results.to_csv(result_path, index=False)

    dev = results[results.window_end <= CUTOFF]
    audit = results[results.window_start >= CUTOFF]
    score_rows = []
    for name, g in results.groupby("candidate"):
        for label, mask in (
            ("development", g.window_end <= CUTOFF),
            ("historical_audit", g.window_start >= CUTOFF),
            ("all_seen", g.window_start.notna()),
        ):
            score_rows.append(dict(candidate=name, family=g.family.iloc[0], split=label,
                                   **summary(g[mask])))
    pd.DataFrame(score_rows).to_csv(outdir / "candidate_summary.csv", index=False)

    strict = {}
    for label, frame in (("development", dev), ("historical_audit", audit), ("all_seen", results)):
        per = frame.groupby("candidate").net_pnl.agg(["min", "mean", "count"])
        good = per[per["min"] > 1e-8].sort_values(["min", "mean"], ascending=False)
        strict[label] = {
            "windows": int(frame.window_start.nunique()),
            "every_window_positive_candidates": int(len(good)),
            "candidates": good.head(20).reset_index().to_dict("records"),
        }

    hard_rows = []
    for start, g in results.groupby("window_start"):
        best = g.loc[g.net_pnl.idxmax()]
        hard_rows.append({
            "window_start": start, "window_end": best.window_end,
            "window_type": best.window_type,
            "positive_candidates": int((g.net_pnl > 1e-8).sum()),
            "candidate_count": int(len(g)),
            "oracle_best_pnl": float(best.net_pnl),
            "oracle_best_candidate": best.candidate,
            "median_pnl": float(g.net_pnl.median()),
        })
    hard = pd.DataFrame(hard_rows).sort_values("window_start")
    hard.to_csv(outdir / "hard_windows.csv", index=False)

    frozen_rows, selections = [], []
    global_name = choose_stability(dev, min_windows=12)
    for typ in ["ALL"] + sorted(w.window_type.unique()):
        d = dev if typ == "ALL" else dev[dev.window_type == typ]
        minw = 12 if typ == "ALL" else 5
        name = choose_stability(d, min_windows=minw)
        if typ != "ALL" and name is None:
            name = global_name
        test_w = w[w.window_start >= CUTOFF]
        if typ != "ALL":
            test_w = test_w[test_w.window_type == typ]
        chosen = audit[(audit.candidate == name) & audit.window_start.isin(test_w.window_start)].copy() if name else cash_rows(test_w)
        chosen["policy"] = typ
        frozen_rows.append(chosen)
        selections.append(dict(policy=typ, candidate=name or "CASH", **summary(chosen)))
    pd.DataFrame(selections).to_csv(outdir / "frozen_selections.csv", index=False)
    pd.concat(frozen_rows, ignore_index=True).to_csv(outdir / "frozen_windows.csv", index=False)

    local_weeks = (w.window_start.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
                   .dt.to_period("W-SUN").dt.start_time)
    wf_global, wf_typed, decisions = [], [], []
    for week in sorted(local_weeks.unique()):
        boundary = pd.Timestamp(week, tz="Asia/Shanghai").tz_convert("UTC")
        past = results[results.window_end <= boundary]
        if past.window_start.nunique() < 12:
            continue
        target = w[local_weeks == week]
        global_pick = choose_stability(past, min_windows=12)
        for wr in target.itertuples():
            if global_pick:
                r = results[(results.candidate == global_pick) & (results.window_start == wr.window_start)].copy()
            else:
                r = cash_rows(pd.DataFrame([wr._asdict()]))
            r["selection"] = "global_weekly"
            r["fit_boundary"] = boundary
            wf_global.append(r)
            ptype = past[past.window_type == wr.window_type]
            typed_pick = choose_stability(ptype, min_windows=5) or global_pick
            if typed_pick:
                t = results[(results.candidate == typed_pick) & (results.window_start == wr.window_start)].copy()
            else:
                t = cash_rows(pd.DataFrame([wr._asdict()]))
            t["selection"] = "type_weekly"
            t["fit_boundary"] = boundary
            wf_typed.append(t)
            decisions.append({"week": str(week), "fit_boundary": boundary,
                              "window_start": wr.window_start, "window_type": wr.window_type,
                              "global_candidate": global_pick or "CASH",
                              "typed_candidate": typed_pick or "CASH",
                              "past_windows": int(past.window_start.nunique()),
                              "past_type_windows": int(ptype.window_start.nunique())})
    wf_global = pd.concat(wf_global, ignore_index=True) if wf_global else pd.DataFrame()
    wf_typed = pd.concat(wf_typed, ignore_index=True) if wf_typed else pd.DataFrame()
    pd.concat([wf_global, wf_typed], ignore_index=True).to_csv(outdir / "walkforward_windows.csv", index=False)
    pd.DataFrame(decisions).to_csv(outdir / "walkforward_decisions.csv", index=False)
    wf_summary = []
    for label, frame in (("global_weekly", wf_global), ("type_weekly", wf_typed)):
        s = summary(frame)
        s["ci95_mean_usdt"] = bootstrap_mean(frame) if not frame.empty else [None, None]
        wf_summary.append(dict(selection=label, **s))
        if not frame.empty:
            for typ, g in frame.groupby("window_type"):
                t = summary(g)
                t["ci95_mean_usdt"] = bootstrap_mean(g)
                wf_summary.append(dict(selection=f"{label}:{typ}", **t))
    pd.DataFrame(wf_summary).to_csv(outdir / "walkforward_summary.csv", index=False)

    stress = []
    if global_name:
        byname = {c.name: c for c in reg}
        scenarios = {
            "base": {},
            "fees_slippage_2x": dict(maker=.0004, taker=.0010, slip_bps=10),
            "fill_drop25": dict(fill_drop=.25),
            "volume_cap_0.1pct": dict(participation=.001),
            "buy_first": dict(path_mode=1),
            "sell_first": dict(path_mode=2),
            "combined": dict(maker=.0004, taker=.0010, slip_bps=10,
                             fill_drop=.25, participation=.001),
        }
        for scenario, kw in scenarios.items():
            rr = []
            for wr in w[w.window_start >= CUTOFF].itertuples():
                rec, _ = run_array(a[wr.start_idx:wr.end_idx], byname[global_name], **base_kw, **kw)
                rec.update(window_start=wr.window_start, window_type=wr.window_type)
                rr.append(rec)
            stress.append(dict(candidate=global_name, scenario=scenario, **summary(pd.DataFrame(rr))))
    pd.DataFrame(stress).to_csv(outdir / "stress.csv", index=False)

    fam_rows = []
    for fam, g in audit.groupby("family"):
        per = []
        for name, x in g.groupby("candidate"):
            s = summary(x)
            per.append((s["win_rate"], s["es20"], s["mean"], name, s))
        per.sort(reverse=True, key=lambda q: (q[0], q[1], q[2]))
        best = per[0]
        fam_rows.append(dict(family=fam, candidate=best[3], **best[4]))
    pd.DataFrame(fam_rows).sort_values(["win_rate", "es20", "mean"], ascending=False).to_csv(
        outdir / "family_diagnostics.csv", index=False)

    protocol = {
        "status": "retrospective_exploration_no_fresh_holdout",
        "symbol": symbol,
        "candidates": len(reg),
        "candidate_families": len(set(c.family for c in reg)),
        "development_cutoff_utc": str(CUTOFF),
        "selection_objective": "maximize positive-window rate, then minimize loss rate, then maximize ES20, worst window, mean; positive mean and >=50% fill coverage required",
        "success": "a single causal policy must produce strictly positive net PnL in every eligible window; zero/no-fill does not count",
        "execution": "maker 2bps, taker 5bps, 5bps forced-exit slippage, <=1% bar volume participation, adverse OHLC side ordering",
        "data_hashes": hashes,
        "registry_sha256": hashlib.sha256(json.dumps([asdict(x) for x in reg], sort_keys=True).encode()).hexdigest(),
    }
    (outdir / "protocol.json").write_text(json.dumps(protocol, indent=2, default=str))

    report = {
        "protocol": protocol,
        "strict_search": strict,
        "oracle": {
            "windows": int(len(hard)),
            "windows_with_any_profitable_candidate": int((hard.positive_candidates > 0).sum()),
            "windows_with_no_profitable_candidate": int((hard.positive_candidates == 0).sum()),
            "min_oracle_best_pnl": float(hard.oracle_best_pnl.min()),
        },
        "frozen_selections": selections,
        "walkforward": wf_summary,
        "hardest_windows": hard.nsmallest(10, "oracle_best_pnl").to_dict("records"),
    }
    (outdir / "summary.json").write_text(json.dumps(report, indent=2, default=str))

    lines = [
        "# ZHONGJIUSDT targeted dynamic ATR grid laboratory", "",
        "All dates in this report were already inspected in earlier research; this is retrospective exploration, not a fresh holdout.", "",
        f"- candidates: **{len(reg)}** across **{len(set(c.family for c in reg))}** ATR/dynamic families",
        f"- usable closed-market windows: **{len(w)}**",
        f"- strict all-seen every-window-positive candidates: **{strict['all_seen']['every_window_positive_candidates']}**",
        f"- historical-audit every-window-positive candidates: **{strict['historical_audit']['every_window_positive_candidates']}**",
        f"- windows with no profitable candidate even under per-window oracle: **{report['oracle']['windows_with_no_profitable_candidate']}**",
        "", "## Frozen development selections", "",
        pd.DataFrame(selections).to_csv(index=False), "",
        "## Causal weekly walk-forward", "",
        pd.DataFrame(wf_summary).to_csv(index=False), "",
        "## Interpretation", "",
        "A per-window oracle is diagnostic only and is not tradable. A strategy is not called stable merely because a post-hoc candidate or window-specific oracle is profitable.",
    ]
    (outdir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "strict": strict,
        "oracle": report["oracle"],
        "frozen": selections,
        "walkforward": wf_summary,
    }, indent=2, default=str))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--output", default="outputs/zhongji_atr_dynamic")
    p.add_argument("--reuse-results", action="store_true",
                   help="reuse an existing all_windows.csv.gz and only rerun summaries")
    args = p.parse_args()
    run_symbol(args)


if __name__ == "__main__":
    main()
