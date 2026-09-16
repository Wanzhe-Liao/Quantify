"""Auto-generated research report (report.md).

Conclusion wording is restricted to: promising | inconclusive | not robust.
"""
from __future__ import annotations

import pandas as pd


def _fmt_money(x: float) -> str:
    return f"{x:+.2f} USDT"


def _conclusion(dev: dict, eval_: dict, robust_ok: bool, min_windows: int = 15) -> str:
    if eval_.get("n_windows", 0) < min_windows:
        return "inconclusive (exploratory only: too few evaluation windows)"
    ci = eval_.get("bootstrap_ci95_mean_return", [0, 0])
    if eval_.get("mean_return_pct", 0) > 0 and ci[0] > 0 and robust_ok:
        return "promising"
    if eval_.get("mean_return_pct", 0) <= 0 or not robust_ok:
        return "not robust"
    return "inconclusive"


def render_report(symbol: str, meta: dict, market: str, windows_all: pd.DataFrame,
                  dev_summary: dict, eval_summary: dict, best_params: dict,
                  robustness: pd.DataFrame, strategy_summaries: dict,
                  strat_windows: dict) -> str:
    cons_5 = robustness[(robustness.tag == "slippage") &
                        (robustness.exit_slippage_bps == 5)] if len(robustness) else robustness
    robust_ok = bool(len(cons_5) and cons_5.iloc[0]["mean_return_pct"] > 0)
    verdict = _conclusion(dev_summary, eval_summary, robust_ok)

    wtype_counts = windows_all["window_type"].value_counts().to_dict() if len(windows_all) else {}

    lines = [
        f"# Closed-market grid backtest: {symbol}",
        "",
        f"data source: `{meta.get('data_source')}` | contract: `{meta.get('contractType')}` "
        f"| underlying: `{meta.get('underlyingType')}` | market: {market}",
        "",
        "## Current finding",
        "",
        f"**{verdict}**",
        "",
    ]
    if eval_summary.get("n_windows"):
        src = []
        if eval_summary["total_gross_grid_pnl"]:
            src.append(f"completed grid cycles {_fmt_money(eval_summary['total_gross_grid_pnl'])}")
        if eval_summary["total_inventory_liq_pnl"]:
            src.append(f"inventory/liquidation {_fmt_money(eval_summary['total_inventory_liq_pnl'])}")
        if eval_summary["total_funding_pnl"]:
            src.append(f"funding {_fmt_money(eval_summary['total_funding_pnl'])}")
        src.append(f"fees -{eval_summary['total_fees']:.2f} USDT")
        src.append(f"exit slippage -{eval_summary['total_slippage_cost']:.2f} USDT")
        lines += [
            f"closed-market grid net pnl (eval, best dev params): "
            f"**{_fmt_money(eval_summary['total_net_pnl'])}** over "
            f"{eval_summary['n_windows']} windows "
            f"(mean {eval_summary['mean_return_pct']*100:+.3f}%/window, "
            f"CI95 [{eval_summary['bootstrap_ci95_mean_return'][0]*100:+.3f}%, "
            f"{eval_summary['bootstrap_ci95_mean_return'][1]*100:+.3f}%]).",
            "",
            "P&L attribution: " + "; ".join(src) + ".",
            "",
        ]
    lines += [
        "## Evidence",
        "",
        "| split | n | total net | mean%/win | win rate | max DD | worst window |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in (("development", dev_summary), ("evaluation", eval_summary)):
        if s.get("n_windows"):
            lines.append(
                f"| {name} | {s['n_windows']} | {_fmt_money(s['total_net_pnl'])} "
                f"| {s['mean_return_pct']*100:+.3f} | {s['win_rate']:.0%} "
                f"| {s['max_drawdown_pct']*100:.2f}% "
                f"| {_fmt_money(s['worst_window_net_pnl'])} @ {s['worst_window_start'][:16]} |")
    lines += ["", "### Strategy benchmarks (same grid params)", ""]
    for k, s in strategy_summaries.items():
        if s.get("n_windows"):
            lines.append(
                f"- **{k}**: n={s['n_windows']}, net {_fmt_money(s['total_net_pnl'])}, "
                f"mean {s['mean_return_pct']*100:+.3f}%/window, win {s['win_rate']:.0%}")
    lines += [
        "",
        f"window types traded: {wtype_counts}",
        "",
        "## Failure modes",
        "",
        _failure_modes(windows_all, eval_summary),
        "",
        "## Robustness (best dev params, evaluation windows)",
        "",
        "| test | setting | mean%/win | total net | win rate |",
        "|---|---|---|---|---|",
    ]
    for _, r in robustness.iterrows():
        lines.append(
            f"| {r['tag']} | {r['setting']} | {r['mean_return_pct']*100:+.3f} "
            f"| {_fmt_money(r['total_net_pnl'])} | {r['win_rate']:.0%} |")
    lines += [
        "",
        "## Conclusion",
        "",
        f"**{verdict}**.",
        "",
        "Caveats: short sample, single-contract evidence, kline-based fills "
        "(no queue/liquidity model), funding settled at Binance mark price.",
    ]
    return "\n".join(lines) + "\n"


def _failure_modes(windows: pd.DataFrame, s: dict) -> str:
    if not len(windows):
        return "- insufficient data"
    out = []
    worst = windows.loc[windows.net_pnl.idxmin()]
    out.append(
        f"- worst window {worst['window_start']} ({worst['window_type']}): "
        f"net {_fmt_money(worst['net_pnl'])}, underlying moved "
        f"{worst['window_return']*100:+.2f}% -> "
        + ("one-way trend overwhelmed the grid" if abs(worst['window_return']) > 0.01
           else "fees/drag exceeded grid capture"))
    tot = windows["net_pnl"].sum()
    for col, label, sign in (("inventory_liquidation_pnl", "tail inventory liquidation", +1),
                             ("funding_pnl", "funding", +1),
                             ("slippage_cost", "exit slippage cost", -1)):
        v = windows[col].sum() * sign
        if v and tot and abs(v) > 0.2 * abs(tot):
            out.append(f"- {label}: {_fmt_money(v)} total ({v / tot:.0%} of net)")
    if len(out) == 1:
        out.append("- no dominant single failure mode in this sample")
    return "\n".join(out)
