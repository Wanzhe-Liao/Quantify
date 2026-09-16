"""Aggregate window-level results + bootstrap CI (unit = closed-market window)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_windows(df: pd.DataFrame, initial_capital: float,
                      bootstrap_iters: int = 2000, seed: int = 7) -> dict:
    """Summary stats over per-window records (df rows = sessions).

    Sessions are simulated with fixed ``initial_capital`` for comparable sizing,
    but sequence-level drawdown is computed from cumulative net PnL across
    chronologically ordered windows.  Using each session's independent
    ``final_equity`` would reset the account after every window and understate
    multi-window losing streaks.
    """
    if df.empty:
        return {"n_windows": 0}
    r = df["return_pct"].to_numpy(dtype=float)
    net = df["net_pnl"].to_numpy(dtype=float)
    pos, neg = net[net > 0].sum(), -net[net < 0].sum()
    mean_r, std_r = r.mean(), r.std(ddof=1) if len(r) > 1 else 0.0

    rng = np.random.default_rng(seed)
    boots = rng.choice(r, size=(bootstrap_iters, len(r)), replace=True).mean(axis=1)
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])

    ordered = df.sort_values("window_start")
    cumulative_equity = initial_capital + ordered["net_pnl"].astype(float).cumsum().to_numpy()
    curve = np.concatenate([[initial_capital], cumulative_equity])
    running_max = np.maximum.accumulate(curve)
    dd = (curve / running_max - 1.0).min()

    q20 = float(np.quantile(r, 0.20))
    cvar20 = float(r[r <= q20].mean())
    worst = df.loc[df["net_pnl"].idxmin()]
    best = df.loc[df["net_pnl"].idxmax()]
    return {
        "n_windows": int(len(df)),
        "total_net_pnl": float(net.sum()),
        "total_return_pct": float(net.sum() / initial_capital),
        "mean_return_pct": float(mean_r),
        "median_return_pct": float(np.median(r)),
        "win_rate": float((net > 0).mean()),
        "loss_rate": float((net < 0).mean()),
        "profit_factor": float(pos / neg) if neg > 0 else float("inf"),
        "sharpe_like": float(mean_r / std_r) if std_r > 0 else float("nan"),
        "t_stat": float(mean_r / (std_r / np.sqrt(len(r)))) if std_r > 0 else float("nan"),
        "max_drawdown_pct": float(dd),
        "cvar20_return_pct": cvar20,
        "worst_window_net_pnl": float(worst["net_pnl"]),
        "worst_window_start": str(worst["window_start"]),
        "best_window_net_pnl": float(best["net_pnl"]),
        "best_window_start": str(best["window_start"]),
        "bootstrap_ci95_mean_return": [float(ci_lo), float(ci_hi)],
        "total_funding_pnl": float(df["funding_pnl"].sum()),
        "total_gross_grid_pnl": float(df["gross_grid_pnl"].sum()),
        "total_inventory_liq_pnl": float(df["inventory_liquidation_pnl"].sum()),
        "total_fees": float((df["maker_fees"] + df["taker_fees"]).sum()),
        "total_slippage_cost": float(df["slippage_cost"].sum()),
    }


def dev_eval_split(windows_df: pd.DataFrame, dev_fraction: float = 0.6) -> tuple[pd.Index, pd.Index]:
    """Chronological split of unique window starts."""
    starts = windows_df.sort_values("window_start")["window_start"].unique()
    cut = max(1, int(len(starts) * dev_fraction))
    return pd.Index(starts[:cut]), pd.Index(starts[cut:])


def parameter_surface(results: pd.DataFrame, split: pd.Index,
                      group_cols=("spacing", "range_pct", "levels", "max_inventory_pct")) -> pd.DataFrame:
    """Per-parameter-combo stats on a subset of windows."""
    df = results[results["window_start"].isin(split)]
    if df.empty:
        return pd.DataFrame()
    df = df.assign(total_fees=df["maker_fees"] + df["taker_fees"])
    g = df.groupby(list(group_cols))
    return g.agg(
        n_windows=("net_pnl", "size"),
        mean_return_pct=("return_pct", "mean"),
        total_net_pnl=("net_pnl", "sum"),
        win_rate=("net_pnl", lambda s: (s > 0).mean()),
        worst_net_pnl=("net_pnl", "min"),
        mean_funding=("funding_pnl", "mean"),
        mean_fees=("total_fees", "mean"),
    ).reset_index()
