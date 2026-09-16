"""Causal regime features for closed-market grid experiments.

The key rule is that a filter may only use information observed before the
strategy starts trading.  We therefore observe the first N minutes of each
closed-market window, compute simple volatility/trend features, and start the
grid only after that observation period.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GateThresholds:
    vol_q50: float
    vol_q25: float
    vol_q75: float
    drift_q75: float


def observe_window(
    klines: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    observation_minutes: int = 30,
    min_observation_bars: int = 10,
    min_trade_minutes: int = 10,
) -> dict | None:
    """Return causal features from the first part of a closed-market window.

    Bars at/after ``trade_start`` are never used in the features.  Returning
    ``None`` means the window is too short or has insufficient data to support
    the requested observation/trading split.
    """
    if observation_minutes <= 0:
        raise ValueError("observation_minutes must be positive")

    trade_start = window_start + timedelta(minutes=observation_minutes)
    if trade_start + timedelta(minutes=min_trade_minutes) > window_end:
        return None

    obs = klines[
        (klines.open_time >= window_start) & (klines.open_time < trade_start)
    ].sort_values("open_time")
    if len(obs) < min_observation_bars:
        return None

    first_open = float(obs.iloc[0].open)
    last_close = float(obs.iloc[-1].close)
    closes = obs.close.astype(float).to_numpy()
    log_returns = np.diff(np.log(closes)) if len(closes) > 1 else np.array([])
    realized_vol = float(np.std(log_returns, ddof=0)) if len(log_returns) else 0.0
    obs_return = last_close / first_open - 1.0
    high = float(obs.high.max())
    low = float(obs.low.min())
    range_pct = high / low - 1.0 if low > 0 else float("nan")

    return {
        "window_start": window_start,
        "window_end": window_end,
        "trade_start": trade_start,
        "observation_minutes": int(observation_minutes),
        "observation_bars": int(len(obs)),
        "observation_vol": realized_vol,
        "observation_return": float(obs_return),
        "abs_observation_return": float(abs(obs_return)),
        "observation_range_pct": float(range_pct),
    }


def build_observation_table(
    klines: pd.DataFrame,
    windows: pd.DataFrame,
    observation_minutes: int = 30,
    min_observation_bars: int = 10,
    min_trade_minutes: int = 10,
) -> pd.DataFrame:
    """Compute causal observation features for all eligible windows."""
    rows = []
    for w in windows.itertuples(index=False):
        rec = observe_window(
            klines,
            w.window_start,
            w.window_end,
            observation_minutes=observation_minutes,
            min_observation_bars=min_observation_bars,
            min_trade_minutes=min_trade_minutes,
        )
        if rec is not None:
            rec["window_type"] = w.window_type
            rec["complete"] = bool(w.complete)
            rows.append(rec)
    return pd.DataFrame(rows)


def fit_gate_thresholds(features: pd.DataFrame, dev_starts: pd.Index) -> GateThresholds:
    """Fit fixed quantile thresholds using development windows only."""
    dev = features[features.window_start.isin(dev_starts)]
    if dev.empty:
        raise ValueError("no development features available for gate fitting")
    vol = dev.observation_vol.astype(float)
    drift = dev.abs_observation_return.astype(float)
    return GateThresholds(
        vol_q50=float(vol.quantile(0.50)),
        vol_q25=float(vol.quantile(0.25)),
        vol_q75=float(vol.quantile(0.75)),
        drift_q75=float(drift.quantile(0.75)),
    )


def gate_mask(features: pd.DataFrame, thresholds: GateThresholds, gate: str) -> pd.Series:
    """Return a boolean mask for a pre-specified causal gate.

    ``delayed_all`` is the matched baseline: every eligible window trades, but
    only after the same observation delay used by gated strategies.
    ``lowvol_q50`` is the primary regime test.  q25/q75 are sensitivity checks,
    not candidates to be selected on evaluation performance.
    ``lowvol_q50_driftguard`` adds a loose trend guard using the dev 75th
    percentile of absolute observation-period return.
    """
    if gate == "delayed_all":
        return pd.Series(True, index=features.index)
    if gate == "lowvol_q25":
        return features.observation_vol <= thresholds.vol_q25
    if gate == "lowvol_q50":
        return features.observation_vol <= thresholds.vol_q50
    if gate == "lowvol_q75":
        return features.observation_vol <= thresholds.vol_q75
    if gate == "lowvol_q50_driftguard":
        return (
            (features.observation_vol <= thresholds.vol_q50)
            & (features.abs_observation_return <= thresholds.drift_q75)
        )
    raise ValueError(f"unknown gate: {gate}")
