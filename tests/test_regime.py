"""Tests for causal low-volatility regime features and dev-only thresholds."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.regime import (
    GateThresholds,
    fit_gate_thresholds,
    gate_mask,
    observe_window,
)


def _minute_bars(start: str, n: int, step: float = 0.01) -> pd.DataFrame:
    t = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    px = [100.0 + i * step for i in range(n)]
    return pd.DataFrame({
        "open_time": t,
        "open": px,
        "high": [p + 0.02 for p in px],
        "low": [p - 0.02 for p in px],
        "close": [p + step / 2 for p in px],
    })


def test_observation_is_strictly_pre_trade_and_causal():
    bars = _minute_bars("2026-09-01 08:00", 90)
    ws = pd.Timestamp("2026-09-01 08:00", tz="UTC")
    we = pd.Timestamp("2026-09-01 09:30", tz="UTC")
    a = observe_window(bars, ws, we, observation_minutes=30)
    assert a is not None
    assert a["trade_start"] == pd.Timestamp("2026-09-01 08:30", tz="UTC")
    assert a["observation_bars"] == 30

    # Mutating future bars must not change a feature used to decide whether to trade.
    future = bars.copy()
    future.loc[future.open_time >= a["trade_start"], ["open", "high", "low", "close"]] *= 10
    b = observe_window(future, ws, we, observation_minutes=30)
    for col in ("observation_vol", "observation_return", "observation_range_pct"):
        assert b[col] == pytest.approx(a[col])


def test_short_window_is_rejected():
    bars = _minute_bars("2026-09-01 08:00", 35)
    ws = pd.Timestamp("2026-09-01 08:00", tz="UTC")
    we = pd.Timestamp("2026-09-01 08:35", tz="UTC")
    assert observe_window(
        bars, ws, we, observation_minutes=30, min_trade_minutes=10
    ) is None


def test_thresholds_use_development_rows_only():
    features = pd.DataFrame({
        "window_start": pd.to_datetime([
            "2026-09-01 00:00Z", "2026-09-02 00:00Z",
            "2026-09-03 00:00Z", "2026-09-04 00:00Z",
        ]),
        "observation_vol": [0.001, 0.003, 0.5, 0.9],
        "abs_observation_return": [0.01, 0.03, 0.8, 0.9],
    })
    dev = pd.Index(features.window_start.iloc[:2])
    t = fit_gate_thresholds(features, dev)
    assert t.vol_q50 == pytest.approx(0.002)
    assert t.drift_q75 == pytest.approx(0.025)


def test_primary_gate_and_drift_guard():
    features = pd.DataFrame({
        "observation_vol": [0.001, 0.002, 0.004],
        "abs_observation_return": [0.01, 0.06, 0.01],
    })
    t = GateThresholds(vol_q50=0.003, vol_q25=0.0015, vol_q75=0.0045,
                       drift_q75=0.05)
    assert gate_mask(features, t, "lowvol_q50").tolist() == [True, True, False]
    assert gate_mask(features, t, "lowvol_q50_driftguard").tolist() == [True, False, False]
    assert gate_mask(features, t, "delayed_all").tolist() == [True, True, True]
