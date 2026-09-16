import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.atr_dynamic import ATRCandidate, candidate_registry, run_array
from scripts.run_zhongji_atr_lab import prepare_atr


def _matrix(n=180, amp=0.7, atr=0.4):
    # 19-column causal feature matrix expected by atr_dynamic.
    a = np.zeros((n, 19), dtype=float)
    base = 100 + amp * np.sin(np.arange(n) / 3.0)
    a[:, 0] = np.r_[100, base[:-1]]
    a[:, 3] = base
    a[:, 1] = np.maximum(a[:, 0], a[:, 3]) + 0.5
    a[:, 2] = np.minimum(a[:, 0], a[:, 3]) - 0.5
    a[:, 4] = 10000
    a[:, 5] = np.r_[100, base[:-1]]
    a[:, 6] = a[:, 5]
    a[:, 7] = a[:, 5]
    a[:, 8] = a[:, 5]
    a[:, 9:14] = atr
    a[:, 14] = 0.001
    a[:, 15] = 0.0
    a[:, 16] = 0.0
    a[:, 17] = 0.2
    a[:, 18] = 0.0
    return a


def test_registry_is_bounded_and_unique():
    reg = candidate_registry()
    assert len(reg) == 870
    assert len({c.name for c in reg}) == len(reg)
    assert {c.family for c in reg} >= {
        "atr_refresh_v2", "atr_inventory", "atr_combo",
        "atr_ratio_adaptive", "atr_shock_pause",
    }


def test_atr_multiplier_changes_fill_activity():
    a = _matrix()
    narrow = ATRCandidate("n", "t", atr_idx=1, atr_mult=0.5, levels=2,
                          refresh=0, anchor=0)
    wide = ATRCandidate("w", "t", atr_idx=1, atr_mult=3.0, levels=2,
                        refresh=0, anchor=0)
    rn, _ = run_array(a, narrow, tick=.01, step=.01)
    rw, _ = run_array(a, wide, tick=.01, step=.01)
    assert rn["fills"] >= rw["fills"]
    assert abs(rn["reconciliation_error"]) < 1e-8
    assert abs(rw["reconciliation_error"]) < 1e-8


def test_vol_target_reduces_risk_in_high_atr():
    a = _matrix(atr=1.5)
    fixed = ATRCandidate("f", "t", atr_idx=1, atr_mult=0.5, levels=2,
                         refresh=15, anchor=1, inventory=.25)
    scaled = ATRCandidate("s", "t", atr_idx=1, atr_mult=0.5, levels=2,
                          refresh=15, anchor=1, inventory=.25, vol_target=.001)
    rf, _ = run_array(a, fixed, tick=.01, step=.01)
    rs, _ = run_array(a, scaled, tick=.01, step=.01)
    assert rs["max_gross_marked_notional"] <= rf["max_gross_marked_notional"] + 1e-9


def test_prepare_atr_features_are_lagged():
    n = 160
    t = pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC")
    close = np.linspace(100, 110, n)
    k = pd.DataFrame({
        "open_time": t,
        "open": close,
        "high": close + .2,
        "low": close - .2,
        "close": close,
        "volume": 100.0,
    })
    f = pd.DataFrame({"funding_time": pd.to_datetime([], utc=True),
                      "funding_rate": [], "mark_price": []})
    a1 = prepare_atr(k, f)
    k2 = k.copy()
    k2.loc[n-1, ["open", "high", "low", "close"]] = [999, 1000, 998, 999]
    a2 = prepare_atr(k2, f)
    # Changing the final bar cannot alter features or prices of prior rows.
    np.testing.assert_allclose(a1[:-1], a2[:-1], equal_nan=True)
