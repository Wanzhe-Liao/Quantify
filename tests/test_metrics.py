"""Tests for sequence-level window metrics."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics import summarize_windows


def _rows(net):
    n = len(net)
    return pd.DataFrame({
        "window_start": pd.date_range("2026-09-01", periods=n, freq="D", tz="UTC"),
        "net_pnl": net,
        "return_pct": [x / 1000.0 for x in net],
        # final_equity deliberately resets per session, as run_session does.
        "final_equity": [1000.0 + x for x in net],
        "funding_pnl": [0.0] * n,
        "gross_grid_pnl": [0.0] * n,
        "inventory_liquidation_pnl": [0.0] * n,
        "maker_fees": [0.0] * n,
        "taker_fees": [0.0] * n,
        "slippage_cost": [0.0] * n,
    })


def test_drawdown_accumulates_across_windows_instead_of_resetting():
    # 1000 -> 990 -> 980 -> 985, so true sequence DD is -2%.
    s = summarize_windows(_rows([-10.0, -10.0, 5.0]), 1000.0, bootstrap_iters=100)
    assert s["max_drawdown_pct"] == pytest.approx(-0.02)
    assert s["total_net_pnl"] == pytest.approx(-15.0)
    assert s["total_return_pct"] == pytest.approx(-0.015)


def test_cvar20_tracks_bad_tail():
    s = summarize_windows(_rows([-20.0, -10.0, 0.0, 10.0, 20.0]),
                          1000.0, bootstrap_iters=100)
    assert s["cvar20_return_pct"] == pytest.approx(-0.02)
