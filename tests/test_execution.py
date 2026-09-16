"""Execution-mode tests: strict touches, no intra-bar cascade, inventory cap."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.accounting import Ledger
from src.execution import process_bar
from src.grid import build_grid


def _bar(o, h, l, c):
    return SimpleNamespace(open=o, high=h, low=l, close=c)


def _book(center=100.0, spacing=0.01, levels=5, qty=1.0):
    return build_grid(center=center, ref_price=center, spacing=spacing,
                      range_pct=spacing * levels, levels_each_side=levels,
                      qty_per_order=qty, tick_size=0.01)


def test_optimistic_fills_on_touch():
    book = _book()
    led = Ledger(cash=1000.0)
    rng = np.random.default_rng(0)
    # low touches the -1% buy level exactly (99.0)
    process_bar(_bar(100, 100.5, 99.0, 100.2), book, led, 1e9, 0.0,
                "optimistic", 0.0, rng)
    assert led.qty > 0


def test_conservative_requires_strict_penetration():
    book = _book()
    led = Ledger(cash=1000.0)
    rng = np.random.default_rng(0)
    # low == buy price exactly -> NOT a fill under conservative
    process_bar(_bar(100, 100.5, 99.0, 100.2), book, led, 1e9, 0.0,
                "conservative", 0.0, rng)
    assert led.qty == 0.0
    # low below the level -> fill
    process_bar(_bar(100, 100.5, 98.9, 100.2), book, led, 1e9, 0.0,
                "conservative", 0.0, rng)
    assert led.qty > 0


def test_conservative_no_intrabar_cascade():
    """Bar sweeps buys at 99/98 and sells at 101. Resting-at-start orders fill;
    orders armed mid-bar must NOT fill under conservative mode."""
    book = _book(spacing=0.01, levels=2, qty=1.0)   # levels +/-1%,+/-2%
    led = Ledger(cash=1000.0)
    rng = np.random.default_rng(0)
    process_bar(_bar(100.5, 101.5, 97.5, 100.0), book, led, 1e9, 0.0,
                "conservative", 0.0, rng)
    # fills: buy@99, buy@98, sell@101 -> net +1, 3 fills
    assert led.num_fills == 3
    assert led.qty == 1.0
    # sell@101 closes against avg entry (99+98)/2 -> +2.5 realized
    assert abs(led.gross_grid_pnl - 2.5) < 1e-9


def test_optimistic_allows_cascade():
    book = _book(spacing=0.01, levels=2, qty=1.0)
    led = Ledger(cash=1000.0)
    rng = np.random.default_rng(0)
    process_bar(_bar(100.5, 101.5, 97.5, 100.0), book, led, 1e9, 0.0,
                "optimistic", 0.0, rng)
    assert led.num_fills > 3        # re-armed orders also fill


def test_inventory_cap_blocks_same_direction():
    book = _book(spacing=0.005, levels=4, qty=1.0)
    led = Ledger(cash=1000.0)
    rng = np.random.default_rng(0)
    cap = 150.0                                     # allows ~1.5 units notional
    # bar sweeps through all 4 buy levels
    process_bar(_bar(100, 100, 96.0, 97.0), book, led, cap, 0.0,
                "optimistic", 0.0, rng)
    assert led.qty <= 1.0 + 1e-9                    # second buy would breach cap
    assert led.num_fills == 1


def test_drop_prob_deterministic():
    b1, b2 = _book(), _book()
    l1, l2 = Ledger(cash=1000.0), Ledger(cash=1000.0)
    r1, r2 = np.random.default_rng(3), np.random.default_rng(3)
    bar = _bar(100, 102.5, 97.5, 100.0)
    process_bar(bar, b1, l1, 1e9, 0.0, "conservative", 0.25, r1)
    process_bar(bar, b2, l2, 1e9, 0.0, "conservative", 0.25, r2)
    assert l1.num_fills == l2.num_fills and l1.cash == l2.cash
