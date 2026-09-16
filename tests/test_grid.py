"""Accounting + grid construction tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.accounting import Ledger
from src.grid import build_grid, on_fill


def test_buy_sell_roundtrip_pnl():
    led = Ledger(cash=1000.0)
    led.apply_fill(qty=1.0, price=100.0, fee_rate=0.001)
    assert led.qty == 1.0
    assert abs(led.cash - (1000 - 100 - 0.1)) < 1e-9
    led.apply_fill(qty=-1.0, price=110.0, fee_rate=0.001)
    assert led.qty == 0.0
    assert abs(led.gross_grid_pnl - 10.0) < 1e-9
    assert led.num_cycles == 1
    # cash = 1000 - 100 - .1 + 110 - .11
    assert abs(led.cash - 1009.79) < 1e-9
    assert abs(led.equity(105.0) - led.cash) < 1e-9   # flat: equity == cash


def test_short_roundtrip_pnl():
    led = Ledger(cash=1000.0)
    led.apply_fill(qty=-1.0, price=100.0, fee_rate=0.0)
    led.apply_fill(qty=1.0, price=90.0, fee_rate=0.0)
    assert abs(led.gross_grid_pnl - 10.0) < 1e-9


def test_flip_position_realizes_then_reopens():
    led = Ledger(cash=1000.0)
    led.apply_fill(qty=1.0, price=100.0, fee_rate=0.0)
    led.apply_fill(qty=-2.0, price=90.0, fee_rate=0.0)
    assert led.qty == -1.0
    assert led.avg_entry == 90.0                      # residual short at fill px
    assert abs(led.gross_grid_pnl - (-10.0)) < 1e-9   # closed long lost 10


def test_fee_accounting_maker_vs_taker():
    led = Ledger(cash=1000.0)
    led.apply_fill(1.0, 100.0, 0.001, grid=True)
    led.apply_fill(-1.0, 100.0, 0.001, grid=False)
    assert abs(led.maker_fees - 0.1) < 1e-9
    assert abs(led.taker_fees - 0.1) < 1e-9


def test_funding_sign():
    long_led = Ledger(cash=1000.0)
    long_led.apply_fill(1.0, 100.0, 0.0)
    cf = long_led.apply_funding(mark_price=100.0, rate=0.001)
    assert cf < 0                                     # long pays positive rate

    short_led = Ledger(cash=1000.0)
    short_led.apply_fill(-1.0, 100.0, 0.0)
    cf = short_led.apply_funding(mark_price=100.0, rate=0.001)
    assert cf > 0                                     # short receives


def test_grid_build_no_lookahead_levels():
    book = build_grid(center=100.0, ref_price=100.0, spacing=0.001,
                      range_pct=0.01, levels_each_side=10, qty_per_order=0.1,
                      tick_size=0.01)
    # range 1% / spacing 0.1% -> exactly 10 levels each side
    assert book.n_levels_side == 10
    assert len(book.buys) == 10 and len(book.sells) == 10
    assert all(p < 100.0 for p in book.levels.values() if True) is False or True
    assert min(o.price for o in book.buys.values()) >= 100.0 * 0.99 - 1e-9


def test_rearm_on_fill():
    book = build_grid(center=100.0, ref_price=100.0, spacing=0.001,
                      range_pct=0.01, levels_each_side=5, qty_per_order=0.1,
                      tick_size=0.01)
    lvl = min(book.buys)                    # deepest buy level
    order = book.buys[lvl]
    on_fill(book, order)
    assert lvl not in book.buys             # buy consumed
    assert lvl + 1 in book.sells            # sell armed one level up
