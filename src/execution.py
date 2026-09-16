"""Bar-level fill simulation for resting grid limit orders.

Mode A  optimistic : low <= limit (buy) / high >= limit (sell); fills may cascade
                     within a bar (orders armed mid-bar can fill too); when both
                     sides are touched the ordering giving HIGHER end-of-bar
                     equity is chosen. Screening only.

Mode B  conservative : strict low < limit (buy) / high > limit (sell); only
                     orders resting at bar start can fill (no intra-bar cascade);
                     when a bar touches BOTH sides the intrabar sequence cannot
                     be determined from OHLC, so both orderings are simulated and
                     the one with LOWER end-of-bar equity is kept
                     ("more adverse ordering wins").
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .accounting import Ledger
from .grid import GridBook, Order, on_fill


def _eligible(book: GridBook, o: float, h: float, l: float, strict: bool):
    if strict:
        buys = [od for od in book.buys.values() if l < od.price]
        sells = [od for od in book.sells.values() if h > od.price]
    else:
        buys = [od for od in book.buys.values() if l <= od.price]
        sells = [od for od in book.sells.values() if h >= od.price]
    buys.sort(key=lambda x: -x.price)    # sweep down: highest buy first
    sells.sort(key=lambda x: x.price)    # sweep up: lowest sell first
    return buys, sells


def _try_fill(order: Order, book: GridBook, led: Ledger, cap_notional: float,
              fee: float, drop_prob: float, rng: np.random.Generator) -> bool:
    if drop_prob > 0.0 and rng.random() < drop_prob:
        return False                                    # queue/latency miss
    qty = order.qty if order.side == "buy" else -order.qty
    new_qty = led.qty + qty
    if abs(new_qty) * order.price > cap_notional + 1e-9 and abs(new_qty) > abs(led.qty):
        return False                                    # inventory cap
    led.apply_fill(qty, order.price, fee, grid=True)
    on_fill(book, order)
    return True


def _sequence(orders: list[Order], book: GridBook, led: Ledger, cap: float,
              fee: float, drop: float, rng) -> int:
    n = 0
    for od in orders:
        if _try_fill(od, book, led, cap, fee, drop, rng):
            n += 1
    return n


def _snapshot(book: GridBook, led: Ledger):
    return dict(book.buys), dict(book.sells), replace(led)


def _restore(book: GridBook, led: Ledger, snap) -> None:
    buys, sells, led_state = snap
    book.buys, book.sells = buys, sells
    for f in led_state.__dataclass_fields__:
        setattr(led, f, getattr(led_state, f))


def process_bar(bar, book: GridBook, led: Ledger, cap_notional: float,
                maker_fee: float, mode: str, drop_prob: float,
                rng: np.random.Generator) -> None:
    o, h, l, c = bar.open, bar.high, bar.low, bar.close
    if mode == "optimistic":
        for _ in range(4 * (len(book.levels) + 1)):     # bounded cascade
            buys, sells = _eligible(book, o, h, l, strict=False)
            if not buys and not sells:
                return
            if c >= o:                                  # up bar: O->L->H->C
                _sequence(buys, book, led, cap_notional, maker_fee, drop_prob, rng)
                _sequence(sells, book, led, cap_notional, maker_fee, drop_prob, rng)
            else:                                       # down bar: O->H->L->C
                _sequence(sells, book, led, cap_notional, maker_fee, drop_prob, rng)
                _sequence(buys, book, led, cap_notional, maker_fee, drop_prob, rng)
        return

    # conservative
    buys, sells = _eligible(book, o, h, l, strict=True)
    if not buys and not sells:
        return
    if buys and sells:
        # ambiguous intrabar order: keep the worse end-of-bar equity.
        # Each candidate ordering gets its own derived RNG so fill-drop
        # randomness cannot leak between scenarios (deterministic given rng).
        snap0 = _snapshot(book, led)
        seed_a, seed_b = (int(x) for x in rng.integers(0, 2**32, 2))
        _sequence(buys, book, led, cap_notional, maker_fee, drop_prob,
                  np.random.default_rng(seed_a))
        _sequence(sells, book, led, cap_notional, maker_fee, drop_prob,
                  np.random.default_rng(seed_a))
        eq_a, snap_a = led.equity(c), _snapshot(book, led)
        _restore(book, led, snap0)
        _sequence(sells, book, led, cap_notional, maker_fee, drop_prob,
                  np.random.default_rng(seed_b))
        _sequence(buys, book, led, cap_notional, maker_fee, drop_prob,
                  np.random.default_rng(seed_b))
        eq_b, snap_b = led.equity(c), _snapshot(book, led)
        _restore(book, led, snap_a if eq_a <= eq_b else snap_b)
    elif buys:
        _sequence(buys, book, led, cap_notional, maker_fee, drop_prob, rng)
    else:
        _sequence(sells, book, led, cap_notional, maker_fee, drop_prob, rng)
