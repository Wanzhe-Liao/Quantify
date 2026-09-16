"""Symmetric fixed-grid order book.

Levels: L_i = center * (1 + i * spacing), i in [-N, +N], |i*spacing| <= range.
Order at level i is a BUY if L_i < reference price at window start, SELL if above.
On a buy fill at level i, a sell order is (re)armed at level i+1.
On a sell fill at level i, a buy order is (re)armed at level i-1.
Fixed order qty per level; center never moves.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Order:
    level: int            # grid index i
    side: str             # 'buy' | 'sell'
    price: float
    qty: float


@dataclass
class GridBook:
    center: float
    spacing: float
    range_pct: float
    qty_per_order: float
    tick_size: float
    levels: dict[int, float] = field(default_factory=dict)   # i -> price
    buys: dict[int, Order] = field(default_factory=dict)     # armed orders by level
    sells: dict[int, Order] = field(default_factory=dict)

    @property
    def n_levels_side(self) -> int:
        return sum(1 for i in self.levels if i > 0)


def build_grid(center: float, ref_price: float, spacing: float, range_pct: float,
               levels_each_side: int, qty_per_order: float, tick_size: float) -> GridBook:
    """Build level ladder + initial resting orders around ref_price.

    Uses only information available at window start (center/ref are current prices).
    """
    n = min(levels_each_side, int(range_pct / spacing + 1e-12))
    levels = {i: round(center * (1 + i * spacing) / tick_size) * tick_size
              for i in range(-n, n + 1) if i != 0}
    book = GridBook(center=center, spacing=spacing, range_pct=range_pct,
                    qty_per_order=qty_per_order, tick_size=tick_size, levels=levels)
    for i, p in levels.items():
        if p < ref_price:
            book.buys[i] = Order(i, "buy", p, qty_per_order)
        elif p > ref_price:
            book.sells[i] = Order(i, "sell", p, qty_per_order)
    return book


def on_fill(book: GridBook, order: Order) -> None:
    """After `order` fills, arm the opposite order one level toward the center's
    other side (standard grid: buy at i -> sell at i+1)."""
    if order.side == "buy":
        book.buys.pop(order.level, None)
        nxt = order.level + 1
        if nxt in book.levels:
            book.sells[nxt] = Order(nxt, "sell", book.levels[nxt], book.qty_per_order)
    else:
        book.sells.pop(order.level, None)
        prv = order.level - 1
        if prv in book.levels:
            book.buys[prv] = Order(prv, "buy", book.levels[prv], book.qty_per_order)


def cancel_all(book: GridBook) -> None:
    book.buys.clear()
    book.sells.clear()
