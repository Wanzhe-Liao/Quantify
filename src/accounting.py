"""Full-account ledger: cash, signed inventory, average-cost realized PnL.

Conventions
  qty > 0  long ; qty < 0  short (perpetual, no borrow constraint beyond cap)
  cash changes by -qty*price on every fill (buy spends, sell receives)
  fee charged on |qty*price|, debited from cash immediately
  equity(mark) = cash + qty * mark
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Ledger:
    cash: float
    qty: float = 0.0
    avg_entry: float = 0.0
    maker_fees: float = 0.0
    taker_fees: float = 0.0
    funding_pnl: float = 0.0
    gross_grid_pnl: float = 0.0          # realized pnl of in-window grid closes
    num_cycles: int = 0                  # completed round trips
    num_fills: int = 0
    turnover: float = 0.0                # sum |qty*price|
    max_long_notional: float = 0.0
    max_short_notional: float = 0.0

    def equity(self, mark: float) -> float:
        return self.cash + self.qty * mark

    def apply_fill(self, qty: float, price: float, fee_rate: float,
                   grid: bool = True) -> float:
        """Apply one fill. qty>0 buy, qty<0 sell. Returns realized pnl of the
        closing component (0 if the fill only opens/adds)."""
        notional = qty * price
        fee = abs(notional) * fee_rate
        self.cash -= notional + fee
        if fee_rate and not grid:
            self.taker_fees += fee
        elif fee_rate:
            self.maker_fees += fee
        self.turnover += abs(notional)
        self.num_fills += 1

        realized = 0.0
        if self.qty == 0.0 or (self.qty > 0) == (qty > 0):
            # open or add
            new_qty = self.qty + qty
            self.avg_entry = ((abs(self.qty) * self.avg_entry + abs(qty) * price)
                              / abs(new_qty))
            self.qty = new_qty
        else:
            # reducing / closing / flipping
            closing = min(abs(self.qty), abs(qty))
            direction = 1.0 if self.qty > 0 else -1.0
            realized = direction * (price - self.avg_entry) * closing
            if grid:
                self.gross_grid_pnl += realized
                self.num_cycles += 1
            new_qty = self.qty + qty
            if new_qty == 0.0:
                self.avg_entry = 0.0
            elif (new_qty > 0) != (self.qty > 0):
                self.avg_entry = price      # flipped: residual opens at fill px
            self.qty = new_qty

        notional_now = abs(self.qty) * price
        if self.qty > 0:
            self.max_long_notional = max(self.max_long_notional, notional_now)
        elif self.qty < 0:
            self.max_short_notional = max(self.max_short_notional, notional_now)
        return realized

    def apply_funding(self, mark_price: float, rate: float) -> float:
        """Binance convention: positive rate -> longs pay shorts.

        cashflow = -qty * mark * rate  (long pays when rate>0; short receives).
        """
        cf = -self.qty * mark_price * rate
        self.cash += cf
        self.funding_pnl += cf
        return cf
