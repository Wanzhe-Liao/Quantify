"""Backtest engine: one grid session per closed-market window.

Each session starts flat, rests a symmetric grid around the window-start price,
settles funding at real Binance funding timestamps, and force-liquidates all
inventory (taker fee + exit slippage) on the last bar before the underlying
market reopens. No lookahead: grid center/range use only window-start price.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .accounting import Ledger
from .execution import process_bar
from .grid import build_grid, cancel_all


@dataclass
class SessionResult:
    record: dict
    equity_curve: pd.Series     # equity at each bar close, indexed by open_time


def _slice_bars(klines: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return klines[(klines.open_time >= start) & (klines.open_time < end)]


def run_session(klines: pd.DataFrame, funding: pd.DataFrame,
                start: pd.Timestamp, end: pd.Timestamp, *,
                symbol: str, initial_capital: float, spacing: float,
                range_pct: float, levels_each_side: int, max_inventory_pct: float,
                maker_fee: float, taker_fee: float, exit_slippage_bps: float,
                tick_size: float, step_size: float, mode: str,
                drop_prob: float, seed: int, label: str,
                window_type: str = "session", complete: bool = True) -> SessionResult | None:
    """Run one grid session over [start, end). Returns None if no bars."""
    bars = _slice_bars(klines, start, end)
    if len(bars) == 0:
        return None

    led = Ledger(cash=initial_capital)
    ref_price = float(bars.iloc[0].open)           # first observable price
    cap_notional = initial_capital * max_inventory_pct
    qty_per_order = cap_notional / levels_each_side / ref_price
    qty_per_order = np.floor(qty_per_order / step_size) * step_size
    rng = np.random.default_rng(seed)

    book = build_grid(center=ref_price, ref_price=ref_price, spacing=spacing,
                      range_pct=range_pct, levels_each_side=levels_each_side,
                      qty_per_order=qty_per_order, tick_size=tick_size)

    # funding events inside the session, sorted
    fund = funding[(funding.funding_time >= start) & (funding.funding_time < end)] \
        if len(funding) else funding
    fund_iter = iter(fund.itertuples(index=False)) if len(fund) else iter(())
    next_fund = next(fund_iter, None)

    equity_idx, equity_val = [], []
    realized_vol_ret = []
    prev_close = None

    for bar in bars.itertuples(index=False):
        # funding settles at its real timestamp, before/at this bar's close
        while next_fund is not None and next_fund.funding_time <= bar.close_time:
            led.apply_funding(next_fund.mark_price, next_fund.funding_rate)
            next_fund = next(fund_iter, None)
        if book.levels:
            process_bar(bar, book, led, cap_notional, maker_fee, mode, drop_prob, rng)
        if prev_close is not None:
            realized_vol_ret.append(bar.close / prev_close - 1.0)
        prev_close = bar.close
        equity_idx.append(bar.open_time)
        equity_val.append(led.equity(bar.close))

    # ---- forced liquidation at last bar close, taker fee + slippage ----
    mark = float(bars.iloc[-1].close)
    inv_liq_pnl_at_mark = 0.0
    slippage_cost = 0.0
    if led.qty != 0.0:
        direction = 1.0 if led.qty > 0 else -1.0
        inv_liq_pnl_at_mark = direction * (mark - led.avg_entry) * abs(led.qty)
        liq_qty = -led.qty                            # flatten
        slippage_cost = abs(liq_qty) * mark * (exit_slippage_bps / 1e4)
        # fill at slipped price for cash; attribute pnl at mark + separate slippage term
        led.cash -= liq_qty * mark                    # proceeds at mark
        led.cash -= abs(liq_qty) * mark * taker_fee   # taker fee
        led.cash -= slippage_cost                     # adverse exit slippage
        led.taker_fees += abs(liq_qty) * mark * taker_fee
        led.turnover += abs(liq_qty) * mark
        led.qty = 0.0
    cancel_all(book)

    final_equity = led.cash                           # qty == 0 now
    net_pnl = final_equity - initial_capital
    eq = pd.Series(equity_val, index=pd.Index(equity_idx, name="open_time"))
    running_max = eq.cummax()
    max_dd = float((eq / running_max - 1.0).min()) if len(eq) else 0.0

    record = {
        "symbol": symbol, "label": label, "window_type": window_type,
        "complete": complete,
        "window_start": start, "window_end": end,
        "mode": mode, "spacing": spacing, "range_pct": range_pct,
        "levels": levels_each_side, "max_inventory_pct": max_inventory_pct,
        "maker_fee": maker_fee, "taker_fee": taker_fee,
        "exit_slippage_bps": exit_slippage_bps, "drop_prob": drop_prob,
        "start_price": ref_price, "end_price": mark,
        "window_return": mark / ref_price - 1.0,
        "realized_vol": float(np.std(realized_vol_ret)) if realized_vol_ret else 0.0,
        "n_bars": len(bars),
        "num_fills": led.num_fills, "num_cycles": led.num_cycles,
        "gross_grid_pnl": led.gross_grid_pnl,
        "inventory_liquidation_pnl": inv_liq_pnl_at_mark,
        "funding_pnl": led.funding_pnl,
        "maker_fees": led.maker_fees, "taker_fees": led.taker_fees,
        "slippage_cost": slippage_cost,
        "trading_pnl": led.gross_grid_pnl + inv_liq_pnl_at_mark,
        "net_pnl": net_pnl, "return_pct": net_pnl / initial_capital,
        "final_equity": final_equity,
        "max_drawdown": max_dd,
        "max_long_inventory": led.max_long_notional,
        "max_short_inventory": led.max_short_notional,
        "turnover": led.turnover / initial_capital,
    }
    return SessionResult(record=record, equity_curve=eq)
