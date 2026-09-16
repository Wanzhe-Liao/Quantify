"""Execution engine for the targeted dynamic ATR grid study."""
from __future__ import annotations

import math
import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        return lambda fn: fn

@njit(cache=True)
def _atr_value(row, idx):
    # row 9..13 = ATR10/30/60/120/EWMA30
    j = 9 + int(idx)
    return max(row[j], 1e-12)


@njit(cache=True)
def _spacing_pct(row, params, maker, tick):
    (atr_idx, atr_mult, floor_spacing, cap_spacing, levels, inventory, refresh,
     anchor, hysteresis_atr, inv_widen, inv_skew, trend_asym, vol_target,
     breakout_pause, breakout_stop, taper, no_new_tail, delay, stop, target,
     trail, direction, trend_size_bias, regime_eff, ratio_beta, ratio_risk,
     shock_pause) = params
    prev = row[5]
    atr = _atr_value(row, atr_idx)
    s = atr_mult * atr / prev
    s = max(s, floor_spacing, 2.0 * maker + 0.0002, 2.0 * tick / prev)
    if ratio_beta > 0:
        ratio = max(0.25, min(4.0, row[9] / max(row[12], 1e-12)))
        s *= ratio ** ratio_beta
    if cap_spacing > 0:
        s = min(s, cap_spacing)
    return s


@njit(cache=True)
def _execute(state, held, size, entry, lower, upper, directions, order,
             eligible, p, cap, fee, step, min_notional, volume_budget):
    st = state.copy()
    hh, qq, ee = held.copy(), size.copy(), entry.copy()
    remaining = volume_budget
    for j in order:
        if not eligible[j]:
            continue
        d = directions[j]
        price = (upper[j] if d > 0 else lower[j]) if hh[j] else (lower[j] if d > 0 else upper[j])
        q = qq[j]
        if q <= 0 or q > remaining + 1e-10:
            continue
        if not hh[j]:
            if (st[2] + q) * max(price, p) > cap + 1e-8:
                continue
            if price * q < min_notional - 1e-9:
                continue
            st[0] -= d * q * price
            st[1] += d * q
            st[2] += q
            hh[j], ee[j] = True, price
        else:
            st[0] += d * q * price
            st[1] -= d * q
            st[2] = max(0.0, st[2] - q)
            st[3] += d * q * (price - ee[j])
            st[9] += 1
            hh[j] = False
        cost = q * price * fee
        st[0] -= cost
        st[5] += cost
        st[8] += 1
        st[10] += q * price
        remaining -= q
    return st, hh, qq, ee


@njit(cache=True)
def _flatten(st, held, size, entry, directions, mark, taker, slip):
    inv_pnl = 0.0
    for j in range(len(held)):
        if held[j]:
            inv_pnl += directions[j] * size[j] * (mark - entry[j])
    notional = abs(st[1]) * mark
    st[0] += st[1] * mark - notional * (taker + slip / 10000.0)
    st[6] += notional * taker
    st[7] += notional * slip / 10000.0
    st[10] += notional
    st[1], st[2] = 0.0, 0.0
    held[:] = False
    return inv_pnl


@njit(cache=True)
def simulate(a, params, capital, tick, step, min_notional, maker, taker,
             slip_bps, participation, fill_drop, randoms, path_mode=0):
    """Matrix columns:
    O,H,L,C,V,prevC,EMA30,EMA120,VWAP60,ATR10,ATR30,ATR60,ATR120,
    ATR_EWMA30,RV30,ret10,ret30,efficiency30,funding_rate_times_mark.
    All features after V are lagged except the funding settlement cashflow.
    """
    (atr_idx, atr_mult, floor_spacing, cap_spacing, nl, inventory, refresh,
     anchor, hysteresis_atr, inv_widen, inv_skew, trend_asym, vol_target,
     breakout_pause, breakout_stop, taper, no_new_tail, delay, stop, target,
     trail, direction, trend_size_bias, regime_eff, ratio_beta, ratio_risk,
     shock_pause) = params
    n = int(nl)
    m = 2 * n
    total = len(a)
    start = int(delay)
    eq = np.full(total, capital)
    st = np.zeros(11)
    st[0] = capital
    details = np.zeros(9)
    if total < start + 3:
        details[1] = 6
        return st, eq, details

    ref = a[start, 5]
    center = ref
    initial_atr = _atr_value(a[start], atr_idx)
    base_cap = capital * inventory
    held = np.zeros(m, dtype=np.bool_)
    size = np.zeros(m)
    entry = np.zeros(m)
    lower = np.zeros(m)
    upper = np.zeros(m)
    live = np.zeros(m, dtype=np.bool_)
    directions = np.ones(m)
    directions[n:] = -1.0
    peak = capital
    inv_pnl = 0.0
    stop_signal = 0

    for i in range(start, total):
        o, h, lo, c, volume = a[i, :5]
        prev = a[i, 5]
        atr = _atr_value(a[i], atr_idx)
        atr_pct = atr / prev
        cashflow = -st[1] * a[i, 18]
        st[0] += cashflow
        st[4] += cashflow
        if a[i, 18] != 0:
            details[7] += 1

        scheduled = i == total - 1
        if stop_signal or scheduled:
            inv_pnl += _flatten(st, held, size, entry, directions, o, taker, slip_bps)
            eq[i:] = st[0]
            details[1] = stop_signal if stop_signal else 1
            break

        cap = base_cap
        if vol_target > 0:
            scale = vol_target / max(atr_pct, 1e-12)
            scale = min(1.0, max(0.20, scale))
            cap = base_cap * scale
        if ratio_risk > 0:
            ratio = max(0.25, min(4.0, a[i, 9] / max(a[i, 12], 1e-12)))
            cap *= min(1.0, max(0.20, (1.0 / ratio) ** ratio_risk))

        renew = i == start or (refresh > 0 and (i - start) % int(refresh) == 0)
        if renew:
            proposed = prev
            if anchor == 1:
                proposed = a[i, 6]
            elif anchor == 2:
                proposed = a[i, 7]
            elif anchor == 3:
                proposed = a[i, 8]
            if i == start or hysteresis_atr <= 0 or abs(proposed - center) >= hysteresis_atr * atr:
                center = proposed
            center -= inv_skew * atr * (st[1] * prev / max(base_cap, 1e-12))
            spacing = _spacing_pct(a[i], params, maker, tick)
            if inv_widen > 0:
                gross_ratio = min(2.0, st[2] * prev / max(base_cap, 1e-12))
                spacing *= 1.0 + inv_widen * gross_ratio
                if cap_spacing > 0:
                    spacing = min(spacing, cap_spacing)
            trend = 0.0
            if a[i, 16] > 1e-12:
                trend = 1.0
            elif a[i, 16] < -1e-12:
                trend = -1.0
            for j in range(m):
                if held[j]:
                    continue
                live[j] = False
                k = j + 1 if j < n else j - n + 1
                d = directions[j]
                asym = 1.0
                if trend_asym > 0 and trend != 0 and d != trend:
                    asym += trend_asym
                dist = spacing * asym
                near = center * (1.0 - d * (k - 1) * dist)
                far = center * (1.0 - d * k * dist)
                lower[j] = math.floor(min(near, far) / tick + 1e-9) * tick
                upper[j] = math.ceil(max(near, far) / tick - 1e-9) * tick
                q = cap / n / ref
                if trend_size_bias > 0 and trend != 0:
                    q *= max(0.20, 1.0 + trend_size_bias * d * trend)
                if taper:
                    q *= max(0.20, 1.0 - st[2] * prev / max(base_cap, 1e-12))
                size[j] = math.floor(q / step + 1e-9) * step
                if lower[j] <= 0 or upper[j] <= lower[j]:
                    size[j] = 0.0

        pause_open = False
        if breakout_pause > 0 and abs(prev - center) > breakout_pause * atr:
            pause_open = True
        if shock_pause > 0 and a[i, 9] / max(a[i, 12], 1e-12) >= shock_pause:
            pause_open = True
        if pause_open:
            details[8] += 1

        eligible = np.zeros(m, dtype=np.bool_)
        buy_price = np.full(m, -1e100)
        sell_price = np.full(m, 1e100)
        for j in range(m):
            d = directions[j]
            opening = not held[j]
            if opening and pause_open:
                continue
            if opening:
                allowed = 0.0
                lag_trend = 0.0
                if a[i, 16] > 1e-12:
                    lag_trend = 1.0
                elif a[i, 16] < -1e-12:
                    lag_trend = -1.0
                if abs(direction) == 1:
                    allowed = direction
                elif abs(direction) == 2:
                    allowed = lag_trend * (1.0 if direction > 0 else -1.0)
                elif direction == 3:
                    if a[i, 17] >= regime_eff and abs(a[i, 16]) >= 2.0 * atr_pct:
                        allowed = lag_trend
                if allowed != 0 and d != allowed:
                    continue
                if abs(direction) == 2 and allowed == 0:
                    continue
            if opening and no_new_tail > 0 and i >= start + (total - start) * (1.0 - no_new_tail):
                continue
            signed_side = d if opening else -d
            price = lower[j] if signed_side > 0 else upper[j]
            if not live[j]:
                if (signed_side > 0 and price >= prev) or (signed_side < 0 and price <= prev):
                    details[2] += 1
                    continue
                live[j] = True
            touched = lo < price if signed_side > 0 else h > price
            if touched and size[j] > 0 and volume > 0 and randoms[i, j] >= fill_drop:
                eligible[j] = True
                if signed_side > 0:
                    buy_price[j] = price
                else:
                    sell_price[j] = price

        buys = np.argsort(-buy_price)
        sells = np.argsort(sell_price)
        order_a = np.empty(m, dtype=np.int64)
        order_b = np.empty(m, dtype=np.int64)
        count = 0
        for j in buys:
            if buy_price[j] > -1e99:
                order_a[count] = j
                count += 1
        nb = count
        for j in sells:
            if sell_price[j] < 1e99:
                order_a[count] = j
                count += 1
        nc = count
        count = 0
        for j in sells:
            if sell_price[j] < 1e99:
                order_b[count] = j
                count += 1
        for j in buys:
            if buy_price[j] > -1e99:
                order_b[count] = j
                count += 1

        if nc:
            budget = volume * participation
            result_a = _execute(st, held, size, entry, lower, upper, directions,
                                order_a[:nc], eligible, o, cap, maker, step,
                                min_notional, budget)
            if nb and nc > nb:
                details[3] += 1
                result_b = _execute(st, held, size, entry, lower, upper, directions,
                                    order_b[:nc], eligible, o, cap, maker, step,
                                    min_notional, budget)
                va = result_a[0][0] + result_a[0][1] * c - abs(result_a[0][1]) * c * (taker + slip_bps/1e4)
                vb = result_b[0][0] + result_b[0][1] * c - abs(result_b[0][1]) * c * (taker + slip_bps/1e4)
                if path_mode == 2 or (path_mode == 0 and vb < va):
                    result_a = result_b
            live[held != result_a[1]] = False
            st, held, size, entry = result_a

        value = st[0] + st[1] * c
        eq[i] = value
        peak = max(peak, value)
        details[4] = max(details[4], st[2] * max(h, o))
        details[5] = max(details[5], abs(st[1]) * max(h, o))
        details[6] = max(details[6], abs(st[1]) * c)
        if stop > 0 and value <= capital * (1.0 - stop):
            stop_signal = 2
        elif trail > 0 and peak >= capital * (1.0 + target) and value <= peak - capital * trail:
            stop_signal = 3
        elif target > 0 and trail == 0 and value >= capital * (1.0 + target):
            stop_signal = 3
        elif breakout_stop > 0 and abs(c - ref) > breakout_stop * max(atr, initial_atr):
            stop_signal = 4

    details[0] = inv_pnl
    return st, eq, details


def run_array(a, candidate, *, capital=1000.0, tick=0.01,
              step=0.01, min_notional=5.0, maker=0.0002, taker=0.0005,
              slip_bps=5.0, participation=0.01, fill_drop=0.0, seed=7,
              path_mode=0):
    a = np.ascontiguousarray(a, dtype=float)
    if a.ndim != 2 or a.shape[1] != 19 or len(a) == 0 or not np.isfinite(a).all():
        raise ValueError("Expected nonempty finite (n,19) matrix")
    if min(capital, tick, step, participation) <= 0 or not 0 <= fill_drop <= 1:
        raise ValueError("Invalid account or execution parameters")
    if candidate.levels < 1 or not 0 < candidate.inventory <= 1:
        raise ValueError("Invalid grid risk budget")
    rng = np.random.default_rng(seed)
    randoms = rng.random((len(a), 2 * candidate.levels))
    st, eq, d = simulate(a, candidate.vector(), capital, tick, step, min_notional,
                         maker, taker, slip_bps, participation, fill_drop,
                         randoms, path_mode)
    net = st[0] - capital
    expect = st[3] + d[0] + st[4] - st[5] - st[6] - st[7]
    if abs(net - expect) > 1e-7 or abs(st[1]) > 1e-9:
        raise AssertionError(f"PnL reconciliation/flattening failed: {net=} {expect=} qty={st[1]}")
    curve = np.r_[capital, eq]
    out = {
        "candidate": candidate.name, "family": candidate.family,
        "net_pnl": net, "return_pct": net / capital,
        "cycle_pnl": st[3], "inventory_pnl": d[0], "funding_pnl": st[4],
        "maker_fees": st[5], "taker_fees": st[6], "slippage_cost": st[7],
        "fills": int(st[8]), "cycles": int(st[9]), "turnover": st[10],
        "exit_reason": int(d[1]), "ambiguous_bars": int(d[3]),
        "paused_bars": int(d[8]), "max_gross_marked_notional": d[4],
        "max_net_marked_notional": d[5],
        "max_drawdown": float(np.min(curve / np.maximum.accumulate(curve) - 1.0)),
        "reconciliation_error": float(net - expect),
    }
    return out, eq
