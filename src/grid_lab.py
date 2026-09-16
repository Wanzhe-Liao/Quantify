"""Causal paired-rung grid laboratory. Research only; no order routing.

Cash, inventory and funding are netted as a one-way futures account. Gross
virtual inventory is additionally capped. At most one fill per rung per
minute; newly armed exits cannot fill in that minute. Post-only orders that
would cross the last observed price are rejected and retried when passive.

Two OHLC side orderings are replayed from the same state. The adverse model
keeps the lower liquidation value; this is NOT a lower bound on all tick
paths. Stops observe completed-minute equity and execute at the NEXT open.
Scheduled end liquidation executes at the final minute's OPEN.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import numpy as np
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return lambda fn: fn


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    spacing: float = .002
    levels: int = 4
    inventory: float = .25
    geometry: int = 0
    anchor: int = 0
    refresh: int = 0
    volscale: int = 0
    direction: int = 0
    skew: float = 0.0
    invskew: float = 0.0
    taper: int = 0
    gate: int = 0
    stop: float = 0.0
    target: float = 0.0
    trail: float = 0.0
    breakout: float = 0.0
    no_new_tail: float = 0.0
    max_minutes: int = 0
    reset: int = 0
    delay: int = 0
    switch: int = 0

    def vector(self):
        return np.array([v for k, v in asdict(self).items()
                         if k not in ('name', 'family')], dtype=float)


def candidate_registry():
    """Frozen search space, enumerated before evaluating new results.

    anchor: 0 initial, 1 EMA30, 2 EMA120, 3 VWAP60.
    volscale: 1 ATR, 2 return standard deviation.
    direction: 1 long, -1 short, 2 momentum, -2 contrarian.
    gate: 1 drift, 2 efficiency, 3 minimum activity.
    """
    families = [
        ('fixed', {}), ('geometric', {'geometry': 1}),
        ('wide_outer', {'geometry': 2}), ('atr_static', {'volscale': 1}),
        ('rv_static', {'volscale': 2}), ('ema30_anchor', {'anchor': 1}),
        ('ema120_anchor', {'anchor': 2}), ('vwap_anchor', {'anchor': 3}),
        ('ema_refresh15', {'anchor': 1, 'refresh': 15}),
        ('ema_refresh60', {'anchor': 1, 'refresh': 60}),
        ('atr_refresh', {'anchor': 1, 'refresh': 15, 'volscale': 1}),
        ('inventory_skew', {'anchor': 1, 'refresh': 15, 'invskew': 2}),
        ('inventory_taper', {'taper': 1}), ('long_only', {'direction': 1}),
        ('short_only', {'direction': -1}), ('momentum_side', {'direction': 2}),
        ('contrarian_side', {'direction': -2}), ('momentum_skew', {'skew': .5}),
        ('contrarian_skew', {'skew': -.5}), ('drift_guard', {'gate': 1}),
        ('efficiency_guard', {'gate': 2}), ('activity_gate', {'gate': 3}),
        ('time_taper', {'no_new_tail': .33}), ('loss_stop', {'stop': .002}),
        ('take_profit', {'target': .001}),
        ('trailing_profit', {'target': .001, 'trail': .0005}),
        ('breakout_stop', {'breakout': 1.5}), ('time_exit120', {'max_minutes': 120}),
        ('reset120', {'reset': 120}),
        ('regime_switch', {'anchor': 1, 'refresh': 15, 'switch': 1}),
        ('delayed15', {'delay': 15}), ('delayed30', {'delay': 30}),
    ]
    return [Candidate(name=f'{fam}_s{int(s*10000):02d}_n{n}', family=fam,
                      spacing=s, levels=n, **kw)
            for fam, kw in families for s in (.001, .002, .004) for n in (2, 4)]


@njit(cache=True)
def _spacing(ref, atr, rv, base, mode, fee, tick):
    v = base
    if mode == 1:
        v = max(v, 1.5 * atr / ref)
    elif mode == 2:
        v = max(v, 2.0 * rv)
    return max(v, 2.0 * fee + .0002, 2.0 * tick / ref)


@njit(cache=True)
def _execute(state, held, size, entry, lower, upper, directions, order,
             eligible, p, cap, fee, step, min_notional, volume_budget):
    # state: cash, net qty, gross qty, cycles pnl, funding, maker fees,
    # taker fees, slippage, fill count, cycle count, turnover.
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
            st[2] = max(0., st[2] - q)
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
    inv_pnl = 0.
    for j in range(len(held)):
        if held[j]:
            inv_pnl += directions[j] * size[j] * (mark - entry[j])
    notional = abs(st[1]) * mark
    st[0] += st[1] * mark - notional * (taker + slip / 10000.)
    st[6] += notional * taker
    st[7] += notional * slip / 10000.
    st[10] += notional
    st[1], st[2] = 0., 0.
    held[:] = False
    return inv_pnl


@njit(cache=True)
def simulate(a, params, capital, tick, step, min_notional,
             maker, taker, slip_bps, participation, fill_drop, randoms, path_mode=0):
    """Columns: O,H,L,C,V,prevC,EMA30,EMA120,VWAP60,ATR30,RV30,
    ret30,efficiency30,settled_rate_times_mark. Features must be lagged.
    path_mode: 0 adverse, 1 buys then sells, 2 sells then buys.
    """
    (spacing, nl, inv, geometry, anchor, refresh, volscale, direction, skew,
     invskew, taper, gate, stop, target, trail, breakout, no_new_tail,
     max_minutes, reset, delay, switch) = params
    n, m, total = int(nl), int(nl) * 2, len(a)
    start = int(delay)
    eq = np.full(total, capital)
    st = np.zeros(11)
    st[0] = capital
    details = np.zeros(8)
    if total < start + 3:
        details[1] = 6
        return st, eq, details
    ref = a[start, 5]
    center = ref
    init_momentum = 1. if a[start, 11] > 0 else -1.
    if abs(a[start, 11]) < 1e-12:
        init_momentum = 0.
    s0 = _spacing(ref, a[start, 9], a[start, 10], spacing, volscale, maker, tick)
    if ((gate == 1 and abs(a[start, 11]) > n * s0) or
        (gate == 2 and a[start, 12] > .5) or
        (gate == 3 and a[start, 9] / ref < 2 * maker)):
        details[1] = 5
        return st, eq, details
    cap = capital * inv
    held = np.zeros(m, dtype=np.bool_)
    size = np.zeros(m)
    entry = np.zeros(m)
    lower = np.zeros(m)
    upper = np.zeros(m)
    live = np.zeros(m, dtype=np.bool_)
    directions = np.ones(m)
    directions[n:] = -1.
    peak = capital
    inventory_pnl = 0.
    stop_signal = 0
    for i in range(start, total):
        o, h, lo, c, volume = a[i, :5]
        previous_close = a[i, 5]
        cashflow = -st[1] * a[i, 13]
        st[0] += cashflow
        st[4] += cashflow
        if a[i, 13] != 0:
            details[7] += 1
        scheduled = (i == total - 1 or (max_minutes > 0 and i - start >= max_minutes))
        if stop_signal or scheduled:
            inventory_pnl += _flatten(st, held, size, entry, directions, o, taker, slip_bps)
            eq[i:] = st[0]
            details[1] = stop_signal if stop_signal else 1
            break
        is_reset = reset > 0 and i > start and (i - start) % int(reset) == 0
        if is_reset:
            inventory_pnl += _flatten(st, held, size, entry, directions, o, taker, slip_bps)
            live[:] = False
            # Flatten at the open; new quotes wait until the following minute.
            eq[i] = st[0]
            ref = previous_close
            continue
        renew = (i == start or (refresh > 0 and (i - start) % int(refresh) == 0)
                 or (reset > 0 and i > start and (i - start) % int(reset) == 1))
        if renew:
            center = ref
            if anchor == 1:
                center = a[i, 6]
            elif anchor == 2:
                center = a[i, 7]
            elif anchor == 3:
                center = a[i, 8]
            spacing_now = _spacing(previous_close, a[i, 9], a[i, 10], spacing, volscale, maker, tick)
            center -= invskew * spacing_now * previous_close * (st[1] * previous_close / cap)
            for j in range(m):
                if held[j]:
                    continue  # Never move an existing lot's exit target.
                live[j] = False
                k = j + 1 if j < n else j - n + 1
                d = directions[j]
                if geometry == 1:
                    near = center * (1. + spacing_now) ** (-d * (k - 1))
                    far = center * (1. + spacing_now) ** (-d * k)
                else:
                    exponent = 1.5 if geometry == 2 else 1.
                    near = center * (1. - d * (k - 1) ** exponent * spacing_now)
                    far = center * (1. - d * k ** exponent * spacing_now)
                lower[j] = math.floor(min(near, far) / tick + 1e-9) * tick
                upper[j] = math.ceil(max(near, far) / tick - 1e-9) * tick
                q = cap / n / ref * (1. + skew * d * init_momentum)
                if taper:
                    q *= max(.25, 1. - st[2] * previous_close / cap)
                size[j] = math.floor(q / step + 1e-9) * step
                if lower[j] <= 0 or upper[j] <= lower[j]:
                    size[j] = 0
        eligible = np.zeros(m, dtype=np.bool_)
        buy_price = np.full(m, -1e100)
        sell_price = np.full(m, 1e100)
        for j in range(m):
            d = directions[j]
            opening = not held[j]
            signed_side = d if opening else -d
            price = lower[j] if signed_side > 0 else upper[j]
            if opening:
                allowed = 0.
                if abs(direction) == 1:
                    allowed = direction
                elif abs(direction) == 2:
                    allowed = init_momentum * (1. if direction > 0 else -1.)
                    if allowed == 0:
                        continue
                if switch and a[i, 12] > .5:
                    allowed = 1. if a[i, 11] > 0 else -1.
                if allowed != 0 and d != allowed:
                    continue
                if no_new_tail > 0 and i >= start + (total - start) * (1 - no_new_tail):
                    continue
            if not live[j]:
                if (signed_side > 0 and price >= previous_close) or (signed_side < 0 and price <= previous_close):
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
                                order_a[:nc], eligible, o, cap, maker, step, min_notional, budget)
            if nb and nc > nb:
                details[3] += 1
                result_b = _execute(st, held, size, entry, lower, upper, directions,
                                    order_b[:nc], eligible, o, cap, maker, step, min_notional, budget)
                va = result_a[0][0] + result_a[0][1] * c - abs(result_a[0][1]) * c * (taker + slip_bps / 1e4)
                vb = result_b[0][0] + result_b[0][1] * c - abs(result_b[0][1]) * c * (taker + slip_bps / 1e4)
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
        if stop > 0 and value <= capital * (1 - stop):
            stop_signal = 2
        elif trail > 0 and peak >= capital * (1 + target) and value <= peak - capital * trail:
            stop_signal = 3
        elif target > 0 and trail == 0 and value >= capital * (1 + target):
            stop_signal = 3
        elif breakout > 0 and abs(c / ref - 1) > breakout * n * s0:
            stop_signal = 4
    details[0] = inventory_pnl
    return st, eq, details


def run_array(a, candidate, *, capital=1000., tick=.01, step=.01,
              min_notional=5., maker=.0002, taker=.0005, slip_bps=5.,
              participation=.01, fill_drop=0., seed=7, path_mode=0):
    a = np.ascontiguousarray(a, dtype=float)
    if a.ndim != 2 or a.shape[1] != 14 or len(a) == 0 or not np.isfinite(a).all():
        raise ValueError('Expected nonempty finite (n,14) matrix')
    if min(capital, tick, step, participation) <= 0 or not 0 <= fill_drop <= 1:
        raise ValueError('Invalid account or execution parameters')
    if candidate.levels < 1 or candidate.inventory <= 0 or candidate.inventory > 1:
        raise ValueError('Invalid grid risk budget')
    rng = np.random.default_rng(seed)
    randoms = rng.random((len(a), 2 * candidate.levels))
    st, eq, d = simulate(a, candidate.vector(), capital, tick, step, min_notional,
                        maker, taker, slip_bps, participation, fill_drop, randoms, path_mode)
    net = st[0] - capital
    expect = st[3] + d[0] + st[4] - st[5] - st[6] - st[7]
    if abs(net - expect) > 1e-7 or abs(st[1]) > 1e-9:
        raise AssertionError(f'PnL reconciliation/flattening failed: {net=} {expect=} qty={st[1]}')
    curve = np.r_[capital, eq]
    out = {'candidate': candidate.name, 'family': candidate.family,
           'net_pnl': net, 'return_pct': net / capital, 'cycle_pnl': st[3],
           'inventory_pnl': d[0], 'funding_pnl': st[4], 'maker_fees': st[5],
           'taker_fees': st[6], 'slippage_cost': st[7], 'fills': int(st[8]),
           'cycles': int(st[9]), 'turnover': st[10], 'exit_reason': int(d[1]),
           'ambiguous_bars': int(d[3]), 'max_gross_marked_notional': d[4],
           'max_net_marked_notional': d[5],
           'max_drawdown': float(np.min(curve / np.maximum.accumulate(curve) - 1)),
           'reconciliation_error': float(net - expect)}
    return out, eq
