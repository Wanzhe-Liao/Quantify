"""ZHONGJI-focused dynamic ATR grid research engine.

Research only.  All adaptive signals use lagged features.  The engine keeps the
paired-rung accounting/execution conventions used by ``grid_lab`` while adding
ATR-horizon selection, ATR-multiple spacing, dynamic risk sizing, inventory
widening/skew, trend-asymmetric spacing, hysteretic re-centering and breakout
protection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        return lambda fn: fn


@dataclass(frozen=True)
class ATRCandidate:
    name: str
    family: str
    atr_idx: int = 1             # 0/1/2/3/4 = ATR10/30/60/120/EWMA30
    atr_mult: float = 1.0
    floor_spacing: float = 0.0006
    cap_spacing: float = 0.012
    levels: int = 2
    inventory: float = 0.25
    refresh: int = 15
    anchor: int = 1              # prev close / EMA30 / EMA120 / VWAP60
    hysteresis_atr: float = 0.0
    inv_widen: float = 0.0
    inv_skew: float = 0.0
    trend_asym: float = 0.0
    vol_target: float = 0.0      # target ATR/price; 0 = fixed inventory budget
    breakout_pause: float = 0.0  # |lagged price-center| / ATR; pause openings
    breakout_stop: float = 0.0   # |close-start ref| / current ATR; stop next open
    taper: int = 0
    no_new_tail: float = 0.0
    delay: int = 0
    stop: float = 0.0
    target: float = 0.0
    trail: float = 0.0
    direction: int = 0          # 0 symmetric, +/-1 fixed side, +/-2 trend/contrarian, 3 regime trend
    trend_size_bias: float = 0.0
    regime_eff: float = 0.5
    ratio_beta: float = 0.0       # spacing multiplier from ATR10/ATR120 term structure
    ratio_risk: float = 0.0       # inverse inventory scaling from ATR10/ATR120
    shock_pause: float = 0.0      # pause openings when ATR10/ATR120 exceeds threshold

    def vector(self) -> np.ndarray:
        return np.array([v for k, v in asdict(self).items()
                         if k not in ("name", "family")], dtype=float)


def candidate_registry() -> list[ATRCandidate]:
    """Bounded ATR-focused registry; definitions are outcome-independent.

    The registry intentionally tests distinct mechanisms rather than a dense
    cartesian product.  Names encode the key parameters for reproducibility.
    """
    out: list[ATRCandidate] = []
    atr_names = {0: "a10", 1: "a30", 2: "a60", 3: "a120", 4: "aewm"}

    # 1) Multi-horizon ATR spacing, no re-centering.
    for ai in range(5):
        for mult in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
            for n in (2, 4):
                out.append(ATRCandidate(
                    f"atr_static_{atr_names[ai]}_m{mult:g}_n{n}", "atr_static_v2",
                    atr_idx=ai, atr_mult=mult, levels=n, refresh=0, anchor=0))

    # 2) Dynamic re-centering.  ATR30/60/120 and EMA/VWAP anchors.
    for ai in (1, 2, 3, 4):
        for mult in (0.75, 1.0, 1.5, 2.0):
            for n in (2, 4):
                for refresh in (5, 15, 30, 60):
                    for anchor in (1, 3):
                        out.append(ATRCandidate(
                            f"atr_refresh_{atr_names[ai]}_m{mult:g}_n{n}_r{refresh}_c{anchor}",
                            "atr_refresh_v2", atr_idx=ai, atr_mult=mult, levels=n,
                            refresh=refresh, anchor=anchor))

    # 3) Hysteretic center updates: avoid chasing noise.
    for ai in (1, 2):
        for mult in (1.0, 1.5):
            for n in (2, 4):
                for hys in (0.5, 1.0, 2.0):
                    out.append(ATRCandidate(
                        f"atr_hysteresis_{atr_names[ai]}_m{mult:g}_n{n}_h{hys:g}",
                        "atr_hysteresis", atr_idx=ai, atr_mult=mult, levels=n,
                        refresh=5, anchor=1, hysteresis_atr=hys))

    # 4) Inventory-aware spacing / center skew.
    for ai in (1, 2):
        for mult in (1.0, 1.5):
            for n in (2, 4):
                for widen in (1.0, 2.0, 4.0):
                    for skew in (0.0, 1.0):
                        out.append(ATRCandidate(
                            f"atr_inv_{atr_names[ai]}_m{mult:g}_n{n}_w{widen:g}_k{skew:g}",
                            "atr_inventory", atr_idx=ai, atr_mult=mult, levels=n,
                            refresh=15, anchor=1, inv_widen=widen, inv_skew=skew,
                            taper=1))

    # 5) Volatility-targeted inventory cap.  Target is ATR/price.
    for ai in (1, 2):
        for mult in (1.0, 1.5):
            for n in (2, 4):
                for vt in (0.0005, 0.0010, 0.0015, 0.0025):
                    out.append(ATRCandidate(
                        f"atr_voltarget_{atr_names[ai]}_m{mult:g}_n{n}_v{int(vt*1e4):02d}",
                        "atr_vol_target", atr_idx=ai, atr_mult=mult, levels=n,
                        refresh=15, anchor=1, vol_target=vt))

    # 6) Trend-asymmetric grids: widen the side opposing lagged 30m trend.
    for ai in (1, 2):
        for mult in (1.0, 1.5):
            for n in (2, 4):
                for asym in (0.5, 1.0, 2.0):
                    out.append(ATRCandidate(
                        f"atr_asym_{atr_names[ai]}_m{mult:g}_n{n}_x{asym:g}",
                        "atr_trend_asym", atr_idx=ai, atr_mult=mult, levels=n,
                        refresh=15, anchor=1, trend_asym=asym))

    # 7) Breakout-aware entry pause and hard breakout stop.
    for ai in (1, 2):
        for mult in (1.0, 1.5):
            for n in (2, 4):
                for pause in (1.5, 2.5, 4.0):
                    out.append(ATRCandidate(
                        f"atr_pause_{atr_names[ai]}_m{mult:g}_n{n}_b{pause:g}",
                        "atr_breakout_pause", atr_idx=ai, atr_mult=mult, levels=n,
                        refresh=15, anchor=1, breakout_pause=pause))
                for stop in (3.0, 5.0, 8.0):
                    out.append(ATRCandidate(
                        f"atr_bstop_{atr_names[ai]}_m{mult:g}_n{n}_b{stop:g}",
                        "atr_breakout_stop", atr_idx=ai, atr_mult=mult, levels=n,
                        refresh=15, anchor=1, breakout_stop=stop))

    # 8) Combined risk-control candidates, deliberately small and interpretable.
    for ai in (1, 2):
        for mult in (1.0, 1.5, 2.0):
            for n in (2, 4):
                for vt in (0.0010, 0.0015):
                    out.append(ATRCandidate(
                        f"atr_combo_{atr_names[ai]}_m{mult:g}_n{n}_v{int(vt*1e4):02d}",
                        "atr_combo", atr_idx=ai, atr_mult=mult, levels=n,
                        refresh=15, anchor=1, hysteresis_atr=1.0,
                        inv_widen=2.0, inv_skew=1.0, trend_asym=1.0,
                        vol_target=vt, breakout_pause=2.5, taper=1,
                        no_new_tail=0.25))

    # 9) Same combined logic with explicit equity stops / profit protection.
    for stop in (0.0015, 0.0025, 0.0040):
        for target, trail in ((0.0010, 0.0), (0.0015, 0.0005)):
            out.append(ATRCandidate(
                f"atr_combo_exit_s{int(stop*1e4):02d}_t{int(target*1e4):02d}_r{int(trail*1e4):02d}",
                "atr_combo_exit", atr_idx=2, atr_mult=1.5, levels=4,
                refresh=15, anchor=1, hysteresis_atr=1.0,
                inv_widen=2.0, inv_skew=1.0, trend_asym=1.0,
                vol_target=0.0015, breakout_pause=2.5, taper=1,
                no_new_tail=0.25, stop=stop, target=target, trail=trail))

    # 10) Trailing price-centered ATR grids: follow the market instead of EMA/VWAP.
    for ai in (1, 2):
        for mult in (0.75, 1.0, 1.5, 2.0):
            for n in (2, 4):
                for refresh in (5, 15, 30):
                    out.append(ATRCandidate(
                        f"atr_price_refresh_{atr_names[ai]}_m{mult:g}_n{n}_r{refresh}",
                        "atr_price_refresh", atr_idx=ai, atr_mult=mult, levels=n,
                        refresh=refresh, anchor=0))

    # 11) Directional ATR grids from lagged 30-minute trend.
    for ai in (1, 2):
        for mult in (1.0, 1.5, 2.0):
            for n in (2, 4):
                for refresh in (5, 15):
                    for anchor in (0, 1):
                        out.append(ATRCandidate(
                            f"atr_trendside_{atr_names[ai]}_m{mult:g}_n{n}_r{refresh}_c{anchor}",
                            "atr_trend_side", atr_idx=ai, atr_mult=mult, levels=n,
                            refresh=refresh, anchor=anchor, direction=2))
                        out.append(ATRCandidate(
                            f"atr_contraside_{atr_names[ai]}_m{mult:g}_n{n}_r{refresh}_c{anchor}",
                            "atr_contrarian_side", atr_idx=ai, atr_mult=mult, levels=n,
                            refresh=refresh, anchor=anchor, direction=-2))

    # 12) Symmetric grid with trend-biased order size rather than hard side filtering.
    for ai in (1, 2):
        for mult in (1.0, 1.5):
            for n in (2, 4):
                for bias in (0.25, 0.5, 0.75):
                    out.append(ATRCandidate(
                        f"atr_sizebias_{atr_names[ai]}_m{mult:g}_n{n}_z{bias:g}",
                        "atr_trend_size", atr_idx=ai, atr_mult=mult, levels=n,
                        refresh=10, anchor=0, trend_size_bias=bias))

    # 13) Regime switch: trend-side only when lagged move is large and efficient.
    for ai in (1, 2):
        for mult in (1.0, 1.5, 2.0):
            for n in (2, 4):
                for eff in (0.3, 0.5, 0.7):
                    out.append(ATRCandidate(
                        f"atr_regime_{atr_names[ai]}_m{mult:g}_n{n}_e{eff:g}",
                        "atr_regime_side", atr_idx=ai, atr_mult=mult, levels=n,
                        refresh=10, anchor=0, direction=3, regime_eff=eff,
                        inv_widen=1.0, taper=1))

    # 14) Fixed long/short ATR grids are diagnostics for directional failure modes.
    for side, tag in ((1, "long"), (-1, "short")):
        for ai in (1, 2):
            for mult in (1.0, 1.5, 2.0):
                for n in (2, 4):
                    out.append(ATRCandidate(
                        f"atr_{tag}_{atr_names[ai]}_m{mult:g}_n{n}",
                        f"atr_{tag}_only", atr_idx=ai, atr_mult=mult, levels=n,
                        refresh=10, anchor=0, direction=side))

    # 15) ATR term-structure adaptation: react when short ATR rises over long ATR.
    for ai in (1, 2):
        for mult in (1.0, 1.5):
            for n in (2, 4):
                for refresh in (5, 15):
                    for beta in (0.5, 1.0, 1.5):
                        for risk in (0.0, 1.0):
                            out.append(ATRCandidate(
                                f"atr_ratio_{atr_names[ai]}_m{mult:g}_n{n}_r{refresh}_b{beta:g}_k{risk:g}",
                                "atr_ratio_adaptive", atr_idx=ai, atr_mult=mult, levels=n,
                                refresh=refresh, anchor=0, ratio_beta=beta, ratio_risk=risk,
                                inv_widen=1.0 if risk else 0.0, taper=1 if risk else 0))

    # 16) Shock pause: keep exits live but do not add inventory during ATR spikes.
    for ai in (1, 2):
        for mult in (1.0, 1.5):
            for n in (2, 4):
                for shock in (1.5, 2.0, 3.0):
                    out.append(ATRCandidate(
                        f"atr_shockpause_{atr_names[ai]}_m{mult:g}_n{n}_q{shock:g}",
                        "atr_shock_pause", atr_idx=ai, atr_mult=mult, levels=n,
                        refresh=5, anchor=0, ratio_beta=1.0, ratio_risk=1.0,
                        shock_pause=shock, taper=1))

    names = [c.name for c in out]
    if len(names) != len(set(names)):
        raise AssertionError("duplicate ATR candidate names")
    return out


from .atr_dynamic_engine import run_array
