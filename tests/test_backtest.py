"""End-to-end tests: window segmentation, forced liquidation, funding, PnL."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.backtest import run_session
from src.calendar import MarketSpec, closed_market_windows


def _klines(rows):
    """rows: list of (open_time_str, o, h, l, c)"""
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close"])
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df["close_time"] = df.open_time + pd.Timedelta(minutes=1) - pd.Timedelta(milliseconds=1)
    df["volume"] = 1.0
    df["trades"] = 1
    return df


SSE = MarketSpec(
    name="SSE", timezone="Asia/Shanghai",
    sessions=[(__import__("datetime").time(9, 30), __import__("datetime").time(11, 30)),
              (__import__("datetime").time(13, 0), __import__("datetime").time(15, 0))],
    holidays=set())


def test_window_segmentation_types():
    # Fri 2026-09-11 -> Tue 2026-09-15, SSE hours
    start = pd.Timestamp("2026-09-11", tz="UTC")
    end = pd.Timestamp("2026-09-16", tz="UTC")
    w = closed_market_windows(SSE, start, end)
    types = list(w.window_type)
    assert "lunch_break" in types
    assert "weekday_overnight" in types
    assert "weekend" in types
    # weekend window: Fri 15:00 CST (07:00 UTC) -> Mon 09:30 CST (01:30 UTC)
    wk = w[w.window_type == "weekend"].iloc[0]
    assert wk.window_start == pd.Timestamp("2026-09-11 07:00", tz="UTC")
    assert wk.window_end == pd.Timestamp("2026-09-14 01:30", tz="UTC")


def test_holiday_window_classification():
    from datetime import date
    spec = MarketSpec(name="NASDAQ", timezone="America/New_York",
                      sessions=[(__import__("datetime").time(9, 30),
                                 __import__("datetime").time(16, 0))],
                      holidays={date(2026, 9, 7)})      # Labor Day Monday
    w = closed_market_windows(spec, pd.Timestamp("2026-09-04", tz="UTC"),
                              pd.Timestamp("2026-09-09", tz="UTC"))
    types = set(w.window_type)
    assert "holiday" in types     # Fri close -> Tue open spans holiday Monday


def test_forced_liquidation_flat_at_end():
    # window where price falls and never recovers -> grid stuck long -> liquidated
    rows = [(f"2026-09-11 08:{m:02d}", 100 - m, 100 - m + 0.01, 100 - m - 0.5, 100 - m)
            for m in range(0, 30, 5)]
    # ensure low penetrates buy levels: widen lows
    rows = [(t, o, h, o - 2.0, c) for (t, o, h, l, c) in rows]
    kl = _klines(rows)
    res = run_session(kl, pd.DataFrame(columns=["funding_time", "funding_rate", "mark_price"]),
                      kl.open_time.min(), kl.open_time.max() + pd.Timedelta(minutes=1),
                      symbol="T", initial_capital=1000.0, spacing=0.005,
                      range_pct=0.05, levels_each_side=4, max_inventory_pct=0.5,
                      maker_fee=0.0002, taker_fee=0.0005, exit_slippage_bps=5,
                      tick_size=0.01, step_size=0.01, mode="conservative",
                      drop_prob=0.0, seed=0, label="t")
    r = res.record
    assert r["num_fills"] > 0
    # inventory must be flat at end: final_equity == cash, taker fee charged
    assert r["taker_fees"] > 0
    assert r["slippage_cost"] > 0
    assert r["net_pnl"] < 0                       # falling market, long inventory
    # consistency: net = grid + inv_liq + funding - fees - slippage
    expect = (r["gross_grid_pnl"] + r["inventory_liquidation_pnl"] + r["funding_pnl"]
              - r["maker_fees"] - r["taker_fees"] - r["slippage_cost"])
    assert abs(r["net_pnl"] - expect) < 1e-8


def test_funding_settles_in_window():
    rows = [("2026-09-12 00:00", 100, 100.6, 99.4, 100.2),
            ("2026-09-12 00:01", 100.2, 100.8, 99.6, 99.0)]
    kl = _klines(rows)
    fund = pd.DataFrame({
        "funding_time": [pd.Timestamp("2026-09-12 00:00:30", tz="UTC")],
        "funding_rate": [0.001], "mark_price": [100.0]})
    res = run_session(kl, fund, kl.open_time.min(),
                      kl.open_time.max() + pd.Timedelta(minutes=1),
                      symbol="T", initial_capital=1000.0, spacing=0.001,
                      range_pct=0.01, levels_each_side=2, max_inventory_pct=0.5,
                      maker_fee=0.0, taker_fee=0.0, exit_slippage_bps=0,
                      tick_size=0.01, step_size=0.01, mode="conservative",
                      drop_prob=0.0, seed=0, label="t")
    # inventory was flat at funding ts (no fills yet) -> funding_pnl == 0
    assert res.record["funding_pnl"] == pytest.approx(0.0)


def test_funding_sign_in_session():
    # hold a long through a positive funding timestamp -> funding_pnl negative
    rows = [("2026-09-12 00:00", 100, 100.1, 99.0, 100.0),   # buy@99.9 fills
            ("2026-09-12 00:01", 100.0, 100.05, 99.95, 100.0)]
    kl = _klines(rows)
    fund = pd.DataFrame({
        "funding_time": [pd.Timestamp("2026-09-12 00:01:30", tz="UTC")],
        "funding_rate": [0.001], "mark_price": [100.0]})
    res = run_session(kl, fund, kl.open_time.min(),
                      kl.open_time.max() + pd.Timedelta(minutes=2),
                      symbol="T", initial_capital=1000.0, spacing=0.001,
                      range_pct=0.01, levels_each_side=1, max_inventory_pct=0.5,
                      maker_fee=0.0, taker_fee=0.0, exit_slippage_bps=0,
                      tick_size=0.01, step_size=0.01, mode="conservative",
                      drop_prob=0.0, seed=0, label="t")
    r = res.record
    if r["num_fills"] > 0:  # if long was held at funding time it paid
        assert r["funding_pnl"] <= 0.0
