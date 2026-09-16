"""Underlying-market session calendar -> closed_market_window segmentation.

A closed-market window is the gap between two consecutive underlying trading
sessions: [session_close_local -> next_session_open_local], expressed in UTC.

Window types:
  lunch_break        - gap between two sessions on the same local trading day
  weekday_overnight  - close of day D -> open of the next calendar day
  weekend            - gap spanning only Sat/Sun non-trading days
  holiday            - gap spanning >=1 non-trading weekday that is a holiday
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd


@dataclass
class MarketSpec:
    name: str
    timezone: str
    sessions: list[tuple[time, time]]          # local (open, close) per trading day
    holidays: set[date]
    trading_weekdays: tuple[int, ...] = (0, 1, 2, 3, 4)  # Mon-Fri


def spec_from_config(cfg: dict) -> MarketSpec:
    tz = cfg["timezone"]
    sessions = []
    for o, c in cfg["sessions"]:
        oh, om = map(int, o.split(":"))
        ch, cm = map(int, c.split(":"))
        sessions.append((time(oh, om), time(ch, cm)))
    holidays = {date.fromisoformat(h) for h in cfg.get("holidays", [])}
    return MarketSpec(name=cfg["market"], timezone=tz, sessions=sessions, holidays=holidays)


def _open_intervals(spec: MarketSpec, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp, date]]:
    """All trading-session open intervals intersecting [start_utc, end_utc], as
    (open_utc, close_utc, local_date)."""
    tz = ZoneInfo(spec.timezone)
    first = start_utc.tz_convert(tz).date() - timedelta(days=1)
    last = end_utc.tz_convert(tz).date() + timedelta(days=1)
    out = []
    d = first
    while d <= last:
        if d.weekday() in spec.trading_weekdays and d not in spec.holidays:
            for so, sc in spec.sessions:
                o = pd.Timestamp(datetime.combine(d, so), tz=tz).tz_convert("UTC")
                c = pd.Timestamp(datetime.combine(d, sc), tz=tz).tz_convert("UTC")
                if c > start_utc and o < end_utc:
                    out.append((o, c, d))
        d += timedelta(days=1)
    out.sort(key=lambda x: x[0])
    return out


def _is_trading_day(spec: MarketSpec, d: date) -> bool:
    return d.weekday() in spec.trading_weekdays and d not in spec.holidays


def classify_gap(spec: MarketSpec, close_local_date: date, next_open_local_date: date) -> str:
    if next_open_local_date == close_local_date:
        return "lunch_break"
    gap_days = []
    d = close_local_date + timedelta(days=1)
    while d < next_open_local_date:
        gap_days.append(d)
        d += timedelta(days=1)
    if not gap_days:
        return "weekday_overnight"
    if any(d.weekday() in spec.trading_weekdays for d in gap_days):
        return "holiday"
    return "weekend"


def closed_market_windows(spec: MarketSpec, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.DataFrame:
    """Closed windows fully or partially inside [start_utc, end_utc].

    Returns DataFrame[window_start, window_end, window_type, market, tz,
                     close_local_date, next_open_local_date, complete].
    """
    intervals = _open_intervals(spec, start_utc, end_utc)
    rows = []
    for (o1, c1, d1), (o2, c2, d2) in zip(intervals, intervals[1:]):
        ws, we = c1, o2                      # close -> next open
        complete = True
        if we <= start_utc or ws >= end_utc:
            continue
        ws_c, we_c = max(ws, start_utc), min(we, end_utc)
        if ws_c >= we_c:
            continue
        if ws_c > ws or we_c < we:
            complete = False                 # window clipped by data range
        rows.append({
            "window_start": ws_c, "window_end": we_c,
            "window_type": classify_gap(spec, d1, d2),
            "market": spec.name,
            "close_local_date": str(d1), "next_open_local_date": str(d2),
            "complete": complete,
        })
    # trailing window: last close -> data end (market still closed, incomplete)
    if intervals:
        o_last, c_last, d_last = intervals[-1]
        if c_last < end_utc and c_last >= start_utc:
            rows.append({
                "window_start": c_last, "window_end": end_utc,
                "window_type": "unclosed_tail", "market": spec.name,
                "close_local_date": str(d_last), "next_open_local_date": None,
                "complete": False,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("window_start").reset_index(drop=True)
    return df
