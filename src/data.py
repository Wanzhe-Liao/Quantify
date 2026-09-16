"""Binance USD-M futures data download + local parquet cache.

Covers TRADIFI_PERPETUAL (stock-linked) contracts which live on the standard
fapi endpoints. If the symbol is missing from fapi the downloader fails loudly
and (optionally) writes a clearly-marked synthetic dataset instead.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

FAPI = "https://fapi.binance.com"
KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
NUM_COLS = ["open", "high", "low", "close", "volume", "quote_volume"]


class DataUnavailable(Exception):
    pass


def _get(path: str, params: dict, retries: int = 4) -> list | dict:
    url = FAPI + path
    for attempt in range(retries):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (418, 429):
            time.sleep(2 ** attempt)
            continue
        try:
            msg = r.json()
        except Exception:
            msg = r.text[:200]
        raise DataUnavailable(f"GET {path} {params} -> {r.status_code}: {msg}")
    raise DataUnavailable(f"GET {path} failed after {retries} retries (rate limited)")


def get_symbol_meta(symbol: str) -> dict:
    """Tick size / step size / contract type for one symbol."""
    info = _get("/fapi/v1/exchangeInfo", {})
    for s in info["symbols"]:
        if s["symbol"] == symbol:
            tick = step = None
            for f in s["filters"]:
                if f["filterType"] == "PRICE_FILTER":
                    tick = float(f["tickSize"])
                if f["filterType"] == "LOT_SIZE":
                    step = float(f["stepSize"])
            return {
                "symbol": symbol,
                "contractType": s["contractType"],
                "status": s["status"],
                "underlyingType": s.get("underlyingType"),
                "tick_size": tick,
                "step_size": step,
                "onboard_ms": s.get("onboardDate"),
            }
    raise DataUnavailable(
        f"{symbol} not found on fapi exchangeInfo "
        f"({len(info['symbols'])} symbols scanned). "
        "Symbol may live on a different Binance endpoint."
    )


def fetch_klines(symbol: str, start_ms: int, end_ms: int, interval: str = "1m") -> pd.DataFrame:
    """Paginated 1m klines -> DataFrame indexed by open_time (UTC)."""
    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        batch = _get("/fapi/v1/klines", {
            "symbol": symbol, "interval": interval,
            "startTime": cursor, "endTime": end_ms, "limit": 1500,
        })
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + 1
        if len(batch) < 1500:
            break
    if not rows:
        raise DataUnavailable(f"no klines returned for {symbol} {start_ms}..{end_ms}")
    df = pd.DataFrame(rows, columns=KLINE_COLS)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in NUM_COLS:
        df[c] = df[c].astype(float)
    df["trades"] = df["trades"].astype(int)
    return df[["open_time", "open", "high", "low", "close", "volume",
               "close_time", "trades"]].drop_duplicates("open_time").sort_values("open_time")


def fetch_funding(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Historical funding rates -> DataFrame[funding_time, funding_rate, mark_price]."""
    rows: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        batch = _get("/fapi/v1/fundingRate", {
            "symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000,
        })
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1]["fundingTime"]) + 1
        if len(batch) < 1000:
            break
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["funding_time", "funding_rate", "mark_price"])
    df["funding_time"] = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    df["mark_price"] = df["markPrice"].astype(float)
    return df[["funding_time", "funding_rate", "mark_price"]].drop_duplicates("funding_time")


def make_synthetic(symbol: str, start_ms: int, end_ms: int, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Geometric-brownian 1m bars + zero funding. Clearly marked synthetic."""
    rng = np.random.default_rng(seed)
    n = (end_ms - start_ms) // 60_000
    rets = rng.normal(0, 0.0008, n)  # ~2%/day vol
    price = 100.0 * np.exp(np.cumsum(rets))
    ot = pd.to_datetime(np.arange(start_ms, start_ms + n * 60_000, 60_000), unit="ms", utc=True)
    spread = np.abs(rng.normal(0, 0.0004, n))
    df = pd.DataFrame({
        "open_time": ot,
        "open": np.roll(price, 1),
        "high": price * (1 + spread),
        "low": price * (1 - spread),
        "close": price,
        "volume": rng.uniform(10, 500, n),
        "close_time": ot + pd.Timedelta(minutes=1) - pd.Timedelta(milliseconds=1),
        "trades": rng.integers(1, 100, n),
    })
    df.loc[0, "open"] = price[0]
    fund = pd.DataFrame(columns=["funding_time", "funding_rate", "mark_price"])
    return df, fund


def download(symbol: str, start: str, end: str, data_dir: Path,
             allow_synthetic: bool = True) -> dict:
    """Download (or synthesize) and cache klines/funding/meta. Returns meta dict."""
    data_dir.mkdir(parents=True, exist_ok=True)
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    meta_path = data_dir / f"{symbol}_meta.json"
    try:
        meta = get_symbol_meta(symbol)
        meta["data_source"] = "binance_fapi"
        klines = fetch_klines(symbol, start_ms, end_ms)
        funding = fetch_funding(symbol, start_ms, end_ms)
    except DataUnavailable as e:
        if not allow_synthetic:
            raise
        print(f"[DATA ERROR] {e}")
        print("[FALLBACK] writing SYNTHETIC dataset - results are NOT real")
        meta = {"symbol": symbol, "contractType": "SYNTHETIC", "tick_size": 0.01,
                "step_size": 0.01, "data_source": "synthetic"}
        klines, funding = make_synthetic(symbol, start_ms, end_ms)
    klines.to_parquet(data_dir / f"{symbol}_1m.parquet", index=False)
    funding.to_parquet(data_dir / f"{symbol}_funding.parquet", index=False)
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def load(symbol: str, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    klines = pd.read_parquet(data_dir / f"{symbol}_1m.parquet")
    funding_path = data_dir / f"{symbol}_funding.parquet"
    funding = pd.read_parquet(funding_path) if funding_path.exists() else pd.DataFrame(
        columns=["funding_time", "funding_rate", "mark_price"])
    meta = json.loads((data_dir / f"{symbol}_meta.json").read_text())
    return klines, funding, meta
