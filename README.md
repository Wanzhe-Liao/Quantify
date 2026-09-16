# grid_backtest

Backtest research framework: **does a symmetric grid on Binance TradFi (stock)
USDT perpetuals earn repeatable risk-adjusted returns while the underlying
stock market is closed?**

The unit of analysis is the `closed_market_window` (underlying session close ->
next open), not the individual fill. Every window starts flat and is force-
liquidated (taker fee + exit slippage) before the underlying reopens. Headline
PnL is always `final_equity_after_liquidation`.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

```bash
python scripts/download_data.py --symbol ZHONGJIUSDT --start 2026-08-14 --end 2026-09-16
python scripts/download_data.py --symbol UNITREEUSDT --start 2026-08-14 --end 2026-09-16
```

Uses public `fapi.binance.com` endpoints (`/fapi/v1/klines`,
`/fapi/v1/fundingRate`, `/fapi/v1/exchangeInfo`). If a symbol is not on fapi the
downloader prints the exact error and writes a clearly-marked synthetic
dataset instead (`data_source: synthetic` in meta) — check `data/<symbol>_meta.json`.

## Run

```bash
python scripts/run_backtest.py --symbol ZHONGJIUSDT --config config/default.yaml
```

Outputs to `outputs/<symbol>/`: `windows.parquet`, `parameter_summary.csv`,
`best_dev_parameters.json`, `evaluation_summary.json`, `robustness_summary.csv`,
`equity_curve.png`, `parameter_surface.png`, `pnl_vs_volatility.png`,
`pnl_vs_window_return.png`, `report.md`.

## Tests

```bash
pytest tests/ -q
```

## Design notes

- **Market calendar** (`src/calendar.py`): per-symbol underlying market spec
  (HKEX / SSE / NASDAQ) with local session times + holiday lists; produces
  `weekday_overnight` / `lunch_break` / `weekend` / `holiday` windows in UTC.
- **Grid** (`src/grid.py`): symmetric levels `P0*(1±k*spacing)`, center fixed at
  window-start price, no re-centering, no martingale.
- **Execution** (`src/execution.py`):
  - `optimistic`: `low <= limit` fills, intra-bar cascade allowed.
  - `conservative` (default): strict `low < limit` / `high > limit`; only orders
    resting at bar start can fill (no cascade); when one bar touches both sides
    both orderings are simulated and the lower-equity outcome is kept.
- **Accounting** (`src/accounting.py`): cash + signed inventory, average-cost
  realized PnL, maker/taker fees, funding settled at real Binance timestamps
  (`cashflow = -qty * mark * rate`, positive rate = longs pay).
- **Metrics** (`src/metrics.py`): per-window stats, bootstrap 95% CI over
  *windows* (never trades), chronological 60/40 dev/eval split — parameters are
  chosen on dev only.

## Known limitations

- 1m-kline fills cannot see queue position or intrabar sequencing; the
  conservative mode + fill-drop stress bound this but trade-level replay
  (Mode C) is the proper check.
- Mark price from the funding endpoint is used for funding notional.
- Holiday calendars are explicit lists in `config/default.yaml`; verify before
  extending the date range.
