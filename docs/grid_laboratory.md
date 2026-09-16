# Grid laboratory

## Reproduce

```bash
pip install -r requirements-lab.txt
pytest -q
python scripts/run_grid_lab.py
```

Results are written to `outputs/grid_lab/`. The runner reads the repository's
cached real klines and funding only. It refuses synthetic metadata, missing
funding, duplicate timestamps, inconsistent OHLC, partial windows, and dates
outside the reviewed 2026-08-14 through 2026-09-16 calendar. CSV-gzip snapshots
are also supported for environments without a parquet reader.

## Experiment

32 families, each with spacings 0.1%, 0.2%, 0.4% and 2 or 4 actual rungs per
side: 192 candidates per symbol. Families include arithmetic/geometric/wide
outer rungs, ATR/realized-volatility spacing, EMA/VWAP anchors, periodic idle
rung recentering, inventory skew/taper, long/short/directional grids, momentum
and contrarian size tilts, drift/efficiency/activity gates, time taper, loss
stop, profit target, trailing profit, breakout stop, 120-minute exit/reset,
regime switching and delayed entry. Different configurations can yield the
same realized path; 192 configurations are not 192 independent hypotheses.

Capital is 1,000 USDT per symbol with a 25% gross virtual inventory cap and
fixed sizing across windows. No leverage escalation or martingale is used.
Every eligible window is retained. Zero returns and no-fill/skipped windows
are not wins. Realized grid-cycle profits alone never determine success.

The new paired-rung engine addresses the missing-center exit in the legacy
book. It is NOT a numerically equivalent rewrite: independent rung accounting,
post-only resubmission, volume caps and scheduling are explicitly different.
Legacy code and reports remain intact for audit.

## Evaluation

The fixed development cutoff is 2026-09-01 00:00 UTC. A straddling window is
not allowed to contribute a future liquidation label to development. Weekly
walk-forward fits use only windows fully ended before the week's boundary,
with at least 12 past windows. The score is mean net PnL plus 0.5 times exact
lower-tail ES20. A candidate needs positive past mean and fills in at least
25% of past windows; otherwise the selector chooses cash.

The current dataset has already informed prior research. Both the fixed-date
comparison and the walk-forward are RETROSPECTIVE DIAGNOSTICS, not a fresh
out-of-sample test. Searching hundreds of candidates and subsequently choosing
by audit returns is prohibited. Five-date block bootstrap intervals are
conditional historical diagnostics, not selection-adjusted proof of an edge.

Execution stress is applied to frozen development-selected family candidates:
base, double fees/slippage, 25% maker-fill loss, 0.1% volume participation,
buy-first and sell-first paths, and combined stress. No stress-based reselection.

## Execution assumptions

- Lagged signals only; held exit targets never move with a new grid center.
- A rung fills at most once per minute; a new opposite leg cannot fill in the
  same minute. Marketable post-only submissions are rejected, not given free
  maker execution. Cancelled quotes must become passive again before reuse.
- Strict limit penetration, full-order fills only, and maximum participation
  of 1% of the bar's volume by default. This does not model queue priority.
- The lower liquidation value of two side orderings is used in adverse mode;
  this is not a guaranteed bound over every possible tick path.
- All stops are completed-minute signals filled at the next open; losses can
  gap beyond the configured stop. Final scheduled flattening is at the last
  minute's open, before the underlying session/auction begins.
- Actual cached funding rates and mark prices are applied to entering net
  inventory. Timestamp offsets within one second of a minute boundary are
  booked at that boundary; larger offsets are rejected pending tick replay.
- Maker/taker/slippage defaults are research assumptions: 2/5/5 bps. Launch
  minimum notional is 5 USDT, and cached tick/lot steps are used; historical
  filter revisions and user-specific fee tiers are not reconstructed.
- Equity/drawdown uses minute trade-price marks, not tick-level or liquidation
  mark prices. Positive simulated results are not verified executable profits.

## Calendar and contract provenance

The declared underlying is HKEX equity for ZHONGJI and SSE equity for UNITREE.
The laboratory excludes pre-opening/closing auctions and SSE after-hours
trading: HKEX 09:00-12:00 and 13:00-16:10; SSE 09:15-11:30 and 13:00-15:30,
all local times. Holiday lists must be reviewed before extending the sample.

Primary references checked on 2026-09-16:

- HKEX trading hours: https://www.hkex.com.hk/Services/Trading-hours-and-Severe-Weather-Arrangements/Trading-Hours/Securities-Market?sc_lang=en
- SSE stock trading: https://one.sse.com.cn/onething/gptz/
- UNITREE launch terms: https://www.binance.com/en/support/announcement/detail/3e662272597c44b7939f5db5c8c86d4f
- ZHONGJI launch summary: https://www.binance.com/en-IN/square/post/08-13-2026-binance-futures-will-list-6-usdt-priced-perpetual-contracts-for-traditional-asset-based-trading-355181035543298

## Stopping and next evidence

This is a finite, fully logged search, not an optimizer that keeps changing
parameters until the viewed test set passes. All losing candidates/windows
remain available. An all-positive historical result, even if found, would
still require untouched later windows and more realistic trade/quote replay.
No real trading, credentials, automatic parameter deployment, or recurring
background optimization is part of this PR.
