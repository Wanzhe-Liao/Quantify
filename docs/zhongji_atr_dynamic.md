# ZHONGJIUSDT targeted dynamic ATR grid study

Status: **retrospective exploratory research**. The Aug 14–Sep 16, 2026 period was already inspected in earlier grid work, so none of these results is a fresh holdout.

## Scope

The targeted search expands the prior ATR implementation into 870 configurations across 17 dynamic families. It tests:

- ATR10 / ATR30 / ATR60 / ATR120 / EWMA-ATR horizons;
- ATR multipliers and dynamic spacing caps/floors;
- fixed vs 5/15/30/60 minute center refresh;
- EMA30 / VWAP60 / trailing-price centers;
- hysteretic re-centering;
- inventory-dependent widening and center skew;
- volatility-targeted inventory budgets;
- trend-asymmetric spacing and trend-biased size;
- trend-side / contrarian / fixed long / fixed short grids;
- breakout pause / breakout stop;
- ATR10:ATR120 term-structure widening, risk reduction, and shock pause;
- combined dynamic risk-control grids.

Execution assumptions remain conservative: maker 2 bps, taker 5 bps, 5 bps forced-exit slippage, <=1% of 1-minute bar volume, funding at cached Binance timestamps, adverse OHLC side ordering, flat before the underlying market reopens.

## Main findings

- 870 candidates × 44 usable closed-market windows = 38,280 candidate-window simulations.
- No single candidate is strictly profitable in every development, historical-audit, or all-seen window.
- A per-window oracle finds at least one profitable ATR candidate in 43/44 windows. The remaining window is a lunch break with insufficient movement to overcome trading costs; its best tested outcome is 0.
- The main structural failure remains weekday overnight directional movement, not insufficient ATR parameter coverage.

### Most useful frozen lunch candidate

`atr_refresh_aewm_m2_n4_r5_c3`

Interpretation:

- EWMA-ATR(30) spacing;
- spacing approximately `2 × ATR / price`, subject to fee/tick floors and a 1.2% cap;
- 4 paired rungs per side;
- VWAP60 center;
- center/spacing refreshed every 5 minutes;
- 25% gross inventory budget.

Historical-audit lunch windows (selection frozen from pre-Sep-1 data):

- 11 windows;
- net PnL +2.27 USDT per 1,000 USDT session capital;
- 8 positive, 1 negative, 2 zero/no-profit windows;
- worst window -0.43 USDT;
- not every-window profitable.

A causal weekly by-window-type selector produced +2.66 USDT over 12 lunch windows (9 positive / 1 negative / 2 zero; block-bootstrap 95% CI for mean window PnL about +0.06 to +0.41 USDT), but weekday overnight remained negative.

## Interpretation

The targeted ATR search supports a narrower hypothesis: **dynamic ATR grids may be useful during ZHONGJI lunch closures**, especially with a fast VWAP-centered refresh. It does **not** support the stronger claim that one ATR grid can profit reliably in every non-trading period. Overnight behavior appears to require a separate directional / cross-market model rather than further ATR multiplier tuning.
