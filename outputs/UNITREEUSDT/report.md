# Closed-market grid backtest: UNITREEUSDT

data source: `binance_fapi` | contract: `TRADIFI_PERPETUAL` | underlying: `CN_EQUITY` | market: SSE

## Current finding

**not robust**

closed-market grid net pnl (eval, best dev params): **-1.81 USDT** over 16 windows (mean -0.011%/window, CI95 [-0.034%, +0.007%]).

P&L attribution: completed grid cycles +1.47 USDT; inventory/liquidation -2.63 USDT; funding -0.05 USDT; fees -0.40 USDT; exit slippage -0.21 USDT.

## Evidence

| split | n | total net | mean%/win | win rate | max DD | worst window |
|---|---|---|---|---|---|---|
| development | 24 | -0.06 USDT | -0.000 | 50% | -0.11% | -0.73 USDT @ 2026-08-24 07:00 |
| evaluation | 16 | -1.81 USDT | -0.011 | 50% | -0.16% | -1.38 USDT @ 2026-09-10 07:00 |

### Strategy benchmarks (same grid params)

- **B_closed**: n=40, net -1.87 USDT, mean -0.005%/window, win 50%
- **C_overnight**: n=15, net -2.48 USDT, mean -0.017%/window, win 47%
- **D_weekend**: n=4, net -0.37 USDT, mean -0.009%/window, win 50%
- **A_24x7**: n=1, net -20.11 USDT, mean -2.011%/window, win 0%

window types traded: {'lunch_break': 20, 'weekday_overnight': 15, 'weekend': 4, 'unclosed_tail': 1}

## Failure modes

- worst window 2026-09-10 07:00:00+00:00 (weekday_overnight): net -1.38 USDT, underlying moved -3.35% -> one-way trend overwhelmed the grid
- tail inventory liquidation: -9.28 USDT total (495% of net)
- funding: -0.87 USDT total (46% of net)
- exit slippage cost: -0.51 USDT total (27% of net)

## Robustness (best dev params, evaluation windows)

| test | setting | mean%/win | total net | win rate |
|---|---|---|---|---|
| exec_model | optimistic | +0.011 | +1.73 USDT | 56% |
| exec_model | conservative | -0.011 | -1.81 USDT | 50% |
| slippage | 0bps | -0.010 | -1.60 USDT | 62% |
| slippage | 2bps | -0.011 | -1.69 USDT | 56% |
| slippage | 5bps | -0.011 | -1.81 USDT | 50% |
| slippage | 10bps | -0.013 | -2.02 USDT | 50% |
| maker_fee | 0bp | -0.010 | -1.62 USDT | 56% |
| maker_fee | 1bp | -0.011 | -1.72 USDT | 56% |
| maker_fee | 2bp | -0.011 | -1.81 USDT | 50% |
| fill_drop | 0% | -0.011 | -1.81 USDT | 50% |
| fill_drop | 10% | -0.011 | -1.81 USDT | 50% |
| fill_drop | 25% | -0.011 | -1.81 USDT | 50% |

## Conclusion

**not robust**.

Caveats: short sample, single-contract evidence, kline-based fills (no queue/liquidity model), funding settled at Binance mark price.
