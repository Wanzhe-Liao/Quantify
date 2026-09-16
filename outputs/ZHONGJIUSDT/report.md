# Closed-market grid backtest: ZHONGJIUSDT

data source: `binance_fapi` | contract: `TRADIFI_PERPETUAL` | underlying: `HK_EQUITY` | market: HKEX

## Current finding

**inconclusive**

closed-market grid net pnl (eval, best dev params): **+4.08 USDT** over 19 windows (mean +0.021%/window, CI95 [-0.004%, +0.050%]).

P&L attribution: completed grid cycles +9.91 USDT; inventory/liquidation -4.56 USDT; funding +0.09 USDT; fees -1.17 USDT; exit slippage -0.20 USDT.

## Evidence

| split | n | total net | mean%/win | win rate | max DD | worst window |
|---|---|---|---|---|---|---|
| development | 27 | -1.11 USDT | -0.004 | 59% | -0.21% | -1.68 USDT @ 2026-08-17 08:00 |
| evaluation | 19 | +4.08 USDT | +0.021 | 63% | -0.16% | -0.85 USDT @ 2026-09-03 08:00 |

### Strategy benchmarks (same grid params)

- **B_closed**: n=46, net +2.96 USDT, mean +0.006%/window, win 61%
- **C_overnight**: n=17, net +0.75 USDT, mean +0.004%/window, win 59%
- **D_weekend**: n=5, net +0.25 USDT, mean +0.005%/window, win 40%
- **A_24x7**: n=1, net +7.80 USDT, mean +0.780%/window, win 100%

window types traded: {'lunch_break': 23, 'weekday_overnight': 17, 'weekend': 5, 'unclosed_tail': 1}

## Failure modes

- worst window 2026-08-17 08:00:00+00:00 (weekday_overnight): net -1.68 USDT, underlying moved +4.35% -> one-way trend overwhelmed the grid
- tail inventory liquidation: -12.06 USDT total (-407% of net)

## Robustness (best dev params, evaluation windows)

| test | setting | mean%/win | total net | win rate |
|---|---|---|---|---|
| exec_model | optimistic | +0.164 | +31.11 USDT | 79% |
| exec_model | conservative | +0.021 | +4.08 USDT | 63% |
| slippage | 0bps | +0.022 | +4.27 USDT | 63% |
| slippage | 2bps | +0.022 | +4.19 USDT | 63% |
| slippage | 5bps | +0.021 | +4.08 USDT | 63% |
| slippage | 10bps | +0.020 | +3.88 USDT | 63% |
| maker_fee | 0bp | +0.027 | +5.05 USDT | 63% |
| maker_fee | 1bp | +0.024 | +4.56 USDT | 63% |
| maker_fee | 2bp | +0.021 | +4.08 USDT | 63% |
| fill_drop | 0% | +0.021 | +4.08 USDT | 63% |
| fill_drop | 10% | +0.021 | +3.99 USDT | 63% |
| fill_drop | 25% | +0.019 | +3.58 USDT | 63% |

## Conclusion

**inconclusive**.

Caveats: short sample, single-contract evidence, kline-based fills (no queue/liquidity model), funding settled at Binance mark price.
