# Regime-gated closed-market grid: ZHONGJIUSDT

Reference market: **HKEX**
Observation delay: **30 min**
Frozen grid params: `{'spacing': 0.004, 'range_pct': 0.01, 'levels': 10, 'max_inventory_pct': 0.25}`

Thresholds are fit on development windows only. Evaluation windows are never used to choose a gate.
Only complete closed-market windows are included. Skipped windows count as zero-return opportunities.

| gate | split | traded/eligible | coverage | net USDT | mean/oppty | mean/active | active win | max DD | CVaR20 | delta vs baseline (CI95) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| delayed_all | development | 27/27 | 1.000 | -2.49 | -0.00009 | -0.00009 | 0.370 | -0.00393 | -0.00088 | 0.00000 ([0.00000, 0.00000]) |
| delayed_all | evaluation | 18/18 | 1.000 | 2.42 | 0.00013 | 0.00013 | 0.500 | -0.00278 | -0.00077 | 0.00000 ([0.00000, 0.00000]) |
| lowvol_q50 | development | 14/27 | 0.519 | 1.44 | 0.00005 | 0.00010 | 0.286 | -0.00095 | -0.00005 | 0.00015 ([-0.00001, 0.00036]) |
| lowvol_q50 | evaluation | 10/18 | 0.556 | -1.38 | -0.00008 | -0.00014 | 0.400 | -0.00175 | -0.00049 | -0.00021 ([-0.00052, 0.00006]) |
| lowvol_q50_driftguard | development | 10/27 | 0.370 | 0.09 | 0.00000 | 0.00001 | 0.200 | -0.00095 | -0.00004 | 0.00010 ([-0.00008, 0.00032]) |
| lowvol_q50_driftguard | evaluation | 9/18 | 0.500 | 0.12 | 0.00001 | 0.00001 | 0.444 | -0.00025 | -0.00003 | -0.00013 ([-0.00048, 0.00021]) |
| lowvol_q25 | development | 7/27 | 0.259 | 0.22 | 0.00001 | 0.00003 | 0.143 | -0.00015 | -0.00001 | 0.00010 ([-0.00009, 0.00033]) |
| lowvol_q25 | evaluation | 9/18 | 0.500 | -1.38 | -0.00008 | -0.00015 | 0.444 | -0.00175 | -0.00049 | -0.00021 ([-0.00052, 0.00006]) |
| lowvol_q75 | development | 20/27 | 0.741 | -0.59 | -0.00002 | -0.00003 | 0.350 | -0.00232 | -0.00056 | 0.00007 ([-0.00002, 0.00023]) |
| lowvol_q75 | evaluation | 12/18 | 0.667 | 1.06 | 0.00006 | 0.00009 | 0.500 | -0.00175 | -0.00049 | -0.00008 ([-0.00033, 0.00014]) |

## Primary decision rule

Primary gate: `lowvol_q50`. It must improve evaluation mean return per eligible opportunity, max drawdown, and CVaR20 versus `delayed_all`.
Statistical support additionally requires the paired bootstrap CI for gate-minus-baseline mean return to lie entirely above zero.

- point-estimate pass: **False**
- paired delta CI excludes zero: **False**
- statistically supported: **False**

`lowvol_q25` and `lowvol_q75` are sensitivity analyses only; they must not be selected post hoc from evaluation performance.
