# Regime-gated closed-market grid: UNITREEUSDT

Reference market: **SSE**
Observation delay: **30 min**
Frozen grid params: `{'spacing': 0.004, 'range_pct': 0.01, 'levels': 10, 'max_inventory_pct': 0.25}`

Thresholds are fit on development windows only. Evaluation windows are never used to choose a gate.
Only complete closed-market windows are included. Skipped windows count as zero-return opportunities.

| gate | split | traded/eligible | coverage | net USDT | mean/oppty | mean/active | active win | max DD | CVaR20 | delta vs baseline (CI95) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| delayed_all | development | 23/23 | 1.000 | -3.80 | -0.00017 | -0.00017 | 0.348 | -0.00607 | -0.00112 | 0.00000 ([0.00000, 0.00000]) |
| delayed_all | evaluation | 16/16 | 1.000 | -2.90 | -0.00018 | -0.00018 | 0.312 | -0.00359 | -0.00080 | 0.00000 ([0.00000, 0.00000]) |
| lowvol_q50 | development | 12/23 | 0.522 | 1.88 | 0.00008 | 0.00016 | 0.417 | -0.00037 | -0.00002 | 0.00025 ([0.00005, 0.00047]) |
| lowvol_q50 | evaluation | 15/16 | 0.938 | -2.90 | -0.00018 | -0.00019 | 0.333 | -0.00359 | -0.00080 | 0.00000 ([0.00000, 0.00000]) |
| lowvol_q50_driftguard | development | 11/23 | 0.478 | 1.70 | 0.00007 | 0.00015 | 0.364 | -0.00043 | -0.00002 | 0.00024 ([0.00004, 0.00046]) |
| lowvol_q50_driftguard | evaluation | 15/16 | 0.938 | -2.90 | -0.00018 | -0.00019 | 0.333 | -0.00359 | -0.00080 | 0.00000 ([0.00000, 0.00000]) |
| lowvol_q25 | development | 6/23 | 0.261 | 0.47 | 0.00002 | 0.00008 | 0.500 | 0.00000 | 0.00000 | 0.00019 ([-0.00005, 0.00043]) |
| lowvol_q25 | evaluation | 12/16 | 0.750 | -2.22 | -0.00014 | -0.00018 | 0.417 | -0.00291 | -0.00079 | 0.00004 ([0.00000, 0.00012]) |
| lowvol_q75 | development | 17/23 | 0.739 | 0.65 | 0.00003 | 0.00004 | 0.412 | -0.00122 | -0.00042 | 0.00019 ([0.00002, 0.00040]) |
| lowvol_q75 | evaluation | 16/16 | 1.000 | -2.90 | -0.00018 | -0.00018 | 0.312 | -0.00359 | -0.00080 | 0.00000 ([0.00000, 0.00000]) |

## Primary decision rule

Primary gate: `lowvol_q50`. It must improve evaluation mean return per eligible opportunity, max drawdown, and CVaR20 versus `delayed_all`.
Statistical support additionally requires the paired bootstrap CI for gate-minus-baseline mean return to lie entirely above zero.

- point-estimate pass: **False**
- paired delta CI excludes zero: **False**
- statistically supported: **False**

`lowvol_q25` and `lowvol_q75` are sensitivity analyses only; they must not be selected post hoc from evaluation performance.
