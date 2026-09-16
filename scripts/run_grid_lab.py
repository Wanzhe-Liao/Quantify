#!/usr/bin/env python3
"""Bounded 32-family / 192-candidate experiment on cached real data.

Previously inspected dates are retrospective diagnostics, NEVER a fresh test.
No synthetic fallbacks, downloads, trading orders or endless tuning loops.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import time
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
import pandas as pd
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.calendar import spec_from_config, closed_market_windows
from src.grid_lab import candidate_registry, run_array

ROOT = Path(__file__).resolve().parents[1]
CUTOFF = pd.Timestamp('2026-09-01', tz='UTC')
AUDITED_START = pd.Timestamp('2026-08-14', tz='UTC')
AUDITED_END = pd.Timestamp('2026-09-16 00:01', tz='UTC')
PROTOCOL = {
    'status': 'retrospective_exploration_no_fresh_holdout',
    'initial_capital_usdt': 1000., 'fixed_notional_budget_fraction': .25,
    'candidates': 192, 'families': 32,
    'fixed_development_cutoff_utc': str(CUTOFF),
    'selection': 'past mean + 0.5 * exact lower-tail ES20; require positive past mean and >=25% windows with fills; else CASH',
    'walk_forward': 'weekly refit; >=12 fully ended past windows; embargo straddling windows; CASH if no eligible candidate',
    'success': 'every eligible window strictly net positive; zero/skipped/no-fill is not a win; separate execution stress and fresh future validation required',
    'fees_bps': {'maker': 2, 'taker': 5, 'market_slippage': 5},
    'max_bar_volume_participation': .01,
    'calendar': 'auction-safe: HKEX 09:00-12:00/13:00-16:10; SSE 09:15-11:30/13:00-15:30 local, repo holiday list within audited dates only',
    'funding': 'cached actual timestamp/rate/mark; <=1sec boundary offset booked on entering minute inventory; no future rate used in signals',
    'important': 'engine is a new paired-rung grid, not numerically equivalent to legacy dictionary-grid implementation; old results retained unchanged',
}


def load_cached(symbol, data_dir):
    meta = json.loads((data_dir / f'{symbol}_meta.json').read_text())
    if meta.get('data_source') != 'binance_fapi':
        raise ValueError('Real-data research refuses synthetic/unknown data sources')
    frames, hashes = [], {}
    for suffix in ('1m', 'funding'):
        file = data_dir / f'{symbol}_{suffix}.parquet'
        if file.exists():
            frame = pd.read_parquet(file)
        else:
            file = data_dir / f'{symbol}_{suffix}.csv.gz'
            frame = pd.read_csv(file, float_precision='round_trip')
        hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
        for col in ('open_time', 'close_time', 'funding_time'):
            if col in frame:
                frame[col] = pd.to_datetime(frame[col], utc=True, format='mixed')
        frames.append(frame)
    k, f = frames
    k = k.sort_values('open_time').reset_index(drop=True)
    if k.open_time.duplicated().any() or f.funding_time.duplicated().any():
        raise ValueError('Duplicate timestamps in source data')
    vals = k[['open','high','low','close','volume']].to_numpy(dtype=float)
    if not np.isfinite(vals).all() or (vals[:, :4] <= 0).any() or (vals[:, 4] < 0).any():
        raise ValueError('Invalid price or volume')
    if ((k.high + 1e-8 < k[['open','close']].max(axis=1)) |
        (k.low - 1e-8 > k[['open','close']].min(axis=1))).any():
        raise ValueError('Inconsistent OHLC')
    if k.open_time.min() < AUDITED_START or k.open_time.max() >= AUDITED_END:
        raise ValueError('Outside audited period; review holidays before extension')
    if len(f):
        fv = f[['funding_rate','mark_price']].to_numpy(dtype=float)
        if not np.isfinite(fv).all() or (f.mark_price <= 0).any():
            raise ValueError('Invalid funding marks/rates')
        offset = (f.funding_time - f.funding_time.dt.floor('min')).dt.total_seconds()
        if offset.max() > 1:
            raise ValueError('Intraminute funding requires tick sequencing')
    else:
        raise ValueError('Missing funding data cannot be assumed zero')
    return k, f, meta, hashes


def prepare(k, funding):
    c = k.close.astype(float)
    ret = c.pct_change(fill_method=None)
    tr = pd.concat([k.high-k.low, (k.high-c.shift()).abs(), (k.low-c.shift()).abs()], axis=1).max(axis=1)
    vw = (c*k.volume).rolling(60, min_periods=30).sum() / k.volume.rolling(60, min_periods=30).sum().replace(0, np.nan)
    efficiency = (c-c.shift(30)).abs() / c.diff().abs().rolling(30).sum().replace(0, np.nan)
    features = pd.DataFrame({
        'previous_close': c.shift(),
        'ema30': c.ewm(span=30, adjust=False).mean().shift(),
        'ema120': c.ewm(span=120, adjust=False).mean().shift(),
        'vwap60': vw.fillna(c).shift(),
        'atr30': tr.rolling(30, min_periods=30).mean().shift(),
        'rv30': ret.rolling(30, min_periods=30).std(ddof=0).shift(),
        'ret30': (c/c.shift(30)-1).shift(),
        'efficiency30': efficiency.fillna(0).shift(),
    })
    ff = funding.assign(minute=funding.funding_time.dt.floor('min'),
                        weighted=funding.funding_rate*funding.mark_price).groupby('minute').weighted.sum()
    return np.column_stack([k[['open','high','low','close','volume']].to_numpy(), features.to_numpy(),
                            k.open_time.map(ff).fillna(0).to_numpy()])


def windows_for(k, cfg, symbol, policy='auction_safe'):
    spec = spec_from_config(cfg['symbols'][symbol])
    if policy == 'auction_safe':
        if spec.name == 'HKEX':
            spec.sessions = [(time(9),time(12)), (time(13),time(16,10))]
        elif spec.name == 'SSE':
            spec.sessions = [(time(9,15),time(11,30)), (time(13),time(15,30))]
        else:
            raise ValueError('No reviewed auction-safe schedule for this symbol')
    elif policy != 'repo_regular':
        raise ValueError(policy)
    w = closed_market_windows(spec, k.open_time.min(), k.open_time.max()+pd.Timedelta(minutes=1))
    w = w[w.complete].copy()
    w['start_idx'] = k.open_time.searchsorted(w.window_start).astype(int)
    w['end_idx'] = k.open_time.searchsorted(w.window_end).astype(int)
    w['expected_bars'] = ((w.window_end-w.window_start)/pd.Timedelta(minutes=1)).astype(int)
    w['observed_bars'] = w.end_idx-w.start_idx
    w['usable'] = (w.expected_bars == w.observed_bars) & (w.start_idx>=60) & (w.expected_bars>=10)
    for idx, row in w.iterrows():
        times = k.open_time.iloc[row.start_idx:row.end_idx]
        if len(times)>1 and not (times.diff().dropna()==pd.Timedelta(minutes=1)).all():
            w.loc[idx,'usable'] = False
    return w


def es20(values):
    """Exact empirical 20% lower-tail mass, including fractional boundary mass."""
    x = np.sort(np.asarray(values, dtype=float))
    if not len(x):
        return 0.
    mass = len(x)*.2
    whole = int(np.floor(mass))
    return float((x[:whole].sum()+(mass-whole)*x[min(whole,len(x)-1)])/mass)


def summarize(df, capital=1000.):
    if df.empty:
        return {'windows': 0}
    v = df.sort_values('window_start').net_pnl.to_numpy(dtype=float)
    eq = np.r_[capital, capital+np.cumsum(v)]
    return {'windows':len(v), 'net':float(v.sum()), 'mean':float(v.mean()),
            'wins':int((v>1e-8).sum()), 'losses':int((v < -1e-8).sum()),
            'zeros':int((abs(v)<=1e-8).sum()), 'win_rate':float((v>1e-8).mean()),
            'filled_windows':int((df.fills>0).sum()), 'min_window':float(v.min()),
            'max_window':float(v.max()), 'es20':es20(v),
            'window_end_dd':float(np.min(eq/np.maximum.accumulate(eq)-1)),
            'worst_intrawindow_dd':float(df.max_drawdown.min()),
            'every_window_profitable':bool(np.all(v>1e-8))}


def choose(train, family=None):
    if family:
        train = train[train.family==family]
    candidates=[]
    for name,g in train.groupby('candidate'):
        if len(g)<12 or (g.fills>0).mean()<.25 or g.net_pnl.mean()<=0:
            continue
        candidates.append((float(g.net_pnl.mean()+.5*es20(g.net_pnl)), name))
    return sorted(candidates, key=lambda v:(-v[0],v[1]))[0][1] if candidates else None


def cash_rows(windows, label):
    df = windows[['window_start','window_end','window_type']].copy()
    for col in ('net_pnl','return_pct','fills','max_drawdown','cycle_pnl','inventory_pnl',
                'funding_pnl','maker_fees','taker_fees','slippage_cost'):
        df[col] = 0.
    df['candidate']='CASH'
    df['family']=label
    return df


def bootstrap_mean(df, seed=20260916, iterations=2000):
    """Five-local-date blocks, keeping same-day windows together."""
    if df.empty:
        return [None,None]
    temp=df.sort_values('window_start').copy()
    temp['date']=temp.window_start.dt.tz_convert('Asia/Shanghai').dt.date
    daily=temp.groupby('date').net_pnl.agg(['sum','size']).to_numpy()
    rng=np.random.default_rng(seed)
    n=len(daily)
    starts=rng.integers(n,size=(iterations,int(np.ceil(n/5))))
    idx=((starts[:,:,None]+np.arange(5))%n).reshape(iterations,-1)[:,:n]
    samples=daily[idx]
    means=samples[:,:,0].sum(axis=1)/samples[:,:,1].sum(axis=1)
    return np.quantile(means,[.025,.975]).tolist()


def run(symbol, args, cfg, registry, output):
    k,f,meta,hashes=load_cached(symbol, Path(args.data_dir))
    a=prepare(k,f)
    wa=windows_for(k,cfg,symbol)
    wa.to_csv(output/f'{symbol}_window_audit.csv',index=False)
    w=wa[wa.usable].reset_index(drop=True)
    if w.empty:
        raise ValueError('No fully observed windows')
    print(f'{symbol}: {len(k)} bars; {len(w)} usable windows; {len(wa)-len(w)} excluded',flush=True)
    base_kw=dict(tick=meta['tick_size'],step=meta['step_size'])
    all_rows=[]
    lookup={}
    for ci, candidate in enumerate(registry):
        for wr in w.itertuples():
            ar=a[wr.start_idx:wr.end_idx]
            rec,eq=run_array(ar,candidate,**base_kw)
            rec.update(symbol=symbol,window_start=wr.window_start,window_end=wr.window_end,window_type=wr.window_type)
            all_rows.append(rec)
            lookup[(candidate.name,wr.window_start)]=eq
        if (ci+1)%48==0:
            print(f'{symbol}: {ci+1}/{len(registry)} candidates',flush=True)
    results=pd.DataFrame(all_rows)
    results.to_csv(output/f'{symbol}_all_windows.csv.gz',index=False)
    dev=results[results.window_end<=CUTOFF]
    ev=results[results.window_start>=CUTOFF]
    score_rows=[]
    for name,g in results.groupby('candidate'):
        for label,idx in [('development',g.window_end<=CUTOFF),('historical_audit',g.window_start>=CUTOFF),('all_seen',g.window_start.notna())]:
            score_rows.append(dict(candidate=name,family=g.family.iloc[0],split=label,**summarize(g[idx])))
    scores=pd.DataFrame(score_rows)
    scores.to_csv(output/f'{symbol}_candidate_summary.csv',index=False)
    selections=[]; policies=[]
    for family in ['ALL']+sorted(results.family.unique()):
        name=choose(dev,None if family=='ALL' else family)
        evw=w[w.window_start>=CUTOFF]
        chosen=ev[ev.candidate==name].copy() if name else cash_rows(evw,family)
        chosen['policy']=family
        chosen['selection']='frozen_development'
        policies.append(chosen)
        selections.append(dict(policy=family,candidate=name or 'CASH',**summarize(chosen)))
    pd.DataFrame(selections).to_csv(output/f'{symbol}_frozen_policies.csv',index=False)
    wf=[]; decisions=[]
    dates=w.window_start.dt.tz_convert('Asia/Shanghai').dt.tz_localize(None).dt.to_period('W-SUN').dt.start_time
    for week in sorted(dates.unique()):
        boundary=pd.Timestamp(week,tz='Asia/Shanghai').tz_convert('UTC')
        before=results[results.window_end<=boundary]
        if before.window_start.nunique()<12:
            continue
        target_w=w[dates==week]
        for family in ['ALL']+sorted(results.family.unique()):
            name=choose(before,None if family=='ALL' else family)
            selected=(results[(results.candidate==name)&results.window_start.isin(target_w.window_start)].copy()
                      if name else cash_rows(target_w,family))
            selected['policy']=family
            selected['selection']='retrospective_weekly_walkforward'
            selected['fit_boundary']=boundary
            wf.append(selected)
            decisions.append(dict(symbol=symbol,week=str(week),fit_boundary=str(boundary),policy=family,
                                  candidate=name or 'CASH',train_windows=before.window_start.nunique(),
                                  max_train_end=str(before.window_end.max()),test_windows=len(selected)))
    wf=pd.concat(wf,ignore_index=True) if wf else pd.DataFrame()
    wf.to_csv(output/f'{symbol}_walkforward_windows.csv',index=False)
    pd.DataFrame(decisions).to_csv(output/f'{symbol}_walkforward_decisions.csv',index=False)
    wfs=[]
    for fam,g in wf.groupby('policy'):
        wfs.append(dict(policy=fam,**summarize(g),ci95_mean_usdt=bootstrap_mean(g)))
    pd.DataFrame(wfs).to_csv(output/f'{symbol}_walkforward_summary.csv',index=False)
    stress=[]
    scenarios={
        'base':{}, 'fees_slippage_2x':dict(maker=.0004,taker=.001,slip_bps=10),
        'fill_drop25':dict(fill_drop=.25), 'volume_cap_0.1pct':dict(participation=.001),
        'buy_first':dict(path_mode=1), 'sell_first':dict(path_mode=2),
        'combined':dict(maker=.0004,taker=.001,slip_bps=10,fill_drop=.25,participation=.001),
    }
    byname={c.name:c for c in registry}
    unique=sorted({s['candidate'] for s in selections if s['candidate']!='CASH'})
    for name in unique:
        for scenario,kwargs in scenarios.items():
            records=[]
            for wr in w[w.window_start>=CUTOFF].itertuples():
                r,_=run_array(a[wr.start_idx:wr.end_idx],byname[name],**base_kw,**kwargs)
                r.update(window_start=wr.window_start,window_type=wr.window_type)
                records.append(r)
            stress.append(dict(candidate=name,scenario=scenario,**summarize(pd.DataFrame(records))))
    pd.DataFrame(stress).to_csv(output/f'{symbol}_stress.csv',index=False)
    fixed_global=policies[0]
    wf_global=wf[wf.policy=='ALL']
    bytype=[]
    for label,df in [('frozen',fixed_global),('walkforward',wf_global)]:
        for typ,g in df.groupby('window_type'):
            bytype.append(dict(selection=label,window_type=typ,**summarize(g)))
    pd.DataFrame(bytype).to_csv(output/f'{symbol}_by_period.csv',index=False)
    curve_rows=[]
    running=1000.
    for row in wf_global.sort_values('window_start').itertuples():
        wr=w[w.window_start==row.window_start].iloc[0]
        eq=lookup.get((row.candidate,row.window_start),np.full(int(wr.end_idx-wr.start_idx),1000.))
        idx=k.open_time.iloc[int(wr.start_idx):int(wr.end_idx)]
        curve_rows.extend(zip(idx, running+eq-1000.))
        running+=row.net_pnl
    curves=pd.DataFrame(curve_rows,columns=['time','equity'])
    curves.to_csv(output/f'{symbol}_walkforward_equity.csv.gz',index=False)
    ev_summary=scores[scores.split=='historical_audit']
    all_summary=scores[scores.split=='all_seen']
    headline={
        'symbol':symbol,'bars':len(k),'first_bar':str(k.open_time.min()),'last_bar':str(k.open_time.max()),
        'windows':len(w),'window_types':w.window_type.value_counts().to_dict(),'excluded_windows':len(wa)-len(w),
        'source_hashes':hashes,'metadata':meta,
        'historical_audit_all_positive_candidates':int(ev_summary.every_window_profitable.sum()),
        'all_seen_all_positive_candidates':int(all_summary.every_window_profitable.sum()),
        'frozen_development':selections[0],
        'walkforward':dict(summarize(wf_global),ci95_mean_usdt=bootstrap_mean(wf_global)),
        'by_period':bytype,
        'note':'No future unseen test data are present. Positive retrospective results do not establish stable profitability.',
    }
    if len(curves):
        eq=np.r_[1000.,curves.equity.to_numpy()]
        headline['walkforward']['minute_equity_max_drawdown']=float(np.min(eq/np.maximum.accumulate(eq)-1))
    (output/f'{symbol}_summary.json').write_text(json.dumps(headline,indent=2,default=str))
    print(json.dumps(headline,indent=2,default=str),flush=True)
    return headline


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--symbols',nargs='+',default=['ZHONGJIUSDT','UNITREEUSDT'])
    p.add_argument('--data-dir',default=str(ROOT/'data'))
    p.add_argument('--output-dir',default=str(ROOT/'outputs'/'grid_lab'))
    args=p.parse_args()
    output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    registry=candidate_registry()
    frozen={'protocol':PROTOCOL,'registry':[asdict(c) for c in registry]}
    canonical=json.dumps(frozen,sort_keys=True,indent=2)
    (output/'protocol.json').write_text(canonical)
    (output/'protocol.sha256').write_text(hashlib.sha256(canonical.encode()).hexdigest()+'\n')
    cfg=yaml.safe_load((ROOT/'config/default.yaml').read_text())
    headlines=[run(symbol,args,cfg,registry,output) for symbol in args.symbols]
    report=['# Grid laboratory: bounded 32-family experiment','',
            '**Retrospective research only. Previously inspected dates are not a fresh holdout.**','',
            '192 candidates per symbol; 3 spacings (0.1/0.2/0.4%) x 2 actual rung counts (2/4 per side).',
            'Capital: 1,000 USDT per symbol; fixed gross inventory budget 25%; no leverage or martingale.',
            'Auction-safe windows; zero/skipped windows are not counted as wins. Every session is flattened, with fees, funding and market exit slippage included.','',
            '|symbol|complete windows|all-window-positive candidates|frozen-dev selected|audit net|audit wins/losses/zero|walkforward net|WF wins/losses/zero|',
            '|---|---:|---:|---|---:|---|---:|---|']
    for h in headlines:
        d=h['frozen_development'];w=h['walkforward']
        report.append(f"|{h['symbol']}|{h['windows']}|{h['all_seen_all_positive_candidates']}/192|{d['candidate']}|{d['net']:.4f}|{d['wins']}/{d['losses']}/{d['zeros']}|{w['net']:.4f}|{w['wins']}/{w['losses']}/{w['zeros']}|")
    report+=['','## Limits and interpretation','',
             '- Old results remain unchanged; this is a new paired-rung engine with center exits, not a claim to reproduce the legacy book.',
             '- 1m OHLC cannot identify queue priority or true intrabar paths. Adverse ordering and volume caps are model stress, not proof of executable fills.',
             '- Stops trigger on completed bars and fill next open; gap losses can exceed thresholds. Intrawindow drawdown uses minute marks, not true tick/mark-price extrema.',
             '- Tick/quantity/min-notional use cached metadata and launch terms, not point-in-time historical filter changes; 5 USDT minimum assumed unchanged.',
             '- These data have already informed prior research. Walk-forward is useful for historical audit, but is not independent prospective validation.',
             '- Hundreds of candidates create selection bias. No candidate may be chosen by its audit ranking, and a positive mean does not imply every window wins.',
             '- Every failing candidate and every losing/zero window remain in the output. No endless tuning loop or automatic live deployment is implemented.','']
    (output/'report.md').write_text('\n'.join(report))
    (output/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},indent=2))

if __name__=='__main__':
    main()
