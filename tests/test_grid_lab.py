"""Regression tests for causality, accounting, abstention and selection."""
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.grid_lab import Candidate,candidate_registry,run_array
from scripts.run_grid_lab import es20,choose,prepare,summarize


def bars(rows):
    a=np.zeros((len(rows),14));a[:,:4]=rows;a[:,4]=100000
    a[:,5]=np.r_[100.,a[:-1,3]]
    a[:,6:9]=100.;a[:,9]=.3;a[:,10]=.001
    return a


def baseline(**kw):
    return replace(Candidate('test','fixed',spacing=.005,levels=2),**kw)


def test_registry_is_bounded_and_unique():
    r=candidate_registry()
    assert len(r)==192
    assert len({c.name for c in r})==192
    assert len({c.family for c in r})==32
    assert all(c.inventory==.25 for c in r)


def test_center_exit_and_no_same_bar_cascade():
    a=bars([(100,100.1,99.4,99.6),(99.6,100.2,99.55,99.8),(100,100.1,99.9,100)])
    out,eq=run_array(a,baseline(direction=1),maker=0,taker=0,slip_bps=0)
    assert out['cycles']==1
    assert out['cycle_pnl']==pytest.approx(.625)
    assert out['net_pnl']==pytest.approx(.625)
    a=bars([(100,101,99.4,100),(100,100,100,100),(100,100,100,100)])
    out,_=run_array(a,baseline(direction=1),maker=0,taker=0,slip_bps=0)
    assert out['cycles']==0


def test_final_equity_includes_forced_exit_costs():
    a=bars([(100,100.1,99.4,99.6),(99.6,99.8,99.2,99.4),(99,99,98,98)])
    out,eq=run_array(a,baseline(direction=1))
    assert out['taker_fees']>0 and out['slippage_cost']>0
    assert out['net_pnl']==pytest.approx(eq[-1]-1000)
    assert abs(out['reconciliation_error'])<1e-8
    b=a.copy();b[-1,1:4]=[120,80,100]
    out2,_=run_array(b,baseline(direction=1))
    assert out2['net_pnl']==pytest.approx(out['net_pnl'])


def test_zero_volume_and_minimum_order_constraints():
    a=bars([(100,102,98,100)]*4)
    a[:,4]=0
    out,_=run_array(a,baseline())
    assert out['fills']==0 and out['net_pnl']==0
    a[:,4]=.01
    out,_=run_array(a,baseline())
    assert out['fills']==0
    a[:,4]=1e6
    out,_=run_array(a,baseline(),min_notional=10000)
    assert out['fills']==0


def test_flat_market_not_counted_as_winning():
    a=bars([(100,100,100,100)]*36)
    for c in candidate_registry():
        out,_=run_array(a,c)
        assert out['net_pnl']==0
        assert out['fills']==0


def test_funding_on_existing_signed_inventory():
    a=bars([(100,100.1,99.4,99.6),(99.6,99.8,99.4,99.6),(99.6,99.6,99.6,99.6)])
    a[1,13]=.1
    out,_=run_array(a,baseline(direction=1),maker=0,taker=0,slip_bps=0)
    assert out['funding_pnl']==pytest.approx(-.125)
    a[1,13]=-.1
    out,_=run_array(a,baseline(direction=1),maker=0,taker=0,slip_bps=0)
    assert out['funding_pnl']==pytest.approx(.125)


def test_stop_is_next_open_not_retrospective_fill():
    a=bars([(100,100.1,99,99.2),(99.2,99.3,96,96.5),(94,96,93,95),(97,97,97,97)])
    out,eq=run_array(a,baseline(direction=1,stop=.002))
    assert out['exit_reason']==2
    assert eq[2]==eq[3]
    b=a.copy();b[2,0]=93;b[2,2]=92
    out2,_=run_array(b,baseline(direction=1,stop=.002))
    assert out2['net_pnl']<out['net_pnl']


def test_prefix_invariance_to_future_prices_and_funding():
    a=bars([(100,100.3,99.3,99.7)]*12)
    b=a.copy();b[8:,0:4]=[120,121,119,120];b[8:,13]=.1
    for c in candidate_registry()[::6]:
        _,ea=run_array(a,c);_,eb=run_array(b,c)
        assert np.array_equal(ea[:8],eb[:8])


@pytest.mark.parametrize('family',['fixed','geometric','atr_refresh','inventory_skew','reset120','trailing_profit'])
def test_random_paths_reconcile(family):
    rng=np.random.default_rng(22)
    c=100*np.exp(np.cumsum(rng.normal(0,.001,300)))
    o=np.r_[100,c[:-1]]
    a=bars(np.column_stack([o,np.maximum(o,c)+.1,np.minimum(o,c)-.1,c]))
    candidate=next(x for x in candidate_registry() if x.family==family)
    for path in (0,1,2):
        out,eq=run_array(a,candidate,fill_drop=.25,path_mode=path)
        assert abs(out['reconciliation_error'])<1e-7
        assert out['net_pnl']==pytest.approx(eq[-1]-1000)


def test_expected_shortfall_keeps_exact_mass_with_zero_ties():
    assert es20([-10]+[0]*19)==pytest.approx(-2.5)
    assert es20([-10,-5,0])==pytest.approx(-10)


def test_selection_ignores_future_rows_and_cash_is_not_success():
    starts=pd.date_range('2026-08-01',periods=12,tz='UTC')
    df=pd.DataFrame({'candidate':['a']*12,'family':['fixed']*12,'fills':[1]*12,
                     'net_pnl':[-1.]*12,'window_start':starts,'max_drawdown':[-.001]*12})
    assert choose(df) is None
    z=df.assign(net_pnl=0,fills=0)
    assert not summarize(z)['every_window_profitable']
    assert summarize(z)['zeros']==12
    assert choose(df.assign(net_pnl=1))=='a'


def test_features_are_strictly_lagged():
    n=180
    k=pd.DataFrame({'open_time':pd.date_range('2026-08-20',periods=n,freq='min',tz='UTC'),
                     'open':100.,'high':100.2,'low':99.8,'close':100.,'volume':10.})
    f=pd.DataFrame({'funding_time':pd.to_datetime([],utc=True),'funding_rate':[],'mark_price':[]})
    a=prepare(k,f)
    k.loc[100:,'close']=120;k.loc[100:,'high']=121;k.loc[100:,'volume']=10000
    b=prepare(k,f)
    assert np.allclose(a[:101,5:13],b[:101,5:13],equal_nan=True)
