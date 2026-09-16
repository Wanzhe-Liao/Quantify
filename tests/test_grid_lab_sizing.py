"""Inventory sizing must respond after initial fills, not just at startup."""
from dataclasses import replace
import numpy as np
from src.grid_lab import Candidate, candidate_registry, run_array


def test_inventory_taper_changes_size_after_inventory_builds():
    rows=[(100,100.1,99.4,99.6),(99.6,99.7,98.9,99.1),
          (99.1,99.2,98.4,98.6),(98.6,98.7,97.9,98.1),
          (98.1,98.2,98,98.1)]
    a=np.zeros((len(rows),14))
    a[:,:4]=rows
    a[:,4]=100000
    a[:,5]=np.r_[100.,a[:-1,3]]
    a[:,6:9]=100.
    a[:,9]=.3
    a[:,10]=.001
    fixed=Candidate('test','fixed',spacing=.005,levels=4,direction=1)
    tapered=replace(fixed,taper=1,refresh=1)
    r1,_=run_array(a,fixed)
    r2,_=run_array(a,tapered)
    assert r2['turnover'] < r1['turnover']
    registered=next(c for c in candidate_registry() if c.family=='inventory_taper')
    assert registered.refresh==1
