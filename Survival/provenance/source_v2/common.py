"""Shared prediction conversion, baseline estimation and centrally applied metrics."""
import json, os, platform
from importlib import metadata
from pathlib import Path
import numpy as np

def data_configuration(cfg):
    keys=('settings','sample_sizes','replications','data_seed_start','training_seeds',
          'split','rho','weibull_shape','weibull_scale_coefficient','censoring_target',
          'calibration_n','calibration_seeds','coxnet','metrics',
          'required_python_versions','required_r_versions')
    return {key:cfg[key] for key in keys}

def versions(required):
    out = {name:metadata.version(name) for name in required}
    for name, expected in required.items():
        assert out[name] == expected, (name, out[name], expected)
    return {'python':platform.python_version(), 'packages':out,
            'package_locations':{name:str(metadata.distribution(name).locate_file('')) for name in required}}

def process_status():
    values = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in ('VmHWM','VmRSS','Threads'):
            values[key] = int(value.strip().split()[0])
    values['pid'] = os.getpid()
    return values

def step_matrix(times, survival, grid):
    times = np.asarray(times, dtype=float)
    assert np.all(np.diff(times) >= 0)
    survival = np.asarray(survival, dtype=float)
    idx = np.searchsorted(times, grid, side='right')-1
    out = np.ones((survival.shape[0], len(grid)))
    valid = idx >= 0
    out[:,valid] = survival[:,idx[valid]]
    return out

def partial_loglik(times, events, eta):
    order=np.argsort(-np.asarray(times),kind='stable')
    ts=np.asarray(times)[order]; es=np.asarray(events)[order].astype(bool); eta=np.asarray(eta)[order]
    ends=np.r_[np.flatnonzero(np.diff(ts)!=0)+1,len(ts)]
    group_end=np.repeat(ends,np.diff(np.r_[0,ends]))
    shift=eta.max(); risk=np.cumsum(np.exp(eta-shift))
    return float(np.sum(eta[es]-np.log(risk[group_end[es]-1])-shift))

def lambda_max(x, times, events):
    order=np.argsort(-np.asarray(times),kind='stable')
    xs=np.asarray(x)[order]; ts=np.asarray(times)[order]; es=np.asarray(events)[order].astype(bool)
    ends=np.r_[np.flatnonzero(np.diff(ts)!=0)+1,len(ts)]
    group_end=np.repeat(ends,np.diff(np.r_[0,ends]))
    risk_mean=np.cumsum(xs,axis=0)[group_end-1]/group_end[:,None]
    return float(np.abs((xs[es]-risk_mean[es]).sum(axis=0)).max()/len(xs))

def breslow(times, events, eta, grid):
    shift = float(np.max(eta))
    w = np.exp(np.asarray(eta)-shift)
    ts = np.unique(times[events.astype(bool)])
    jumps = np.array([np.sum((times==t)&events.astype(bool))/w[times>=t].sum() for t in ts])
    cumulative = np.cumsum(jumps)
    idx = np.searchsorted(ts, grid, side='right')-1
    h = np.zeros(len(grid)); h[idx>=0] = cumulative[idx[idx>=0]]
    return shift, h

def survival_from_lp(eta, baseline):
    shift, h = baseline
    return np.exp(-np.exp(np.asarray(eta)-shift)[:,None]*h[None,:])

def evaluate(train, test, surv, grid, tied_tol=1e-8):
    from sksurv.util import Surv
    from sksurv.nonparametric import CensoringDistributionEstimator
    from sksurv.metrics import concordance_index_censored
    assert surv.shape == (len(test), len(grid))
    assert np.isfinite(surv).all(), 'Non-finite survival prediction'
    assert surv.min() >= 0 and surv.max() <= 1, 'Survival outside [0,1]'
    assert (np.diff(surv,axis=1) <= 1e-10).all(), 'Increasing survival curve'
    y = Surv.from_arrays(train.event.astype(bool), train.time)
    censor = CensoringDistributionEstimator().fit(y)
    g_grid = censor.predict_proba(grid)
    assert (g_grid>0).all(), 'Zero censoring survival on integration grid'
    t = test.time.to_numpy(); e = test.event.to_numpy().astype(bool)
    used = e & (t<=grid[-1])
    g_y = np.ones(len(test)); g_y[used] = censor.predict_proba(t[used])
    assert (g_y[used]>0).all(), 'Zero censoring survival for an included event'
    scores = []
    for j, horizon in enumerate(grid):
        deaths = e & (t<=horizon); alive = t>horizon
        scores.append(float((np.sum(surv[deaths,j]**2/g_y[deaths])+
                            np.sum((1-surv[alive,j])**2/g_grid[j]))/len(test)))
    ibs = float(np.trapezoid(scores,grid)/(grid[-1]-grid[0]))
    risk = -np.trapezoid(surv,grid,axis=1)/(grid[-1]-grid[0])
    ci = concordance_index_censored(e,t,risk,tied_tol=tied_tol)
    assert np.isfinite(ibs) and np.isfinite(ci[0])
    return {'ibs':ibs,'cindex':float(ci[0]), 'n_test_evaluated':len(test),
            'concordant':int(ci[1]),'discordant':int(ci[2]),'tied_risk':int(ci[3]),
            'tied_time':int(ci[4]),'grid_lower':float(grid[0]),'grid_upper':float(grid[-1]),
            'min_censoring_survival':float(g_grid.min())}
