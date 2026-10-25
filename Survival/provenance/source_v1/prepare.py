"""Generate the 600 datasets once. Never called by fitting array elements."""
import argparse, hashlib, json, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def covariates(rng, n, p, rho):
    x = rng.normal(size=(n, p))
    for j in range(1, p):
        x[:, j] = rho*x[:, j-1] + np.sqrt(1-rho*rho)*x[:, j]
    return x

def event_times(rng, x, setting, cfg):
    if setting == 'highdim':
        b = (-1.0)**np.arange(50) * 0.93**np.arange(50)
        eta = x[:, :50] @ b
    else:
        eta = (.4*x[:,0]+.8*np.sin(x[:,1])+.8*(x[:,2]**2-1)
               +.7*np.tanh(x[:,3]+x[:,4])+.5*np.exp(-x[:,5]**2))
    eta = (eta-eta.mean())/eta.std(ddof=0)
    return (-np.log(rng.uniform(size=len(x)))/
            (cfg['weibull_scale_coefficient']*np.exp(eta)))**(1/cfg['weibull_shape'])

def calibrate(cfg):
    out = {}
    for setting in cfg['settings']:
        name = setting['name']
        rng = np.random.default_rng(cfg['calibration_seeds'][name])
        x = covariates(rng, cfg['calibration_n'], setting['p'], cfg['rho'])
        t = event_times(rng, x, name, cfg)
        lo, hi = 0., float(t.max()/cfg['censoring_target'])
        for _ in range(70):
            mid = (lo+hi)/2
            if np.minimum(t/mid, 1).mean() > cfg['censoring_target']:
                lo = mid
            else:
                hi = mid
        out[name] = {'uniform_upper': (lo+hi)/2, 'seed': cfg['calibration_seeds'][name],
                     'n': cfg['calibration_n'], 'expected_rate_on_calibration_sample':
                     float(np.minimum(t/((lo+hi)/2), 1).mean())}
    return out

def write_csv(frame, path):
    frame.to_csv(path, index=False, float_format='%.17g')

def penalty_grid(x, time, event, cfg):
    # Score of the average Breslow partial log likelihood at beta = 0.
    # Generated event times are continuous; grouping also handles exact ties.
    score = np.zeros(x.shape[1])
    for t in np.unique(time[event.astype(bool)]):
        deaths = (time == t) & event.astype(bool)
        score += x[deaths].sum(axis=0) - deaths.sum()*x[time >= t].mean(axis=0)
    maximum = np.abs(score).max()/len(x)
    return np.geomspace(maximum, maximum*cfg['coxnet']['min_ratio'], cfg['coxnet']['penalties'])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    a = parser.parse_args()
    root = a.root.resolve(); root.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((HERE/'config.json').read_text())
    cfg_hash = sha(HERE/'config.json')
    if (root/'READY.json').exists():
        ready = json.loads((root/'READY.json').read_text())
        assert ready['config_sha256'] == cfg_hash
        print('Existing complete data retained:', ready); return
    rows = []
    for s in cfg['settings']:
        for n in cfg['sample_sizes']:
            for rep in range(1, cfg['replications']+1):
                rows.append(dict(dataset_id=f"{s['name']}_n{n}_r{rep:03d}",
                                 setting=s['name'], n=n, p=s['p'], replication=rep,
                                 data_seed=cfg['data_seed_start']+len(rows),
                                 training_seed=cfg['training_seeds'][(rep-1)%5]))
    manifest = pd.DataFrame(rows)
    write_csv(manifest, root/'datasets.csv')
    splitdir = root/'split_indices'
    subprocess.run(['Rscript', str(HERE/'make_splits.R'), str(root/'datasets.csv'), str(splitdir)], check=True)
    cp = root/'censoring_calibration.json'
    cal = json.loads(cp.read_text()) if cp.exists() else calibrate(cfg)
    cp.write_text(json.dumps(cal, indent=2))
    tasks = []
    for row in rows:
        d = root/'data'/row['dataset_id']; d.mkdir(parents=True, exist_ok=True)
        meta = d/'metadata.json'
        if not meta.exists():
            rng = np.random.default_rng(row['data_seed'])
            n, p = row['n'], row['p']
            x = covariates(rng, n, p, cfg['rho'])
            t = event_times(rng, x, row['setting'], cfg)
            c = rng.uniform(0, cal[row['setting']]['uniform_upper'], n)
            frame = pd.DataFrame(x, columns=[f'x{j+1}' for j in range(p)])
            frame.insert(0, 'row_id', np.arange(n))
            frame['time'] = np.minimum(t,c); frame['event'] = (t <= c).astype(int)
            order = rng.permutation(n)
            pool_idx, test_idx = order[:int(n*.8)], order[int(n*.8):]
            val_pos = np.loadtxt(splitdir/(row['dataset_id']+'.txt'), dtype=int)-1
            train_pos = np.flatnonzero(~np.isin(np.arange(len(pool_idx)), val_pos))
            selections = dict(pool=pool_idx, train=pool_idx[train_pos], val=pool_idx[val_pos], test=test_idx)
            for name, idx in selections.items():
                write_csv(frame.iloc[idx], d/(name+'.csv'))
            tr = frame.iloc[selections['train']]
            features = [f'x{j+1}' for j in range(p)]
            means = tr[features].to_numpy().mean(axis=0)
            scales = tr[features].to_numpy().std(axis=0, ddof=0)
            assert (scales > 0).all()
            write_csv(pd.DataFrame({'mean':means,'scale':scales}), d/'scaling.csv')
            grid = np.linspace(*np.quantile(tr.time, cfg['metrics']['train_time_quantiles']), cfg['metrics']['grid_points'])
            np.savetxt(d/'grid.csv', grid, fmt='%.17g')
            if row['setting'] == 'highdim':
                alphas = penalty_grid((tr[features].to_numpy()-means)/scales,
                                      tr.time.to_numpy(), tr.event.to_numpy(), cfg)
                np.savetxt(d/'penalties.csv', alphas, fmt='%.17g')
            # Explicit row-level split record, including native R validation order.
            records = []
            for name in ('train','val','test'):
                records += [dict(row_id=int(idx), split=name, order=k) for k,idx in enumerate(selections[name])]
            write_csv(pd.DataFrame(records), d/'splits.csv')
            details = {**row, 'config_sha256':cfg_hash,
                       'censoring_rate':float(1-frame.event.mean()),
                       'uniform_censoring_upper':cal[row['setting']]['uniform_upper'],
                       'n_train':len(train_pos), 'n_val':len(val_pos), 'n_test':len(test_idx),
                       'files_sha256':{f.name:sha(f) for f in sorted(d.glob('*.csv'))}}
            meta.write_text(json.dumps(details,indent=2))
        else:
            details = json.loads(meta.read_text())
            assert details['config_sha256'] == cfg_hash
            assert all(sha(d/name) == value for name,value in details['files_sha256'].items())
        models = ['Coxnet' if row['setting']=='highdim' else 'CoxPH', 'RSF','Boosting','DeepSurv']
        for model in models:
            for language in ('R','Python'):
                tasks.append({**row, 'task_id':len(tasks), 'model':model, 'language':language})
        print(row['dataset_id'], flush=True)
    write_csv(pd.DataFrame(tasks), root/'tasks.csv')
    ready = {'config_sha256':cfg_hash, 'datasets':len(rows), 'fits':len(tasks),
             'datasets_sha256':sha(root/'datasets.csv'), 'tasks_sha256':sha(root/'tasks.csv')}
    (root/'READY.json').write_text(json.dumps(ready,indent=2))
    print(ready)

if __name__ == '__main__': main()
