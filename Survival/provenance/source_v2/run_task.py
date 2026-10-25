"""Array-safe runner: one worker per fit, independent attempts and atomic success."""
import argparse, csv, datetime, fcntl, hashlib, json, os, signal, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from common import evaluate,data_configuration

HERE=Path(__file__).resolve().parent
RUNTIME_FILES=('config.json','common.py','fit_python.py','fit_r.R','run_task.py','prepare.py','make_splits.R')

def code_hash():
    h=hashlib.sha256()
    for name in RUNTIME_FILES: h.update(name.encode()); h.update((HERE/name).read_bytes())
    return h.hexdigest()

def atomic_json(path, value):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,indent=2)); tmp.replace(path)

def children(pid):
    found=[]
    try:
        for task in Path(f'/proc/{pid}/task').glob('*'):
            try: found.extend(int(p) for p in (task/'children').read_text().split())
            except FileNotFoundError: pass
    except FileNotFoundError: pass
    return sorted(set(found))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path); ap.add_argument('task_id',type=int)
    ap.add_argument('--phase',choices=['pilot','formal'],default='formal'); ap.add_argument('--timeout',type=int,default=3600)
    a=ap.parse_args(); root=a.root.resolve(); cfg=json.loads((HERE/'config.json').read_text())
    with (root/'tasks.csv').open() as f: rows=list(csv.DictReader(f))
    row=rows[a.task_id]
    for name in ('task_id','n','p','replication','data_seed','training_seed'): row[name]=int(row[name])
    assert row['task_id']==a.task_id
    if (root/'RUN_SCOPE.json').exists():
        assert row['model'] in json.loads((root/'RUN_SCOPE.json').read_text())['models']
    fingerprint=code_hash()
    if a.phase=='formal':
        passed=json.loads((root/'PILOT_VALIDATED.json').read_text())
        assert passed['code_sha256']==fingerprint, 'Formal code has changed since pilot validation'
    data=root/'data'/row['dataset_id']; metadata=json.loads((data/'metadata.json').read_text())
    snapshot=root/'data_config.json'
    data_cfg=json.loads(snapshot.read_text()) if snapshot.exists() else cfg
    assert data_configuration(data_cfg)==data_configuration(cfg), 'Data-generating configuration changed'
    data_hash=hashlib.sha256((snapshot if snapshot.exists() else HERE/'config.json').read_bytes()).hexdigest()
    assert metadata['config_sha256']==data_hash
    config_hash=hashlib.sha256((HERE/'config.json').read_bytes()).hexdigest()
    for name,expected in metadata['files_sha256'].items():
        assert hashlib.sha256((data/name).read_bytes()).hexdigest()==expected, f'Data changed: {name}'
    run=root/a.phase/f"{a.task_id:04d}_{row['dataset_id']}_{row['model']}_{row['language']}"
    run.mkdir(parents=True,exist_ok=True)
    lock=(run/'lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (run/'result.json').exists():
        old=json.loads((run/'result.json').read_text())
        if old['status']=='success':
            if old['code_sha256']==fingerprint:
                print('SKIP verified success',a.task_id); return
            assert a.phase=='pilot', 'Successful formal result belongs to different code'
    attempt=run/f"attempt_{len(list(run.glob('attempt_*')))+1:03d}"; attempt.mkdir()
    request={**row,'data_dir':str(data),'output_dir':str(attempt)}
    atomic_json(attempt/'request.json',request)
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','RCPP_PARALLEL_NUM_THREADS'):
        env[key]='1'
    env.update(OMP_DYNAMIC='FALSE',CUDA_VISIBLE_DEVICES='',RETICULATE_PYTHON=cfg['deep_python'],
               PYTHONHASHSEED=str(row['training_seed']),RF_CORES='1')
    executable=cfg['deep_python'] if row['model']=='DeepSurv' else cfg['python']
    cmd=(['Rscript',str(HERE/'fit_r.R')] if row['language']=='R' else
         [executable,str(HERE/'fit_python.py')])+[str(attempt/'request.json')]
    result={**row,'phase':a.phase,'status':'failed','code_sha256':fingerprint,
            'config_sha256':config_hash,'data_config_sha256':data_hash,'data_files_sha256':metadata['files_sha256'],
            'attempt':attempt.name,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'hostname':os.uname().nodename,'slurm_job_id':env.get('SLURM_JOB_ID'),
            'slurm_array_job_id':env.get('SLURM_ARRAY_JOB_ID'),
            'cpu_affinity':sorted(os.sched_getaffinity(0)),
            'cpu_model':next(l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),
            'requested_cpu_cores':env.get('SLURM_CPUS_PER_TASK'),
            'requested_memory_mb':env.get('SLURM_MEM_PER_NODE'),
            'thread_environment':{k:env[k] for k in env if k.endswith('NUM_THREADS') or k=='RF_CORES'}}
    observed_children={}; start=time.monotonic(); proc=None
    try:
        assert 'Gold 6154' in result['cpu_model'], 'Unexpected processor model'
        with (attempt/'stdout.log').open('w') as stdout,(attempt/'stderr.log').open('w') as stderr:
            proc=subprocess.Popen(cmd,cwd=attempt,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
            while proc.poll() is None:
                for pid in children(proc.pid):
                    try:
                        observed_children[str(pid)]={'command':Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\x00',b' ').decode(errors='replace'),
                            'during_fit':(attempt/'fit_active').exists() or observed_children.get(str(pid),{}).get('during_fit',False)}
                    except FileNotFoundError: pass
                if time.monotonic()-start>a.timeout:
                    os.killpg(proc.pid,signal.SIGKILL); proc.wait(); raise TimeoutError(f'Worker exceeded {a.timeout} seconds')
                time.sleep(.15)
        if proc.returncode: raise RuntimeError(f'Worker exit code {proc.returncode}: '+(attempt/'stderr.log').read_text()[-5000:])
        fit=json.loads((attempt/'fit.json').read_text())
        assert not any(v['during_fit'] for v in observed_children.values()), 'Child process during fitting: Peak RSS coverage requires inspection'
        assert fit['fitting_time_s']>0 and fit['peak_rss_kib']>0
        pred=np.loadtxt(attempt/'survival.csv',delimiter=',')
        train=pd.read_csv(data/'train.csv',float_precision='round_trip')
        test=pd.read_csv(data/'test.csv',float_precision='round_trip')
        assert np.array_equal(np.loadtxt(attempt/'test_row_ids.csv',dtype=int),test.row_id)
        metrics=evaluate(train,test,pred,np.loadtxt(data/'grid.csv'),cfg['metrics']['cindex_tied_tol'])
        result.update(fit); result.update(metrics); result['status']='success'
        result['warnings']=(attempt/'stderr.log').read_text()
        if any(s in result['warnings'].lower() for s in ('did not converge','convergencewarning','ran out of iterations','infinite coefficient')):
            raise RuntimeError('Convergence warning requires inspection')
    except Exception as exc:
        result['status']='failed'; result['error']=str(exc); result['traceback']=traceback.format_exc()
    result['worker_total_wall_s']=time.monotonic()-start
    result['observed_children']=observed_children
    result['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    atomic_json(attempt/'result.json',result); atomic_json(run/'result.json',result)
    print(json.dumps({k:result.get(k) for k in ('task_id','model','language','status','fitting_time_s','peak_rss_mib','error')}),flush=True)
    if result['status']!='success': sys.exit(1)

if __name__=='__main__': main()
