"""Run only frozen evaluations of the nine original Table-1 checkpoints."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/home/mzhang/AI-for-Control-direct-guided-20260727')
OUT=Path('/home/mzhang/pmp_rebuttal_sheet_20261027')
MANIFEST=OUT/'inputs/source_checkpoint_hashes.json'

def group(method,records):
    for rec in records:
        source=Path(rec['checkpoint_path']);label=rec['replicate']
        if not source.exists():
            (OUT/'evidence'/f'{method}_{label}_missing.json').write_text(json.dumps({'status':'NOT FOUND','source':str(source)}))
            continue
        actual=hashlib.sha256(source.read_bytes()).hexdigest()
        if actual!=rec['declared_sha256']:raise RuntimeError('checkpoint hash mismatch '+str(source))
        cache=OUT/'inputs'/f'{method}_cache.npz' if label=='base4_fb31_locked' else OUT/'inputs/exact_caches'/f'{method}_{label}.npz'
        tasks=[('fixed',25600)]
        if label=='base4_fb31_locked':tasks+=[('fixed',51200),('heldout',25600)]
        for cohort,steps in tasks:
            dest=OUT/('frozen' if label=='base4_fb31_locked' else 'frozen_cached')/f'{method}_{label}_{cohort}_{steps}'
            if (dest/'summary.json').exists():continue
            cmd=[sys.executable,'-u',str(OUT/'code/evaluate_frozen_s4.py'),'--source-root',str(ROOT),
                '--checkpoint',str(source),'--method',method,'--out',str(dest),'--steps',str(steps),
                '--batch-size','16','--threads','1']
            if cache is not None:cmd+=['--cache',str(cache)]
            if cohort=='heldout':cmd+=['--directions',str(OUT/'inputs/directions.npz')]
            start=time.perf_counter()
            with (OUT/'logs'/f'{dest.name}.log').open('w') as log:
                status=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}).returncode
            record={'command':cmd,'status':'completed' if status==0 else 'FAILED','returncode':status,'seconds':time.perf_counter()-start}
            (OUT/'evidence'/f'{dest.name}_job.json').write_text(json.dumps(record,indent=2)+'\n')
            print(dest.name,record['status'],flush=True)
            if status!=0:continue
            if cache is None:cache=dest/'policy_cache.npz'

def main():
    (OUT/'logs').mkdir(exist_ok=True);(OUT/'frozen').mkdir(exist_ok=True);(OUT/'frozen_cached').mkdir(exist_ok=True)
    records=json.loads(MANIFEST.read_text())['checkpoints']
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        futures=[ex.submit(group,m,[r for r in records if r['method']==m]) for m in ('time','cf','der')]
        for f in futures:f.result()

if __name__=='__main__':main()
