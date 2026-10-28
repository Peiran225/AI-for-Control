"""One timing replay per original feedback stage; no result replacement."""
import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/home/mzhang/AI-for-Control-direct-guided-20260727');OUT=Path('/home/mzhang/pmp_rebuttal_sheet_20261027')

def main():
    (OUT/'timing_replays').mkdir(exist_ok=True)
    line={**json.loads((OUT/'evidence/lineage.json').read_text())['records'],**json.loads((OUT/'evidence/s6_cf_lineage.json').read_text())['records']}
    jobs=[]
    for path,record in line.items():
        if 'nonuniform' in path or 'feedback_' not in path:continue
        m=record['metadata'];ref=m.get('refinement',{})
        if not ref or 'initialization_checkpoint' not in m:continue
        method=m.get('args',{}).get('option');mult=ref.get('multiplier')
        if method not in ('cf','der') or mult not in (2,4,8):continue
        count=record.get('history_count',0)-1
        if count not in (2,4,6,8,10):raise RuntimeError('unverified original budget '+path)
        initial=count==2
        inner=2 if initial or (method=='cf' and mult==8) else 3
        name=f'{method}_m{mult}_outer{count}'
        dest=OUT/'timing_replays'/name
        args=['--checkpoint',m['initialization_checkpoint'],'--out-dir',str(dest),'--option',method,
              '--trajectory-mode','continuous-policy','--continuous-collocation','nodes','--device','cuda',
              '--threads','4','--query-batch-size','16' if method=='cf' else '128','--interval-start','1.5',
              '--interval-end','8','--no-refresh-nominal-reference','--exclude-nominal-from-training',
              '--train-random-states','0','--validation-random-states','0','--train-scope','all',
              '--lr','1' if method=='cf' else '.5','--selection-metric','scalar','--residual-p','2',
              '--report-scale-factor','400','--multiplier',str(mult),'--outer-steps',str(count),
              '--inner-iterations',str(inner),'--max-eval',str(2*inner),
              '--w0',str(ref['residual_weights'][0]),'--w1',str(ref['residual_weights'][1]),'--w2',str(ref['residual_weights'][2])]
        if method=='cf':args+=['--cf-weight','.01','--cf-scalar-weight','1','--control-trust-weight','0']
        if method=='cf' or not initial:args+=['--reference-multiplier','16','--reference-method','dop853']
        jobs.append({'name':name,'method':method,'original_checkpoint':path,'args':args,'budget':count,'multiplier':mult,'out':str(dest)})
    assert len(jobs)==8,len(jobs)
    # Long stages first, with one process per explicitly available GPU.
    jobs.sort(key=lambda x:x['budget']*x['multiplier'],reverse=True)
    queues=[jobs[i::3] for i in range(3)]
    def worker(gpu,queue):
        for job in queue:
            report=OUT/'evidence'/f'timing_{job["name"]}.json'
            if report.exists():continue
            cmd=[sys.executable,'-u',str(ROOT/'scripts/refine_feedback_offgrid_scalar.py'),*job['args']]
            start=time.perf_counter()
            with (OUT/'logs'/f'timing_{job["name"]}.log').open('w') as log:
                rc=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'4'}).returncode
            record={**job,'gpu':gpu,'seconds':time.perf_counter()-start,'returncode':rc,'status':'completed' if rc==0 else 'FAILED',
                    'source':'new timing measurement, original stage input and recorded configuration; not historical wall time',
                    'replaces_submitted_checkpoint':False,'command':cmd}
            report.write_text(json.dumps(record,indent=2)+'\n');print(job['name'],record['status'],record['seconds'],flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        fs=[ex.submit(worker,5+i,q) for i,q in enumerate(queues)]
        for f in fs:f.result()

if __name__=='__main__':main()
