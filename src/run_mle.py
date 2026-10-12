"""Execute an MLE adapter through the common candidate protocol, then official grade."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from bench_container import measure_container
from bench_resources import measure

from settings import WORKSPACE as ROOT, SOURCE
p=argparse.ArgumentParser();p.add_argument('task');p.add_argument('--candidate',type=Path,required=True)
p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--gpu',type=int)
p.add_argument('--timeout',type=int,default=3600);p.add_argument('--seed',type=int,default=0)
p.add_argument('--validation-only',action='store_true')
a=p.parse_args()
if a.task not in (ROOT/'projects/mle-bench/experiments/splits/low.txt').read_text().split():p.error('Unknown task')
out=a.run_dir.resolve()
if out.exists():p.error('Use a fresh run directory')
out.mkdir(parents=True);work=out/'output';work.mkdir()
script=out/'agent.py';shutil.copy2(a.candidate,script)
spec={'schema_version':1,'benchmark':'mle','task_id':a.task,'competition_id':a.task,'seed':a.seed,
      'public_data_dir':'/data','description_path':'/data/description.md','output_dir':'/output',
      'budget':{'wall_seconds':a.timeout,'cpu_cores':8,'memory_gib':32,'gpu_count':int(a.gpu is not None)}}
(out/'task.json').write_text(json.dumps(spec,indent=2)+'\n')
mounts=[]
for source,target,readonly in [(ROOT/'envs/mle-gpu',ROOT/'envs/mle-gpu',True),
 (ROOT/'tools/python',ROOT/'tools/python',True),(ROOT/'data/mlebench'/a.task/'prepared/public','/data',True),
 (script,'/agent.py',True),(out/'task.json','/task.json',True),(work,'/output',False)]:
 mounts.append({'Source':str(source),'Target':str(target),'Type':'bind','ReadOnly':readonly})
env={'CONDA_PREFIX':str(ROOT/'envs/mle-gpu'),'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'}
lease=None
if a.gpu is not None:
 (ROOT/'runs/gpu-leases').mkdir(parents=True,exist_ok=True)
 lease=(ROOT/f'runs/gpu-leases/gpu-{a.gpu}.lock').open('a+')
 fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 used=int(subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
 if used>512:raise RuntimeError('GPU occupied; preserving other work')
try:
 result=measure_container([str(ROOT/'envs/mle-gpu/bin/python'),'/agent.py','--task','/task.json'],
       '/output',env,out/'execution',mounts,a.timeout,sorted(os.sched_getaffinity(0))[:8],a.gpu)
finally:
 if lease:lease.close()
submission=work/'submission.csv'
if result['exit_code'] or result['timed_out'] or not submission.is_file() or submission.is_symlink():
 error='Candidate execution failed or submission is missing/unsafe; see execution logs'
 (out/'report.json').write_text(json.dumps({'schema_version':1,'benchmark':'mle','task_id':a.task,
     'state':'failed','resources':result,'error':error,'candidate_sha256':hashlib.sha256(script.read_bytes()).hexdigest()},indent=2)+'\n')
 raise RuntimeError(error)
if a.validation_only:
 candidate_result=work/'candidate-result.json'
 if candidate_result.is_symlink():raise ValueError('Unsafe candidate result path')
 report={'schema_version':1,'benchmark':'mle','task_id':a.task,'state':'candidate_executed',
         'resources':result,'candidate_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),
         'validation':json.loads(candidate_result.read_text()) if candidate_result.exists() else None,
         'grading':None,'note':'Validation is adapter-reported on public data; no private grading performed.'}
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 raise SystemExit(0)
entries=out/'submissions.jsonl';entries.write_text(json.dumps({'competition_id':a.task,'submission_path':str(submission)})+'\n')
grading_resources=measure([str(ROOT/'envs/mlebench/bin/python'),str(SOURCE/'mle_grade.py'),'grade',
   '--submission',str(entries),'--output-dir',str(out/'grading'),'--data-dir',str(ROOT/'data/mlebench')],
   ROOT,os.environ.copy(),out/'grading-resources',timeout=a.timeout,cpu_ids=sorted(os.sched_getaffinity(0))[:8])
if grading_resources['exit_code'] or grading_resources['timed_out']:raise RuntimeError('Official grading failed')
grading=json.loads(next((out/'grading').glob('*_grading_report.json')).read_text())
report={'schema_version':1,'benchmark':'mle','task_id':a.task,'state':'completed' if grading['total_valid_submissions']==1 else 'failed',
        'candidate_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'resources':result,'grading':grading,
        'grading_resources':grading_resources}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if report['state']!='completed':raise SystemExit(1)
