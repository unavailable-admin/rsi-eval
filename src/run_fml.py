"""Evaluate FML candidate files in a fresh isolated copy with resource accounting."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from settings import WORKSPACE as ROOT
REPO=ROOT/'projects/fml-bench'
parser=argparse.ArgumentParser()
parser.add_argument('task')
parser.add_argument('--run-dir',type=Path,required=True)
parser.add_argument('--gpu',type=int)
parser.add_argument('--timeout',type=int,default=14400)

parser.add_argument('--candidate-dir',type=Path,required=True)
parser.add_argument('--validation-only',action='store_true')
args=parser.parse_args()
out=args.run_dir.resolve();out.mkdir(parents=True,exist_ok=True)
if (out/'report.json').exists():raise RuntimeError('Use a fresh run directory; previous report preserved')
config_path=REPO/'ml_tasks'/args.task/'config.json'
config=json.loads(config_path.read_text())
source=REPO/config['repo_dir'];sandbox=out/'sandbox'
repo=sandbox/config['repo_dir'];repo.parent.mkdir(parents=True,exist_ok=True)
if repo.exists():raise RuntimeError('Use a fresh run directory; existing sandbox preserved')
shutil.copytree(source,repo,symlinks=True,ignore=shutil.ignore_patterns('.git','__pycache__','results_tmp','model_checkpoint'))
candidate=args.candidate_dir.resolve()
if not candidate.is_dir():raise ValueError('Candidate directory is missing')
copied=[]
for name in config['target_files']:
    path=(candidate/name).resolve()
    if not path.is_relative_to(candidate):raise ValueError('Candidate path escapes its directory')
    if path.is_file():
        shutil.copy2(path,repo/name);copied.append(name)
if not copied:raise ValueError('Candidate has none of the allowed target files')
(sandbox/'ml_tasks').symlink_to(REPO/'ml_tasks',target_is_directory=True)
if (source.parent/'data').exists():
    (repo.parent/'data').symlink_to(source.parent/'data',target_is_directory=True)
env=os.environ.copy();prefix=ROOT/'envs/fml'/config['conda_env']
env.update(PATH=str(prefix/'bin')+os.pathsep+env.get('PATH',''),CONDA_PREFIX=str(prefix),
           OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
           SCIKIT_LEARN_DATA=str(ROOT/'data/fmlbench/sklearn'),HF_HOME=str(ROOT/'cache/huggingface'),
           TORCH_HOME=str(ROOT/'cache/torch'),PYTHONUNBUFFERED='1',
           CUDA_VISIBLE_DEVICES=str(args.gpu) if args.gpu is not None else '')
available_cpus=sorted(os.sched_getaffinity(0))
offset=((args.gpu+1)*8) if args.gpu is not None else 0
cpu_ids=available_cpus[offset:offset+8] or available_cpus[:8]
report={'benchmark':'fml','task_id':args.task,'baseline':'external_candidate','api_calls':0,
        'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'repository_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'execution':'isolated_container','environment':str(prefix),'splits':{},'state':'running'}
report['target_sha256']={name:hashlib.sha256((repo/name).read_bytes()).hexdigest()
                         for name in config['target_files'] if (repo/name).is_file()}
mounts=[]
from bench_container import measure_container
# Old PyTorch builds need the glibc generation used by the host environment.
if config['conda_env'] in ['domainbed','continual_learning','lightly']:
    env['BENCH_CONTAINER_IMAGE']='ubuntu:22.04'
if args.task=='Generalization_domainbed_officehome':
    env['BENCH_SHM_SIZE']='16G'
for path,readonly in [(sandbox,False),(prefix,True),(REPO/'ml_tasks'/args.task,True),
                      (source.parent,True),(ROOT/'data/fmlbench',True),(ROOT/'cache/torch',True)]:
    if path.exists():mounts.append({'Source':str(path),'Target':str(path),'Type':'bind','ReadOnly':readonly})
hf=sandbox/'hf-cache';hf.mkdir()
for name in ['datasets','modules']:
    if (ROOT/'cache/huggingface'/name).exists():
        shutil.copytree(ROOT/'cache/huggingface'/name,hf/name,symlinks=True)
hub=ROOT/'cache/huggingface/hub'
(hf/'hub').symlink_to(hub,target_is_directory=True)
mounts.append({'Source':str(hub),'Target':str(hub),'Type':'bind','ReadOnly':True})
env.update(HF_HOME=str(hf),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
if config['conda_env']=='art':
    art=sandbox/'art-cache';art.mkdir()
    cached=ROOT/'data/fmlbench/art'
    # ART ignores a read-only cache and attempts downloading; give each run a copy.
    shutil.copytree(cached,art/'data')
    (art/'config.json').write_text(json.dumps({'ART_DATA_PATH':str(art/'data')}))
    mounts.append({'Source':str(art),'Target':'/root/.art','Type':'bind','ReadOnly':False})
lease=None
if args.gpu is not None:
    leases=ROOT/'runs/gpu-leases';leases.mkdir(parents=True,exist_ok=True)
    lease=(leases/f'gpu-{args.gpu}.lock').open('a+')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    used=int(subprocess.check_output(['nvidia-smi',f'--id={args.gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    if used>512:raise RuntimeError(f'GPU {args.gpu} already has {used} MiB allocated; preserving other work')
try:
    for split in ['val'] if args.validation_only else ['val','test']:
        result=measure_container(['bash','-c',config[f'{split}_command']],repo,env,out/split,mounts,args.timeout,cpu_ids,args.gpu)
        metrics=repo/'results_tmp'/f'{split}_info.json'
        if metrics.exists() and (metrics.is_symlink() or not metrics.resolve().is_relative_to(sandbox.resolve())):
            raise ValueError('Metric artifact escapes isolated workspace')
        result['metrics']=json.loads(metrics.read_text()) if metrics.exists() else None
        result['valid_result']=result['exit_code']==0 and not result['timed_out'] and result['metrics'] is not None
        report['splits'][split]=result
        if not result['valid_result']:report['state']='failed'
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(args.task,split,result['exit_code'],result['valid_result'],flush=True)
        if not result['valid_result']:break
    else:report['state']='validated' if args.validation_only else 'completed'
    report['sandbox_allocated_bytes']=int(subprocess.check_output(['du','-s','-B1',str(sandbox)],text=True).split()[0])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
except Exception as exc:
    report['state']='failed';report['error']=f'{type(exc).__name__}: {exc}'
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    raise
finally:
    if lease:lease.close()
if report['state'] not in ['completed','validated']:raise SystemExit(1)
