"""Evaluate ALE candidate code with official full public/private case sets."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import time
import resource
from docker_accounting import DockerAccounting

from settings import WORKSPACE as ROOT
runtime_cache=Path('/tmp')/f'ale-bench-runtime-{os.getuid()}'
runtime_cache.mkdir(exist_ok=True)
os.environ.update(ALE_BENCH_DATA=str(ROOT/'data/alebench'),ALE_BENCH_CACHE=str(runtime_cache))
import ale_bench
p=argparse.ArgumentParser();p.add_argument('task');p.add_argument('--run-dir',type=Path,required=True)
p.add_argument('--public-only',action='store_true')
p.add_argument('--submission-file',type=Path,required=True);a=p.parse_args()
out=a.run_dir.resolve();out.mkdir(parents=True,exist_ok=True)
if (out/'report.json').exists():raise RuntimeError('Preserving existing report; use a new run directory')
code=a.submission_file.read_text()
(out/'submission.cpp').write_text(code)
def cached_build(tool_dir):
    source=ROOT/'data/alebench/prepared'/a.task/'tools/target/release'
    destination=tool_dir/'target/release';destination.mkdir(parents=True,exist_ok=True)
    for name in ['gen','vis','tester']:
        if (tool_dir/'src/bin'/f'{name}.rs').exists():shutil.copy2(source/name,destination/name)
importlib.import_module('ale_bench.start').build_rust_tools=cached_build
start=time.monotonic();accounting=DockerAccounting();accounting.__enter__()
cpu_start=resource.getrusage(resource.RUSAGE_SELF)
session=None
report={'benchmark':'ale','task_id':a.task,'baseline':'fixed_trivial_solution','api_calls':0,
        'submission_sha256':hashlib.sha256(code.encode()).hexdigest(),'workers':4,
        'rust_tools':'previously built official tools','splits':{},'state':'running'}
report['runtime_cache']=str(runtime_cache)
if a.submission_file:report['baseline']='external_candidate'
try:
    session=ale_bench.start(a.task,lite_version=False,num_workers=4,run_visualization_server=False)
    for split in ['public'] if a.public_only else ['public','private']:
        begin=time.monotonic()
        if split=='public':result=session.public_eval(code,'cpp20',judge_version='202301',reuse_containers=True)
        else:
            result,rank,performance=session.private_eval(code,'cpp20',judge_version='202301',reuse_containers=True)
            report.update(rank=rank,performance=performance)
        (out/f'{split}.json').write_text(result.model_dump_json(indent=2))
        d=result.model_dump(mode='json');cases=d['case_results']
        report['splits'][split]={'status':d['overall_judge_result'],'score':d['overall_absolute_score'],
                                'cases':len(cases),'wall_seconds':time.monotonic()-begin,
                                'sum_case_execution_seconds':sum(c.get('execution_time') or 0 for c in cases),
                                'peak_case_memory_bytes':max((c.get('memory_usage') or 0 for c in cases),default=0)}
        print(a.task,split,d['overall_judge_result'],flush=True)
        if result.overall_judge_result.value!='ACCEPTED':
            report['state']='failed';break
    else:report['state']='public_verified' if a.public_only else 'completed'
except Exception as exc:
    report['state']='failed';report['error']=f'{type(exc).__name__}: {exc}'
    raise
finally:
    try:
        if session is not None:session.close()
    finally:accounting.__exit__(None,None,None)
    report['wall_seconds']=time.monotonic()-start
    report['resources']=accounting.report()
    cpu_end=resource.getrusage(resource.RUSAGE_SELF)
    report['resources'].update(controller_cpu_seconds=cpu_end.ru_utime+cpu_end.ru_stime-cpu_start.ru_utime-cpu_start.ru_stime,
                               controller_peak_rss_bytes=cpu_end.ru_maxrss*1024,
                               gpu_used=False)
    report['measurement_notes']=['Official per-case execution time has finite precision; zeros are not zero compute.',
                                 'Wall time includes setup, compilation and grading; cached Rust preparation excluded.',
                                 'Case memory does not include controller, compiler, or judge memory.']
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if report['state']=='failed':raise SystemExit(1)
