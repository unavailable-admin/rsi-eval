"""Process-tree measurements for trusted fixed baselines, without an LLM."""
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
import psutil

def measure(command, cwd, env, output, timeout=14400, cpu_ids=None, gpu=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    before=resource.getrusage(resource.RUSAGE_CHILDREN)
    started=time.monotonic();peak_rss=0;peak_vram=0;timed_out=False;samples=0
    if cpu_ids:
        command=['taskset','-c',','.join(map(str,cpu_ids)),*command]
    with (output/'stdout.log').open('w') as log:
        proc=subprocess.Popen(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        parent=psutil.Process(proc.pid)
        while proc.poll() is None:
            pids={proc.pid};rss=0
            try:children=[parent]+parent.children(recursive=True)
            except psutil.Error:children=[]
            for child in children:
                try:
                    pids.add(child.pid);rss+=child.memory_info().rss
                except psutil.Error:pass
            peak_rss=max(peak_rss,rss)
            if gpu is not None:
                query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
                used=0
                for line in query.stdout.splitlines():
                    try:
                        pid,mem=line.split(',')
                        if int(pid) in pids:used+=int(mem)*1024**2
                    except ValueError:pass
                peak_vram=max(peak_vram,used)
            samples+=1
            if time.monotonic()-started>timeout:
                timed_out=True
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                break
            time.sleep(1)
        exit_code=proc.wait()
    elapsed=time.monotonic()-started;after=resource.getrusage(resource.RUSAGE_CHILDREN)
    result={'command':command,'cwd':str(cwd),'exit_code':exit_code,'timed_out':timed_out,
            'wall_seconds':elapsed,'cpu_user_seconds':after.ru_utime-before.ru_utime,
            'cpu_system_seconds':after.ru_stime-before.ru_stime,
            'sampled_peak_tree_rss_bytes':peak_rss,'sampled_peak_process_vram_bytes':peak_vram,
            'gpu':gpu,'gpu_allocated_seconds':elapsed if gpu is not None else 0,
            'cpu_affinity':cpu_ids,'sample_interval_seconds':1,'samples':samples,
            'measurement_notes':['RSS sums process resident sets and can double-count shared pages.',
                                 'Sampled peaks can miss short transients; GPU time is allocation time, not utilization.',
                                 'CPU child usage includes small nvidia-smi sampling overhead.']}
    (output/'resources.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
