"""Account only for Docker containers created by this Python process.

Hooks are scoped to this process and restored when measurement ends. They do
not alter benchmark commands, limits, images, scoring, or other sessions.
"""
import threading
import time
from docker.models.containers import Container, ContainerCollection
from docker.errors import APIError

class DockerAccounting:
    def __init__(self):
        self.records={};self.handles={};self.lock=threading.Lock();self.errors=[]
        self.stop=threading.Event();self.peak_total_memory=0
    def sample(self,container):
        try:
            stats=container.stats(stream=False,one_shot=True)
            cpu=stats.get('cpu_stats',{}).get('cpu_usage',{}).get('total_usage',0)/1e9
            mem=stats.get('memory_stats',{});usage=mem.get('usage',0)
            with self.lock:
                r=self.records[container.id]
                r['cpu_seconds']=max(r['cpu_seconds'],cpu)
                r['peak_memory_bytes']=max(r['peak_memory_bytes'],usage,mem.get('max_usage',0))
                r['samples']+=1
            return usage
        except APIError as e:
            with self.lock:self.errors.append(type(e).__name__)
            return 0
    def __enter__(self):
        self.original_create=ContainerCollection.create;self.original_remove=Container.remove
        owner=self
        def create(collection,*args,**kwargs):
            c=owner.original_create(collection,*args,**kwargs)
            with owner.lock:
                owner.records[c.id]={'cpu_seconds':0,'peak_memory_bytes':0,'samples':0}
                owner.handles[c.id]=c
            return c
        def remove(c,*args,**kwargs):
            with owner.lock:known=c.id in owner.records
            if known:owner.sample(c)
            result=owner.original_remove(c,*args,**kwargs)
            with owner.lock:owner.handles.pop(c.id,None)
            return result
        ContainerCollection.create=create;Container.remove=remove
        def monitor():
            while not self.stop.is_set():
                with self.lock:handles=list(self.handles.values())
                total=sum(self.sample(c) for c in handles)
                self.peak_total_memory=max(self.peak_total_memory,total)
                self.stop.wait(.5)
        self.thread=threading.Thread(target=monitor,daemon=True);self.thread.start()
        return self
    def __exit__(self,*exc):
        self.stop.set();self.thread.join(timeout=30)
        ContainerCollection.create=self.original_create;Container.remove=self.original_remove
    def report(self):
        return {'containers':self.records,'container_cpu_seconds':sum(x['cpu_seconds'] for x in self.records.values()),
                'sampled_peak_total_container_memory_bytes':self.peak_total_memory,'sampling_errors':self.errors,
                'notes':['Container CPU is cumulative cgroup time, sampled before removal.',
                         'Concurrent memory samples are sequential and may miss short-lived peaks.',
                         'Memory includes container page cache. Controller memory is recorded separately.']}
