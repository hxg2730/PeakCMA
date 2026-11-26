"""Small read-only status probe for an unattended full experiment."""
import json
from pathlib import Path
import subprocess
import time

root=Path('Results/weather_full')
def read(path):
    try: return json.loads(path.read_text())
    except (OSError,ValueError): return None

status=read(root/'status.json') or {}
configuration=read(root/'configuration.json') or {}
progress=read(root/'cache_progress.json')
result={'status':status.get('status'),'stage':status.get('stage')}
if status.get('error'): result['error']=status['error']
if progress:
    counts=configuration.get('window_counts',{})
    keys=['train','val','test']; split=progress['split']
    previous=sum(counts.get(k,0) for k in keys[:keys.index(split)])
    total=sum(counts.values())
    done=previous+progress['completed']
    result['cache']={'split':split,'completed':progress['completed'],'split_total':progress['total'],
                     'global_completed':done,'global_total':total,'percent':round(100*done/max(1,total),2)}
    if progress['generated_this_run']:
        seconds_per_row=progress['seconds']/progress['generated_this_run']
        result['cache']['estimated_remaining_minutes']=round((total-done)*seconds_per_row/60,1)
stage=status.get('stage','')
if stage.endswith(('_baseline','_peak')):
    seed,mode=stage.split('_')
    h=read(root/f'training/Weather/96_192/{seed}/{mode}/history.json')
    if h:
        elapsed=max(0,time.time()-status.get('started_unix',time.time()))
        restart=read(root/'ram_restart.json') or {}
        resumed_from=restart.get('resume_epoch',0) if restart.get('new_driver')==status.get('pid') and stage=='2024_baseline' else 0
        epochs_completed=max(1,h[-1]['epoch']-resumed_from)
        result['training']={'epoch':h[-1]['epoch'],'train':h[-1]['train'],'validation':h[-1]['validation'],
                            'best_validation_mse':min(v['validation']['mse'] for v in h),
                            'stage_elapsed_minutes':round(elapsed/60,1),
                            'resumed_from_epoch':resumed_from,
                            'estimated_remaining_minutes':round(elapsed/epochs_completed*max(0,configuration.get('epochs',100)-h[-1]['epoch'])/60,1)}
result['completed_pairs']=[seed for seed in configuration.get('seeds',[]) if
    (root/f'training/Weather/96_192/{seed}/baseline/metrics.json').exists() and
    (root/f'training/Weather/96_192/{seed}/peak/metrics.json').exists()]
try:
    result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw',
                                          '--format=csv,noheader'],text=True).strip()
except Exception: pass
print(json.dumps(result,ensure_ascii=True))
