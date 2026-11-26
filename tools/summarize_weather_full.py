"""Six paired seeds: sample SD, paired tests, relative peak indices and pooled counts."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy import stats

p=argparse.ArgumentParser()
p.add_argument('--root',default='Results/weather_full')
p.add_argument('--partial',action='store_true')
a=p.parse_args(); root=Path(a.root)
config=json.loads((root/'configuration.json').read_text())
rows=[]; missing=[]
for seed in config['seeds']:
    folder=root/'training/Weather/96_192'/str(seed)
    try:
        baseline=json.loads((folder/'baseline/metrics.json').read_text())
        peak=json.loads((folder/'peak/metrics.json').read_text())
    except FileNotFoundError:
        missing.append(seed); continue
    row={'seed':seed}
    for mode in ('baseline','peak'):
        history=json.loads((folder/mode/'history.json').read_text())
        row[mode+'_epochs']=len(history)
        row[mode+'_best_validation_mse']=min(h['validation']['mse'] for h in history)
    for mode,metrics in [('baseline',baseline),('peak',peak)]:
        for key in ('mse','mae','tp','fp','fn','recall','precision','f1'):
            row[mode+'_'+key]=metrics[key]
    for metric in ('neighbourhood','amplitude','timing','rest'):
        ratios=[]
        for channel in (k for k in baseline if k.isdigit()):
            b=baseline[channel][metric]; q=peak[channel][metric]
            if b is not None and q is not None and b>0: ratios.append(q/b*100)
        row['relative_'+metric]=float(np.mean(ratios)) if ratios else None
    rows.append(row)
if missing and not a.partial: raise RuntimeError('Incomplete seed pairs: '+str(missing))
if not rows: raise RuntimeError('No complete pair yet')

def summary(values):
    values=np.asarray([v for v in values if v is not None],dtype=float)
    return dict(n=len(values),mean=float(values.mean()) if len(values) else None,
                sample_std=float(values.std(ddof=1)) if len(values)>1 else None)

def paired(before,after):
    pairs=[(b,q) for b,q in zip(before,after) if b is not None and q is not None]
    d=np.asarray([q-b for b,q in pairs]); n=len(d)
    result={'n':n,'difference':'PeakCMA minus baseline','mean_difference':float(d.mean()) if n else None,
            'ci95':None,'p_two_sided':None}
    if n>1:
        sd=d.std(ddof=1); critical=stats.t.ppf(.975,n-1)
        half=critical*sd/np.sqrt(n)
        result['ci95']=[float(d.mean()-half),float(d.mean()+half)]
        if sd>0: result['p_two_sided']=float(stats.ttest_1samp(d,0).pvalue)
    return result

report={'status':'partial' if missing else 'complete','configuration':config,'missing_seeds':missing,
        'seed_rows':rows,'mean_and_sample_std':{},'paired_tests':{},'pooled_events':{},
        'evaluation_unit':'All legal sliding forecast windows; events are counted per forecast instance, including overlapping forecasts'}
for key in ('mse','mae','recall','precision','f1'):
    before=[r['baseline_'+key] for r in rows]; after=[r['peak_'+key] for r in rows]
    report['mean_and_sample_std'][key]={'baseline':summary(before),'peak':summary(after)}
    report['paired_tests'][key]=paired(before,after)
for key in ('neighbourhood','amplitude','timing','rest'):
    after=[r['relative_'+key] for r in rows]
    report['mean_and_sample_std']['relative_'+key]={'baseline_index':100.,'peak':summary(after)}
    report['paired_tests']['relative_'+key]=paired([100.]*len(rows),after)
for mode in ('baseline','peak'):
    counts={key:sum(r[mode+'_'+key] for r in rows) for key in ('tp','fp','fn')}
    tp,fp,fn=counts['tp'],counts['fp'],counts['fn']
    counts.update(precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
                  f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
    report['pooled_events'][mode]=counts
name='partial_statistics' if missing else 'full_statistics'
(root/(name+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
with (root/(name+'.csv')).open('w',newline='',encoding='utf-8-sig') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
print(json.dumps({k:report[k] for k in ('status','missing_seeds','mean_and_sample_std','pooled_events')},indent=2))
