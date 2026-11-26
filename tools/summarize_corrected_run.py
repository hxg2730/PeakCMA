"""Summarise a completed paired run; never selects settings using test metrics."""
import json
from pathlib import Path
import hashlib

root=Path('Results/weather_corrected')
report=json.loads((root/'run_report.json').read_text())
base=root/'training/Weather/96_192/2024/baseline'
peak=root/'training/Weather/96_192/2024/peak'
baseline=json.loads((base/'metrics.json').read_text())
current=json.loads((peak/'metrics.json').read_text())
histories={name:json.loads((folder/'history.json').read_text()) for name,folder in [('baseline',base),('peak',peak)]}
report['actual_baseline_epochs']=len(histories['baseline'])
report['actual_peak_epochs']=len(histories['peak'])
report['optimizer_updates']={name:len(h)*8 for name,h in histories.items()}
report['preprocessing_version']='sentinel-clean-v2'
report['baseline_metrics']=baseline
report['paired_comparison']={}
for key in ('mse','mae','recall','precision','f1','fp','fn','amplitude','timing','neighbourhood','rest'):
    a,b=baseline[key],current[key]
    report['paired_comparison'][key]={'baseline':a,'peak':b,
        'change':b-a if a is not None and b is not None else None}
report['weight_sha256']=hashlib.sha256(Path(report['weight']).read_bytes()).hexdigest()
report['data_preflight']=json.loads((root/'data_preflight.json').read_text())
report['limitations']=['CPU-sized 64/16/32 windows, one seed, not a full benchmark',
    'Joint validation checkpoint selection differs from paper default MSE selection',
    'Do not directly compare MSE with sentinel-contaminated older experiments']
(root/'run_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:report[k] for k in ('actual_baseline_epochs','actual_peak_epochs','optimizer_updates','paired_comparison','weight')},indent=2))
