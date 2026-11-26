import json
from types import SimpleNamespace
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_provider.peak_data import PeakDataset
from utils.peak import events
args=SimpleNamespace(root_path='dataset',data_path='Weather',seq_len=96,pred_len=192,columns='',peak_columns='wv (m/s),rain (mm)',embedding_root='Embeddings/PeakCMA',max_train_samples=64,max_val_samples=16,max_test_samples=32,window_manifest='Results/weather_real/training/Weather/96_192/2024/peak/data_manifest.json')
report={}
for split in ('train','val','test'):
    data=PeakDataset(args,split,False)
    counts={data.columns[c]:0 for c in data.channels}
    for g,t in data.samples:
        y=data.series[g][0][t:t+192]
        for c in data.channels: counts[data.columns[c]]+=len(events(y[:,c],data.thresholds[g][c]))
    report[split]=dict(windows=len(data),events=counts,normalisation=data.normalisation[0])
Path('Results/weather_corrected/data_preflight.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print({k:{'windows':v['windows'],'events':v['events'],'wind_std':v['normalisation']['std'][11]} for k,v in report.items()})
