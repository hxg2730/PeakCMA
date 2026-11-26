"""Reuse only unchanged raw-history prompts; never reuse training weights."""
from pathlib import Path
import sys
import json
import hashlib
import shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train import parse_args
from data_provider.peak_data import PeakDataset

args=parse_args()
manifest=json.loads(Path(args.window_manifest).read_text())
source=json.loads(Path('dataset/Weather.source.json').read_text())
assert hashlib.sha256(Path('dataset/Weather.csv').read_bytes()).hexdigest()==source['sha256']
report={}
for split in ('train','val','test'):
    data=PeakDataset(args,split,False)
    old=manifest[split]
    assert old['columns']==data.columns
    mapping={tuple(window):i for i,window in enumerate(old['selected_windows'])}
    data.cache.mkdir(parents=True,exist_ok=True)
    copied=0
    for i,(g,t) in enumerate(data.samples):
        previous=mapping.get((g,t))
        if previous is None: continue
        raw=data.raw_series[g][t-args.seq_len:t]
        if not np.isfinite(raw).all(): continue
        file=Path(old['cache'])/f'{previous}.npy'
        if not file.exists(): continue
        embedding=np.load(file)
        if embedding.shape!=(768,len(data.columns)) or not np.isfinite(embedding).all(): continue
        shutil.copyfile(file,data.cache/f'{i}.npy'); copied+=1
    report[split]={'total':len(data),'reused_identical_raw_prompts':copied}
print(json.dumps(report,indent=2),flush=True)
