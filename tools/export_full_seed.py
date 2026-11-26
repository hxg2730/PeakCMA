"""Export a finished full-data seed pair before the other seeds finish."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from data_provider.peak_data import PeakDataset
from models.PeakCMA import PeakCMA

p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,default=2024); a=p.parse_args()
root=Path('Results/weather_full'); configuration=json.loads((root/'configuration.json').read_text())
folder=root/'training/Weather/96_192'/str(a.seed)
baseline=json.loads((folder/'baseline/metrics.json').read_text())
metrics=json.loads((folder/'peak/metrics.json').read_text())
state=torch.load(folder/'peak/best_model.pth',map_location='cpu',weights_only=False)
args=SimpleNamespace(**state['args'])
assert all(getattr(args,'max_'+s+'_samples')==0 for s in ('train','val','test'))
assert args.window_manifest is None
data=PeakDataset(args,'test'); assert len(data)==configuration['window_counts']['test']
device='cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(8)
model=PeakCMA(data.channels,device=device,channel=args.channel,num_nodes=len(data.columns),
              seq_len=args.seq_len,pred_len=args.pred_len,dropout_n=args.dropout_n,d_llm=args.d_llm,
              e_layer=args.e_layer,d_layer=args.d_layer,head=args.head).to(device).eval()
model.load_state_dict(state['model'],strict=True)
x,y,mark,u,emb=data[0]
with torch.inference_mode():
    pred=model(torch.from_numpy(x)[None].to(device).float(),torch.from_numpy(mark)[None].to(device).float(),
               torch.from_numpy(emb)[None].to(device))
saved=np.load(folder/'peak/test_predictions.npz')
np.testing.assert_allclose(pred[0].cpu().numpy(),saved['prediction'][0],rtol=1e-4,atol=1e-4)
state.update(event_threshold=metrics['theta'],dataset_sha256=configuration['source_sha256'],full_data=True,
             full_window_counts=configuration['window_counts'],embedding_precision=configuration['precision'])
weight=root/f'PeakCMA_Weather_full_seed{a.seed}.pth'
tmp=weight.with_suffix('.pth.tmp'); torch.save(state,tmp); tmp.replace(weight)
histories={mode:json.loads((folder/mode/'history.json').read_text()) for mode in ('baseline','peak')}
report=dict(seed=a.seed,full_data=True,window_counts=configuration['window_counts'],
            weight=str(weight),sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),
            actual_epochs={k:len(v) for k,v in histories.items()},best_peak_epoch=state['epoch'],
            baseline_metrics=baseline,peak_metrics=metrics,reload_prediction_verified=True)
(root/f'individual_report_seed{a.seed}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
