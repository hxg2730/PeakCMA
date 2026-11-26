"""Verify and export the validation-selected localisation-focused trial."""
from pathlib import Path
import sys
import os
import json
import hashlib
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); os.chdir(ROOT)
import numpy as np
import torch
from data_provider.peak_data import PeakDataset
from models.PeakCMA import PeakCMA

torch.set_num_threads(4)
out=Path('Results/weather_balanced')
folder=out/'training/Weather/96_192/2024/peak'
baseline_folder=Path('Results/weather_corrected/training/Weather/96_192/2024/baseline')
checkpoint=torch.load(folder/'best_model.pth',map_location='cpu',weights_only=False)
args=SimpleNamespace(**checkpoint['args'])
data=PeakDataset(args,'test')
for split in ('train','val','test'):
    a=json.loads((folder/'data_manifest.json').read_text())[split]['selected_windows']
    b=json.loads((baseline_folder/'data_manifest.json').read_text())[split]['selected_windows']
    assert a==b, 'Paired sample mismatch'
model=PeakCMA(data.channels,device='cpu',channel=args.channel,num_nodes=len(data.columns),
              seq_len=args.seq_len,pred_len=args.pred_len,dropout_n=args.dropout_n,d_llm=args.d_llm,
              e_layer=args.e_layer,d_layer=args.d_layer,head=args.head).eval()
model.load_state_dict(checkpoint['model'],strict=True)
x,y,mark,u,emb=data[0]
with torch.inference_mode():
    pred,aux=model(torch.from_numpy(x)[None].float(),torch.from_numpy(mark)[None].float(),torch.from_numpy(emb)[None],True)
saved=np.load(folder/'test_predictions.npz')
np.testing.assert_allclose(pred[0].numpy(),saved['prediction'][0],rtol=1e-4,atol=1e-5)
assert np.isfinite(pred.numpy()).all() and aux['delta'].abs().sum()>0
history=json.loads((folder/'history.json').read_text())
selected=history[checkpoint['epoch']-1]
assert selected['validation']['mse']<=args.max_validation_mse
metrics=json.loads((folder/'metrics.json').read_text())
baseline=json.loads((baseline_folder/'metrics.json').read_text())
checkpoint.update(event_threshold=metrics['theta'],experiment='clean-v2, validation event-F1 selection with validation-MSE constraint',
                  dataset_sha256=hashlib.sha256(Path('dataset/Weather.csv').read_bytes()).hexdigest())
weight=out/'PeakCMA_Weather_balanced.pth'
torch.save(checkpoint,weight)
report=dict(status='passed',weight=str(weight.resolve()),weight_bytes=weight.stat().st_size,
            weight_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),
            best_epoch=checkpoint['epoch'],trained_epochs=len(history),optimizer_updates=len(history)*8,
            training_config=vars(args),selected_validation=selected['validation'],
            baseline_metrics=baseline,peak_metrics=metrics,reload_prediction_verified=True,
            paired_samples_verified=True,windows=dict(train=64,val=16,test=32),
            comparison={k:dict(baseline=baseline[k],peak=metrics[k]) for k in ('mse','mae','precision','recall','f1','fp','fn','neighbourhood','amplitude','timing','rest')},
            limitations=['Limited CPU sample count and one seed; not a full paper benchmark',
                         'Localisation-focused coefficients, head LR and constrained F1 selection differ from paper defaults',
                         'Threshold and checkpoint selection use validation only; no test threshold optimisation'])
(out/'run_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
(out/'training_config.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8')
print(json.dumps({k:report[k] for k in ('weight','best_epoch','trained_epochs','selected_validation','comparison')},indent=2))
