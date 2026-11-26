"""Audit the actual small-sample run without retraining or test-set tuning."""
import os
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
import json
import argparse
import math
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); os.chdir(ROOT)
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from scipy.stats import rankdata
from data_provider.peak_data import PeakDataset
from models.PeakCMA import PeakCMA
from utils.peak import soft_labels, events, diagnostics

torch.set_num_threads(4)
parser=argparse.ArgumentParser()
parser.add_argument('--run_dir',default='Results/weather_real')
cli=parser.parse_args()
run_dir=Path(cli.run_dir)
weights=list(run_dir.glob('PeakCMA_Weather_*.pth'))
if len(weights)!=1: raise ValueError('Expected exactly one delivered PeakCMA weight')
cp=torch.load(weights[0],map_location='cpu',weights_only=False)
args=SimpleNamespace(**cp['args'])
train=PeakDataset(args,'train')
model=PeakCMA(train.channels,device='cpu',channel=args.channel,num_nodes=len(train.columns),seq_len=args.seq_len,
              pred_len=args.pred_len,dropout_n=args.dropout_n,d_llm=args.d_llm,e_layer=args.e_layer,d_layer=args.d_layer,head=args.head).eval()
model.load_state_dict(cp['model'])
history_file=run_dir/'training/Weather/96_192/2024/peak/history.json'
epochs=len(json.loads(history_file.read_text())) if history_file.exists() else args.epochs
steps=math.ceil(len(train)/args.batch_size)
report={'optimizer_updates':epochs*steps,'backbone_unfrozen_updates':max(0,epochs-args.freeze_epochs)*steps,
        'best_peak_epoch':cp['epoch'],'threshold':cp['event_threshold'],'splits':{}}

def mean(a): return float(np.mean(a)) if np.size(a) else None

for split in ('train','val','test'):
    data=PeakDataset(args,split)
    collected={k:[] for k in ('y','pred','score','base','delta','residual','label','u','logits')}
    with torch.inference_mode():
        for x,y,mark,u,emb in DataLoader(data,batch_size=8):
            pred,aux=model(x.float(),mark.float(),emb.float(),True)
            label=torch.cat([soft_labels(v[None],threshold.numpy(),data.channels) for v,threshold in zip(y,u)])
            for k,v in dict(y=y,pred=pred,score=aux['score'],base=aux['base'],delta=aux['delta'],residual=aux['residual'],label=label,u=u,logits=aux['logits']).items():
                collected[k].append(v.float().numpy())
    a={k:np.concatenate(v) for k,v in collected.items()}
    mask=model.peak_mask.numpy()
    weight=mask*(1+4*a['label'])
    bce=F.binary_cross_entropy_with_logits(torch.from_numpy(a['logits']),torch.from_numpy(a['label']),reduction='none').numpy()
    peakweight=mask*a['label']
    entry={'windows':len(data),'mse':mean((a['pred']-a['y'])**2),
           'localisation_loss':mean((bce*weight).sum((1,2))/(weight.sum((1,2))+1e-8)),
           'peak_loss':mean((abs(a['pred']-a['y'])*peakweight).sum((1,2))/(peakweight.sum((1,2))+1e-8)),
           'channels':{}}
    for c in data.channels:
        gt=np.zeros_like(a['score'][:,:,c],dtype=bool)
        counts=[]
        for b,y in enumerate(a['y']):
            positions=events(y[:,c],a['u'][b,c]); gt[b,positions]=True; counts.append(len(positions))
        s=a['score'][:,:,c]; positive=s[gt]; negative=s[~gt]
        ranks=rankdata(s.ravel())
        np_=gt.sum(); nn_=(~gt).sum()
        auc=float((ranks[gt.ravel()].sum()-np_*(np_+1)/2)/(np_*nn_)) if np_ and nn_ else None
        diagnostics_c=diagnostics(a['y'],a['pred'],a['score'],a['u'],[c],cp['event_threshold'])
        correlations=np.corrcoef(s)
        pairs=correlations[np.triu_indices(len(s),1)]
        entry['channels'][data.columns[c]]={
            'raw_training_threshold':data.normalisation[0]['peak_threshold_raw'][c],
            'events':int(np_),'events_per_window':counts,'windows_without_events':int(sum(v==0 for v in counts)),
            'event_step_fraction':float(gt.mean()),'soft_label_mean':mean(a['label'][:,:,c]),
            'score_quantiles':np.quantile(s,[0,.05,.5,.95,1]).tolist(),
            'mean_score_at_event':mean(positive),'mean_score_elsewhere':mean(negative),
            'event_step_auc':auc,'average_score_profile_correlation_between_windows':mean(pairs),
            'fraction_steps_above_threshold':mean(s>cp['event_threshold']),
            'mean_abs_correction':mean(abs(a['delta'][:,:,c])),
            'mean_abs_correction_at_events':mean(abs(a['delta'][:,:,c][gt])),
            'mean_signed_correction_at_events':mean(a['delta'][:,:,c][gt]),
            'base_mae_at_true_event_time':mean(abs(a['base'][:,:,c][gt]-a['y'][:,:,c][gt])),
            'peak_mae_at_true_event_time':mean(abs(a['pred'][:,:,c][gt]-a['y'][:,:,c][gt])),
            'diagnostics':diagnostics_c}
    if split=='val':
        sweep=[]
        for threshold in np.linspace(0,1,101):
            d=diagnostics(a['y'],a['pred'],a['score'],a['u'],data.channels,float(threshold))
            sweep.append({k:d[k] for k in ('tp','fp','fn','f1')}|{'threshold':float(threshold)})
        entry['top_validation_thresholds']=sorted(sweep,key=lambda d:d['f1'],reverse=True)[:10]
    report['splits'][split]=entry
torch.manual_seed(args.seed)
initial=PeakCMA(train.channels,device='cpu',channel=args.channel,num_nodes=len(train.columns),seq_len=args.seq_len,
                pred_len=args.pred_len,dropout_n=args.dropout_n,d_llm=args.d_llm,e_layer=args.e_layer,d_layer=args.d_layer,head=args.head)
report['new_parameter_relative_changes']={}
for name,p in model.named_parameters():
    if not name.startswith('backbone.'):
        old=dict(initial.named_parameters())[name]
        report['new_parameter_relative_changes'][name]=float(((p-old).norm()/old.norm()).detach()) if old.norm()>0 else None
output=run_dir/'peak_diagnosis.json'
output.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
