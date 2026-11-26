"""Synthetic integration check; random cache is never a research result."""
import json
import argparse
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from data_provider.peak_data import PeakDataset


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', default='Results/smoke_test')
    args_cli=parser.parse_args()
    # Keep evidence (logs/checkpoints/predictions) for inspection after the run.
    tmp=str(Path(args_cli.output).resolve())
    Path(tmp).mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(42)
    pd.DataFrame({'date':pd.date_range('2020-01-01',periods=120,freq='10min'),'wv':rng.normal(size=120),'rain':rng.normal(size=120)}).to_csv(Path(tmp)/'Weather.csv',index=False)
    args=SimpleNamespace(root_path=tmp,data_path='Weather',seq_len=6,pred_len=4,columns='',peak_columns='',embedding_root=tmp+'/emb')
    for split in ('train','val','test'):
        ds=PeakDataset(args,split,False); ds.cache.mkdir(parents=True,exist_ok=True)
        for i in range(len(ds)): np.save(ds.cache/f'{i}.npy',rng.normal(size=(8,2)).astype('float32'))
    command=[sys.executable,'train.py','--root_path',tmp,'--embedding_root',tmp+'/emb','--save',tmp+'/out','--seq_len','6','--pred_len','4','--channel','8','--d_llm','8','--head','2','--epochs','1','--num_workers','0','--batch_size','32']
    def run(extra, name):
        result=subprocess.run(command+extra,capture_output=True,text=True,encoding='utf-8',errors='replace')
        (Path(tmp)/name).write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        print(result.stdout,flush=True)
        if result.returncode:
            print(result.stderr,flush=True)
            result.check_returncode()
    run(['--mode','baseline'], 'baseline.log')
    baseline=Path(tmp)/'out/Weather/6_4/2024/baseline/best_model.pth'
    # Cross epoch four to exercise actual backbone unfreezing, including
    # Windows DataLoader worker spawning.
    run(['--mode','peak','--baseline_checkpoint',str(baseline),'--epochs','5','--num_workers','2'], 'peak.log')
    metrics=json.loads((Path(tmp)/'out/Weather/6_4/2024/peak/metrics.json').read_text())
    assert np.isfinite(metrics['mse']) and 0<=metrics['theta']<=1
    import torch
    state=torch.load(Path(tmp)/'out/Weather/6_4/2024/peak/best_model.pth',weights_only=False)
    assert state['args']['epochs']==5
    assert state['model']['correction.3.weight'].abs().sum()>0
    predictions=np.load(Path(tmp)/'out/Weather/6_4/2024/peak/test_predictions.npz')
    assert predictions['target'].shape==predictions['prediction'].shape
    assert np.isfinite(predictions['prediction']).all()
    (Path(tmp)/'summary.json').write_text(json.dumps({'status':'passed','data':'synthetic','embeddings':'random fixtures (GPT-2 not exercised)','peak_epochs':5,'workers':2,'metrics':metrics},indent=2),encoding='utf-8')
    print('Baseline -> checkpoint -> PeakCMA -> validation threshold -> test artifacts: OK')


if __name__=='__main__': main()
