"""Real Weather + pretrained GPT-2, CPU-friendly training, persistent evidence."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'Results/weather_real'


def run(stage, arguments):
    print(f'=== {stage} ===',flush=True)
    with (OUT/(stage+'.log')).open('w',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,*arguments],cwd=ROOT,env=os.environ,
                                 stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,encoding='utf-8',errors='replace')
        for line in process.stdout:
            log.write(line); log.flush(); print(line,end='',flush=True)
        if process.wait(): raise RuntimeError(f'{stage} failed; inspect {log.name}')


def main():
    os.chdir(ROOT)
    os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONIOENCODING='utf-8',
                      HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    OUT.mkdir(parents=True,exist_ok=True)
    start=time.time()
    common=['--data_path','Weather','--root_path','dataset','--device','cpu',
            '--seq_len','96','--pred_len','192','--channel','32','--d_llm','768',
            '--peak_columns','wv (m/s),rain (mm)','--batch_size','8','--num_workers','0',
            '--max_train_samples','32','--max_val_samples','8','--max_test_samples','8',
            '--seed','2024','--save','Results/weather_real/training',
            '--llm_model_path','models/pretrained/gpt2','--prompt_batch_size','4']
    run('embeddings',['-m','storage.store_emb',*common])
    run('baseline',['train.py',*common,'--mode','baseline','--epochs','10'])
    baseline=OUT/'training/Weather/96_192/2024/baseline/best_model.pth'
    run('peak',['train.py',*common,'--mode','peak','--epochs','8','--baseline_checkpoint',str(baseline)])
    import numpy as np
    import torch
    from types import SimpleNamespace
    from models.PeakCMA import PeakCMA
    from data_provider.peak_data import PeakDataset
    peak_dir=OUT/'training/Weather/96_192/2024/peak'
    checkpoint=torch.load(peak_dir/'best_model.pth',map_location='cpu',weights_only=False)
    args=SimpleNamespace(**checkpoint['args'])
    data=PeakDataset(args,'test')
    model=PeakCMA(data.channels,device='cpu',channel=args.channel,num_nodes=len(data.columns),
                  seq_len=args.seq_len,pred_len=args.pred_len,dropout_n=args.dropout_n,
                  d_llm=args.d_llm,e_layer=args.e_layer,d_layer=args.d_layer,head=args.head).eval()
    model.load_state_dict(checkpoint['model'],strict=True)
    x,y,mark,u,emb=data[0]
    torch.set_num_threads(4)
    with torch.inference_mode():
        prediction,aux=model(torch.from_numpy(x)[None].float(),torch.from_numpy(mark)[None].float(),torch.from_numpy(emb)[None],True)
    saved=np.load(peak_dir/'test_predictions.npz')
    np.testing.assert_allclose(prediction[0].numpy(),saved['prediction'][0],rtol=1e-4,atol=1e-5)
    assert np.isfinite(prediction.numpy()).all()
    assert aux['delta'].abs().sum().item()>0
    from storage.gen_prompt_emb import GenPromptEmb
    generator=GenPromptEmb(data_path='Weather',model_name='models/pretrained/gpt2',
                           device='cpu',input_len=96,prompt_batch_size=1).eval()
    g,t=data.samples[0]
    raw=data.fill_input(data.raw_series[g][t-96:t,:1])
    direct=generator.generate_embeddings(torch.from_numpy(raw)[None],torch.from_numpy(mark)[None])
    np.testing.assert_allclose(direct[0,:,0].numpy(),emb[:,0],rtol=1e-4,atol=1e-4)
    metrics=json.loads((peak_dir/'metrics.json').read_text())
    checkpoint.update(event_threshold=metrics['theta'],dataset_sha256=hashlib.sha256((ROOT/'dataset/Weather.csv').read_bytes()).hexdigest(),
                      experiment='real Weather / real pretrained GPT-2 / limited windows; not a full benchmark')
    final=OUT/'PeakCMA_Weather_minimal.pth'
    torch.save(checkpoint,final)
    report=dict(status='passed',device='cpu',dataset_rows=52696,dynamic_variables=21,
                seq_len=96,pred_len=192,windows=dict(train=32,val=8,test=8),seed=2024,
                baseline_epochs=10,peak_epochs=8,best_peak_epoch=checkpoint['epoch'],
                pretrained_model='openai-community/gpt2',metrics=metrics,
                weight=str(final),weight_bytes=final.stat().st_size,
                reload_prediction_verified=True,nonzero_correction_verified=True,
                pretrained_embedding_recomputed_verified=True,
                elapsed_seconds=round(time.time()-start,2),
                dataset_source='https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/weather.csv')
    (OUT/'run_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
