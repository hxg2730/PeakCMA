"""Full Weather, uncapped chronological windows, paired baseline and PeakCMA."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def atomic_json(path,value):
    tmp=path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf-8'); tmp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',choices=['cpu','cuda'],required=True)
    p.add_argument('--precision',choices=['float32','float16','bfloat16'],default='float32')
    p.add_argument('--seeds',type=int,nargs='+',default=[2024])
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--output',default='Results/weather_full')
    p.add_argument('--llm_path',default='models/pretrained/gpt2')
    p.add_argument('--prompt_batch_size',type=int,default=16)
    p.add_argument('--embedding_batch_size',type=int,default=4)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--prepare_only',action='store_true')
    p.add_argument('--preload_embeddings',action='store_true')
    a=p.parse_args()
    os.chdir(ROOT)
    os.environ.update(OMP_NUM_THREADS=str(a.threads),MKL_NUM_THREADS=str(a.threads),PYTHONIOENCODING='utf-8')
    import torch
    import pandas as pd
    import numpy as np
    import transformers
    from types import SimpleNamespace
    from data_provider.peak_data import PeakDataset
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    if a.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA is not available in this environment')
    if a.device=='cpu' and a.precision!='float32': raise ValueError('CPU encoding requires float32')
    source=Path('dataset/Weather.csv'); meta=json.loads(Path('dataset/Weather.source.json').read_text())
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    df=pd.read_csv(source)
    assert digest==meta['sha256'] and df.shape==(52696,22), 'Incomplete or modified source CSV'
    config=SimpleNamespace(root_path='dataset',data_path='Weather',seq_len=96,pred_len=192,columns='',
                           peak_columns='wv (m/s),rain (mm)',embedding_root='Embeddings/PeakCMA',
                           max_train_samples=0,max_val_samples=0,max_test_samples=0,
                           window_manifest=None,embedding_precision=a.precision)
    datasets={s:PeakDataset(config,s,False) for s in ('train','val','test')}
    audit={s:dict(windows=len(d),cache=str(d.cache),columns=d.columns,
                  selected_windows=d.samples,normalisation=d.normalisation) for s,d in datasets.items()}
    atomic_json(out/'full_data_manifest.json',audit)
    expected=sum(len(d) for d in datasets.values())*768*21*4
    existing=sum(f.stat().st_size for d in datasets.values() for f in d.cache.glob('*.npy'))
    free=shutil.disk_usage(ROOT).free
    configuration=dict(device=a.device,precision=a.precision,seeds=a.seeds,epochs=a.epochs,
        full_dataset_verified=True,source_sha256=digest,source_rows=len(df),
        window_counts={s:len(d) for s,d in datasets.items()},sample_limits=0,
        embedding_bytes=expected,free_disk_bytes=free,torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name(0) if a.device=='cuda' else None,
        numpy_version=np.__version__,pandas_version=pd.__version__,transformers_version=transformers.__version__,
        channel=32,e_layer=1,d_layer=2,batch_size=32,
        preload_embeddings=a.preload_embeddings,
        backbone_configuration_source='https://github.com/ChenxiLiu-HNU/TimeCMA/blob/main/scripts/Weather.sh (192-step configuration)',
        experiment='Full Weather 96 -> 192, paired baseline/PeakCMA, paper loss coefficients and MSE selection')
    atomic_json(out/'configuration.json',configuration)
    if free<max(0,expected-existing)+2*1024**3: raise RuntimeError('Insufficient free disk for cache, checkpoints and predictions')
    common=['--data_path','Weather','--root_path','dataset','--device',a.device,
            '--seq_len','96','--pred_len','192','--channel','32','--head','8','--e_layer','1','--d_layer','2',
            '--d_llm','768','--peak_columns','wv (m/s),rain (mm)','--batch_size','32','--num_workers','0',
            '--max_train_samples','0','--max_val_samples','0','--max_test_samples','0',
            '--epochs',str(a.epochs),'--min_epochs','50','--es_patience','50','--freeze_epochs','4',
            '--learning_rate','0.0001','--backbone_lr','0.00001','--lambda_loc','0.1','--lambda_peak','0.5',
            '--omega','5','--dropout_n','0.1','--weight_decay','0.001','--selection_metric','mse',
            '--llm_model_path',a.llm_path,'--embedding_precision',a.precision,
            '--embedding_batch_size',str(a.embedding_batch_size),'--prompt_batch_size',str(a.prompt_batch_size),
            '--embedding_progress',str(out/'cache_progress.json'),'--save',str(out/'training'),'--resume','auto']
    if a.preload_embeddings: common.append('--preload_embeddings')
    stages=[('embeddings',['-m','storage.store_emb',*common])]
    for seed in a.seeds:
        stages.append((f'{seed}_baseline',['train.py',*common,'--seed',str(seed),'--mode','baseline']))
        baseline=out/'training/Weather/96_192'/str(seed)/'baseline/best_model.pth'
        stages.append((f'{seed}_peak',['train.py',*common,'--seed',str(seed),'--mode','peak','--baseline_checkpoint',str(baseline)]))
    atomic_json(out/'commands.json',dict(stages=stages))
    status=dict(status='prepared',pid=os.getpid(),configuration=str(out/'configuration.json'))
    atomic_json(out/'status.json',status)
    print(json.dumps(configuration,indent=2),flush=True)
    if a.prepare_only: return
    try:
        for name,arguments in stages:
            status.update(status='running',stage=name,started_unix=time.time())
            atomic_json(out/'status.json',status)
            print('=== '+name+' ===',flush=True)
            with (out/(name+'.log')).open('a',encoding='utf-8') as log:
                process=subprocess.Popen([sys.executable,*arguments],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                         text=True,encoding='utf-8',errors='replace')
                status['child_pid']=process.pid; atomic_json(out/'status.json',status)
                for line in process.stdout:
                    log.write(line); log.flush(); print(line,end='',flush=True)
                if process.wait(): raise RuntimeError(name+' failed; see its log')
        exports={}
        for seed in a.seeds:
            folder=out/'training/Weather/96_192'/str(seed)
            peak=json.loads((folder/'peak/metrics.json').read_text())
            baseline=json.loads((folder/'baseline/metrics.json').read_text())
            state=torch.load(folder/'peak/best_model.pth',map_location='cpu',weights_only=False)
            state.update(event_threshold=peak['theta'],dataset_sha256=digest,full_data=True,
                         full_window_counts=configuration['window_counts'],embedding_precision=a.precision)
            from models.PeakCMA import PeakCMA
            data=PeakDataset(config,'test')
            model=PeakCMA(data.channels,device=a.device,channel=32,num_nodes=21,seq_len=96,pred_len=192,
                          dropout_n=.1,d_llm=768,e_layer=1,d_layer=2,head=8).to(a.device).eval()
            model.load_state_dict(state['model'],strict=True)
            x,y,mark,u,emb=data[0]
            with torch.inference_mode():
                prediction=model(torch.from_numpy(x)[None].to(a.device).float(),
                                 torch.from_numpy(mark)[None].to(a.device).float(),
                                 torch.from_numpy(emb)[None].to(a.device))
            saved=np.load(folder/'peak/test_predictions.npz')
            np.testing.assert_allclose(prediction[0].cpu().numpy(),saved['prediction'][0],rtol=1e-4,atol=1e-4)
            weight=out/f'PeakCMA_Weather_full_seed{seed}.pth'
            temp=weight.with_suffix('.pth.tmp'); torch.save(state,temp); temp.replace(weight)
            exports[str(seed)]=dict(weight=str(weight),sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),
                                    baseline_metrics=baseline,peak_metrics=peak,best_epoch=state['epoch'],
                                    reload_prediction_verified=True)
        atomic_json(out/'final_report.json',dict(configuration=configuration,runs=exports))
        status.update(status='complete',finished_unix=time.time(),report=str(out/'final_report.json'))
        atomic_json(out/'status.json',status)
    except BaseException as error:
        status.update(status='failed',error=repr(error)); atomic_json(out/'status.json',status)
        raise


if __name__=='__main__': main()
