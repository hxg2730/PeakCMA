"""Time real full-dataset cache rows; saved rows are reused by the full job."""
from pathlib import Path
import sys
import time
import json
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from data_provider.peak_data import PeakDataset
from storage.gen_prompt_emb import GenPromptEmb

torch.set_num_threads(8)
config=SimpleNamespace(root_path='dataset',data_path='Weather',seq_len=96,pred_len=192,columns='',
    peak_columns='wv (m/s),rain (mm)',embedding_root='Embeddings/PeakCMA',embedding_precision='float16',
    max_train_samples=0,max_val_samples=0,max_test_samples=0,window_manifest=None)
data=PeakDataset(config,'train',False)
generator=GenPromptEmb(data_path='Weather',model_name='models/pretrained/gpt2',device='cuda',
    input_len=96,prompt_batch_size=64,precision='float16').eval()
data.cache.mkdir(parents=True,exist_ok=True)
results=[]
for start in (0,4,8):
    rows=[data[i] for i in range(start,start+4)]
    x=torch.from_numpy(np.stack([r[0] for r in rows])); mark=torch.from_numpy(np.stack([r[2] for r in rows]))
    torch.cuda.synchronize(); began=time.monotonic()
    emb=generator.generate_embeddings(x,mark)
    torch.cuda.synchronize(); seconds=time.monotonic()-began
    for j in range(4):
        array=emb[j].cpu().numpy(); assert np.isfinite(array).all()
        with (data.cache/f'{start+j}.npy.tmp').open('wb') as stream: np.save(stream,array)
        (data.cache/f'{start+j}.npy.tmp').replace(data.cache/f'{start+j}.npy')
    result=dict(start=start,windows=4,seconds=seconds,seconds_per_window=seconds/4,
                peak_gpu_gb=torch.cuda.max_memory_allocated()/1024**3)
    results.append(result); print(json.dumps(result),flush=True)
Path('Results/weather_full/encoding_benchmark.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
