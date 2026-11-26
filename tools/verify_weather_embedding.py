"""Independently recompute a cached real GPT-2 prompt without batch padding."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np
import torch
from data_provider.peak_data import PeakDataset
from storage.gen_prompt_emb import GenPromptEmb

torch.set_num_threads(4)
args=SimpleNamespace(root_path='dataset',data_path='Weather',seq_len=96,pred_len=192,
                     columns='',peak_columns='wv (m/s),rain (mm)',embedding_root='Embeddings/PeakCMA',
                     max_train_samples=32,max_val_samples=8,max_test_samples=8)
data=PeakDataset(args,'test',False)
x,y,mark,u=data[0]
model=GenPromptEmb(data_path='Weather',model_name='models/pretrained/gpt2',device='cpu',
                   input_len=96,prompt_batch_size=1).eval()
direct=model.generate_embeddings(torch.from_numpy(x[None,:,:1]),torch.from_numpy(mark[None]))[0,:,0].numpy()
cached=np.load(data.cache/'0.npy')[:,0]
np.testing.assert_allclose(direct,cached,rtol=1e-4,atol=1e-4)
report=dict(status='passed',model='official pretrained GPT-2 Base',comparison='single prompt vs batched cached embedding',
            maximum_absolute_difference=float(np.max(abs(direct-cached))))
(ROOT/'Results/weather_real/embedding_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(report)
