"""An interrupted/dropout training run must reproduce uninterrupted weights."""
import contextlib
import io
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
import train
from data_provider.peak_data import PeakDataset


class ResumeTests(unittest.TestCase):
    def test_epoch_resume_is_numerically_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rng=np.random.default_rng(10)
            pd.DataFrame({'date':pd.date_range('2020-01-01',periods=100,freq='10min'),
                          'wv':rng.normal(size=100),'rain':rng.normal(size=100)}).to_csv(root/'Weather.csv',index=False)
            config=SimpleNamespace(root_path=tmp,data_path='Weather',seq_len=6,pred_len=4,
                                    columns='',peak_columns='',embedding_root=tmp+'/emb')
            for split in ('train','val','test'):
                data=PeakDataset(config,split,False); data.cache.mkdir(parents=True)
                for i in range(len(data)):
                    np.save(data.cache/f'{i}.npy',rng.normal(size=(8,2)).astype('float32'))
            command=['train.py','--root_path',tmp,'--embedding_root',tmp+'/emb','--save',tmp+'/out',
                     '--seq_len','6','--pred_len','4','--channel','8','--d_llm','8','--head','2',
                     '--epochs','5','--batch_size','16','--num_workers','0','--mode','baseline','--seed','2024']
            original=torch.save
            def capture_first_epoch(value,path,*a,**kw):
                if isinstance(value,dict) and value.get('epoch')==1:
                    if Path(path).name=='last_training.pth.tmp': original(value,root/'first_last.pth')
                    if Path(path).name=='best_model.pth.tmp': original(value,root/'first_best.pth')
                return original(value,path,*a,**kw)
            with patch.object(sys,'argv',command),patch.object(torch,'save',capture_first_epoch),contextlib.redirect_stdout(io.StringIO()):
                train.main()
            folder=root/'out/Weather/6_4/2024/baseline'
            expected=torch.load(folder/'last_training.pth',weights_only=False)
            shutil.copyfile(root/'first_last.pth',folder/'last_training.pth')
            shutil.copyfile(root/'first_best.pth',folder/'best_model.pth')
            with patch.object(sys,'argv',command+['--resume','auto']),contextlib.redirect_stdout(io.StringIO()):
                train.main()
            actual=torch.load(folder/'last_training.pth',weights_only=False)
            self.assertEqual(actual['epoch'],5)
            for name,value in expected['model'].items():
                self.assertTrue(torch.equal(value,actual['model'][name]),name)
            self.assertEqual(expected['history'],actual['history'])
