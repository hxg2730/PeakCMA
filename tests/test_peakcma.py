import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from models.PeakCMA import PeakCMA
from utils.peak import events, soft_labels, peak_loss, match_events, diagnostics, select_threshold
from data_provider.peak_data import PeakDataset


class PeakTests(unittest.TestCase):
    def test_ram_embeddings_equal_disk_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame({'date':pd.date_range('2020-01-01',periods=100,freq='10min'),
                          'wv':np.arange(100),'rain':np.arange(100)}).to_csv(Path(tmp)/'Weather.csv',index=False)
            args=SimpleNamespace(root_path=tmp,data_path='Weather',seq_len=6,pred_len=4,
                                 columns='',peak_columns='',embedding_root=tmp)
            disk=PeakDataset(args,'train',False); disk.cache.mkdir(parents=True)
            for i in range(len(disk)): np.save(disk.cache/f'{i}.npy',np.full((8,2),i,dtype=np.float32))
            args.preload_embeddings=True
            memory=PeakDataset(args,'train')
            args.preload_embeddings=False
            ordinary=PeakDataset(args,'train')
            self.assertEqual(memory.cache,ordinary.cache)
            for i in (0,len(memory)-1):
                for a,b in zip(memory[i],ordinary[i]): np.testing.assert_array_equal(a,b)

    def test_threshold_selection_matches_exhaustive_protocol(self):
        rng=np.random.default_rng(7)
        y=rng.normal(size=(2,12,2)); scores=rng.uniform(size=y.shape)
        thresholds=np.full((2,2),.5)
        theta,counts=select_threshold(y,scores,thresholds,[0,1])
        exhaustive=[]
        for t in np.linspace(0,1,101):
            d=diagnostics(y,y,scores,thresholds,[0,1],float(t))
            exhaustive.append(((d['f1'],-d['fp'],float(t)),float(t),d))
        expected=max(exhaustive,key=lambda item:item[0])
        self.assertEqual(theta,expected[1])
        for key in ('tp','fp','fn','f1'): self.assertEqual(counts[key],expected[2][key])

    def test_missing_sentinel_excluded_from_statistics_and_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame=pd.DataFrame({'date':pd.date_range('2020-01-01',periods=100,freq='10min'),
                                'wv':np.arange(100,dtype=float),'rain':np.arange(100,dtype=float)})
            frame.loc[30,'wv']=-9999
            frame.to_csv(Path(tmp)/'Weather.csv',index=False)
            args=SimpleNamespace(root_path=tmp,data_path='Weather',seq_len=6,pred_len=4,
                                 columns='',peak_columns='',embedding_root=tmp)
            data=PeakDataset(args,'train',False)
            self.assertTrue(np.isnan(data.raw_series[0][30,0]))
            expected=np.std(np.delete(np.arange(70,dtype=float),30))
            self.assertAlmostEqual(data.normalisation[0]['std'][0],expected,places=4)
            for i,(g,t) in enumerate(data.samples):
                self.assertFalse(t<=30<t+4)
                self.assertTrue(np.isfinite(data[i][0]).all())
                self.assertTrue(np.isfinite(data[i][1]).all())
            self.assertEqual(frame.loc[30,'wv'],-9999)

    def test_paper_dataset_shapes(self):
        for name,seq,horizon,nodes,channels in [
            ('Weather',96,192,21,[0,1]),
            ('Weather2K-R',8,8,10,[0,1]),
            ('ERA5',48,24,10,[0,1]),
        ]:
            with self.subTest(dataset=name):
                model=PeakCMA(channels,device='cpu',channel=32,num_nodes=nodes,
                              seq_len=seq,pred_len=horizon,d_llm=768,head=8)
                x=torch.randn(2,seq,nodes)
                pred,aux=model(x,torch.zeros(2,seq,6),torch.randn(2,768,nodes),True)
                self.assertEqual(tuple(pred.shape),(2,horizon,nodes))
                self.assertTrue(torch.equal(pred,aux['base']))
                loss=peak_loss(pred,torch.randn_like(pred),aux['logits'],torch.zeros_like(pred),model.peak_mask)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertIsNotNone(model.correction[-1].weight.grad)

    def test_events_and_matching(self):
        self.assertEqual(events([0,5,5,0,4,0],1,1,3),[1,4])
        self.assertEqual(events([0,3,0,4,0],1,1,3),[3])
        self.assertEqual(match_events([1,3],[0,2],1),[(1,0),(3,2)])
        p=soft_labels(torch.tensor([[[0.],[4.],[0.],[0.],[0.],[0.]]]),[2],[0],1,2)
        self.assertEqual(p[0,1,0],1)
        self.assertEqual(p[0,5,0],0)

    def test_zero_init_mask_and_gradients(self):
        torch.manual_seed(42)
        model=PeakCMA([0],device='cpu',channel=8,num_nodes=2,seq_len=6,pred_len=4,d_llm=8,head=2,dropout_n=0)
        model.eval(); x=torch.randn(2,6,2); mark=torch.zeros(2,6,6); emb=torch.randn(2,8,2)
        pred,aux=model(x,mark,emb,True)
        self.assertTrue(torch.equal(pred,aux['base']))
        with torch.no_grad(): model.correction[-1].bias.fill_(1.)
        pred,aux=model(x,mark,emb,True)
        self.assertTrue(torch.equal(pred[:,:,1],aux['base'][:,:,1]))
        loss=peak_loss(pred,torch.ones_like(pred),aux['logits'],torch.ones_like(pred),model.peak_mask)
        loss.backward()
        self.assertTrue(torch.isfinite(model.localisation.weight.grad).all())
        model.freeze_backbone(); self.assertFalse(any(p.requires_grad for p in model.backbone.parameters()))
        model.freeze_backbone(False); self.assertTrue(all(p.requires_grad for p in model.backbone.parameters()))

    def test_data_split_and_training_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            n=100
            pd.DataFrame({'date':pd.date_range('2020-01-01',periods=n,freq='10min'),'wv':np.arange(n),'rain':np.arange(n)}).to_csv(Path(tmp)/'Weather.csv',index=False)
            args=SimpleNamespace(root_path=tmp,data_path='Weather',seq_len=6,pred_len=4,columns='',peak_columns='',embedding_root=tmp)
            train=PeakDataset(args,'train',False); val=PeakDataset(args,'val',False); test=PeakDataset(args,'test',False)
            self.assertEqual(val.samples[0][1],70)
            self.assertEqual(test.samples[0][1],80)
            np.testing.assert_equal(train.thresholds,val.thresholds)
            self.assertEqual(train.samples[-1][1]+4,70)
            g,t=val.samples[0]
            dates=val.series[g][1][t-val.seq_len:t]
            expected=np.stack([dates.year,dates.month,dates.day,dates.dayofweek,dates.hour,dates.minute],-1).astype(np.float32)
            np.testing.assert_array_equal(val[0][2],expected)
            self.assertTrue(np.isnan(PeakDataset.fill_input(np.array([[1.],[np.nan],[np.nan],[np.nan],[2.]]))).any())

    def test_no_events_undefined(self):
        a=np.zeros((1,5,1))
        r=diagnostics(a,a,a,[1],[0],.5)
        self.assertIsNone(r['amplitude']); self.assertIsNone(r['recall'])

    def test_identical_duplicate_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame=pd.DataFrame({'date':pd.date_range('2020-01-01',periods=100,freq='10min'),
                                'wv':np.arange(100),'rain':np.arange(100)})
            frame=pd.concat([frame.iloc[:20],frame.iloc[19:]],ignore_index=True)
            frame.to_csv(Path(tmp)/'Weather.csv',index=False)
            args=SimpleNamespace(root_path=tmp,data_path='Weather',seq_len=6,pred_len=4,
                                 columns='',peak_columns='',embedding_root=tmp)
            data=PeakDataset(args,'train',False)
            self.assertEqual(len(data.series[0][0]),100)
            self.assertTrue(np.isfinite(data[0][0]).all())


if __name__=='__main__': unittest.main()
