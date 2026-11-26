"""Chronological CSV adapter. Multi-location input uses a location column."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SETTINGS = {'Weather': (96, 192, ['wv', 'rain'], 2, 3, '10min'),
            'Weather2K-R': (8, 8, ['Tair', 'WS'], 1, 2, '3h'),
            'ERA5': (48, 24, ['t2m', 'WS10'], 2, 3, 'h')}


class PeakDataset(Dataset):
    def __init__(self, args, split, embeddings=True):
        self.split = split; self.seq_len = args.seq_len; self.pred_len = args.pred_len
        self.with_embeddings = embeddings
        frame = pd.read_csv(Path(args.root_path) / (args.data_path + '.csv'))
        frame['date'] = pd.to_datetime(frame['date'])
        if args.data_path == 'ERA5' and 'WS10' not in frame:
            frame['WS10'] = np.hypot(frame['u10'], frame['v10'])
        excluded = {'date', 'location', 'latitude', 'longitude', 'altitude'}
        self.columns = args.columns.split(',') if args.columns else [c for c in frame if c not in excluded]
        self.preprocessing_version = 'sentinel-clean-v2'
        self.sentinel_counts = frame[self.columns].eq(-9999).sum().to_dict()
        frame[self.columns] = frame[self.columns].replace(-9999, np.nan)
        if args.data_path in ('Weather2K-R', 'ERA5') and len(self.columns) != 10:
            raise ValueError('Specify --columns with the ten dynamic channels; ERA5 must include derived WS10.')
        self.channels = [self.columns.index(c) for c in (args.peak_columns.split(',') if args.peak_columns else SETTINGS[args.data_path][2])]
        groups = list(frame.groupby('location', sort=True)) if 'location' in frame else [(0, frame)]
        raw = []; train = []; boundaries = []
        for _, group in groups:
            group = group.sort_values('date').set_index('date')
            weather_boundaries = None
            if args.data_path == 'Weather' and len(group) == 52696:
                # Preserve the published raw-index split timestamps even when
                # cleaning the standard CSV's duplicate and missing intervals.
                weather_boundaries = (group.index[36887],group.index[42156])
            if group.index.has_duplicates:
                duplicate_rows = group[group.index.duplicated(keep=False)]
                for _, repeated in duplicate_rows.groupby(level=0):
                    if not (repeated.eq(repeated.iloc[0]) | (repeated.isna() & repeated.iloc[0].isna())).all().all():
                        raise ValueError('Conflicting duplicate timestamps; provide a location column for stations/grid cells.')
                group = group[~group.index.duplicated(keep='first')]
            group = group.reindex(pd.date_range(group.index.min(), group.index.max(), freq=SETTINGS[args.data_path][5]))
            n = len(group)
            if args.data_path == 'Weather2K-R':
                a = int(group.index.searchsorted('2020-01-01')); b = int(group.index.searchsorted('2020-08-01'))
                group = group.loc['2017-01-01':'2021-08-31 23:59:59']
                a = int(group.index.searchsorted('2020-01-01')); b = int(group.index.searchsorted('2020-08-01')); n = len(group)
            elif weather_boundaries is not None:
                a,b = (int(group.index.searchsorted(t)) for t in weather_boundaries)
            else:
                a, b = int(n*.7), int(n*.8)
            values = group[self.columns].to_numpy(dtype=np.float32)
            raw.append((values, group.index)); train.append(values[:a]); boundaries.append((a,b,n))
        pooled = np.concatenate(train)
        physical_thresholds = np.nanquantile(pooled, .95, axis=0)
        self.samples = []; self.series = []; self.thresholds = []; self.raw_series = []
        self.normalisation = []
        self.marks = []
        pooled_mean = np.nanmean(pooled, axis=0); pooled_std = np.nanstd(pooled, axis=0)
        for g, ((values, dates), training, (a,b,n)) in enumerate(zip(raw, train, boundaries)):
            mean = np.nanmean(training, axis=0) if args.data_path == 'ERA5' else pooled_mean
            std = np.nanstd(training, axis=0) if args.data_path == 'ERA5' else pooled_std
            std = np.where(std > 1e-8, std, 1.)
            self.normalisation.append({'mean':mean.tolist(),'std':std.tolist(),
                                       'peak_threshold_raw':physical_thresholds.tolist()})
            self.thresholds.append((physical_thresholds-mean)/std)
            self.series.append(((values-mean)/std, dates))
            self.marks.append(np.stack([dates.year,dates.month,dates.day,dates.dayofweek,
                                         dates.hour,dates.minute],-1).astype(np.float32))
            self.raw_series.append(values)
            start, end = {'train': (0,a), 'val': (a,b), 'test': (b,n)}[split]
            for t in range(max(start, self.seq_len), end-self.pred_len+1):
                x = values[t-self.seq_len:t]; y = values[t:t+self.pred_len]
                if not np.isfinite(y).all(): continue
                # Fill only short internal input gaps, using history exclusively.
                if not np.isfinite(self.fill_input(x)).all(): continue
                self.samples.append((g,t))
        if not self.samples: raise ValueError(f'No usable {split} windows in CSV')
        limit = getattr(args, 'max_'+split+'_samples', 0)
        if limit > 0 and len(self.samples) > limit:
            anchors=[]
            manifest=getattr(args,'window_manifest',None)
            if manifest:
                requested=json.loads(Path(manifest).read_text(encoding='utf-8'))[split]['selected_windows']
                valid=set(self.samples)
                anchors=[tuple(v) for v in requested if tuple(v) in valid][:limit]
            selected=set(anchors)
            remaining=[v for v in self.samples if v not in selected]
            extra=limit-len(anchors)
            if extra:
                anchors += [remaining[i] for i in np.linspace(0,len(remaining)-1,extra,dtype=int)]
            self.samples=sorted(anchors)
        digest = hashlib.sha256()
        with Path(args.root_path, args.data_path+'.csv').open('rb') as source:
            for block in iter(lambda: source.read(1024*1024), b''):
                digest.update(block)
        digest.update(('|'.join(self.columns)+'|prompt5-three-decimals-raw-v1').encode())
        digest.update(repr(self.samples).encode())
        digest.update(self.preprocessing_version.encode())
        precision=getattr(args,'embedding_precision','float32')
        if precision!='float32': digest.update(('llm-'+precision).encode())
        signature = digest.hexdigest()[:16]
        self.cache = Path(args.embedding_root) / args.data_path / f'{args.seq_len}_{args.pred_len}_{signature}' / split
        self.embedding_bank=None
        if embeddings and getattr(args,'preload_embeddings',False):
            print(f'Preloading {split}: {len(self.samples)} embeddings into host RAM',flush=True)
            for i in range(len(self.samples)):
                array=np.load(self.cache/f'{i}.npy').astype(np.float32,copy=False)
                if self.embedding_bank is None:
                    self.embedding_bank=np.empty((len(self.samples),*array.shape),dtype=np.float32)
                if array.shape!=self.embedding_bank.shape[1:] or not np.isfinite(array).all():
                    raise ValueError(f'Invalid cached embedding at {self.cache}/{i}.npy')
                self.embedding_bank[i]=array

    @staticmethod
    def fill_input(x):
        if np.isfinite(x).all():
            return x.copy()
        out = x.copy()
        for c in range(out.shape[1]):
            missing = ~np.isfinite(out[:,c]); i = 0
            while i < len(out):
                if not missing[i]: i += 1; continue
                j = i
                while j < len(out) and missing[j]: j += 1
                if j-i <= 2 and i > 0 and j < len(out):
                    out[i:j,c] = np.linspace(out[i-1,c], out[j,c], j-i+2)[1:-1]
                i = j
        return out

    def __len__(self): return len(self.samples)

    def __getitem__(self, index):
        g, t = self.samples[index]; (data, dates) = self.series[g]
        x = self.fill_input(data[t-self.seq_len:t]); y = data[t:t+self.pred_len]
        mark = self.marks[g][t-self.seq_len:t]
        if not self.with_embeddings:
            return self.fill_input(self.raw_series[g][t-self.seq_len:t]), y, mark, self.thresholds[g].astype(np.float32)
        if self.embedding_bank is not None:
            embedding=self.embedding_bank[index]
        else:
            file = self.cache / f'{index}.npy'
            if not file.exists(): raise FileNotFoundError(f'Generate offline embeddings first: {file}')
            embedding = np.load(file).astype(np.float32)
        return x, y, mark, self.thresholds[g].astype(np.float32), embedding
