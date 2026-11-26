"""Run from root: python -m storage.store_emb [dataset arguments]."""
import numpy as np
import json
import time
from pathlib import Path
from torch.utils.data import DataLoader
from train import parse_args
from data_provider.peak_data import PeakDataset
from storage.gen_prompt_emb import GenPromptEmb


def main():
    args=parse_args()
    generator=GenPromptEmb(data_path=args.data_path,device=args.device,input_len=args.seq_len,
                           model_name=args.llm_model_path,prompt_batch_size=args.prompt_batch_size,
                           precision=args.embedding_precision).eval()
    for split in ('train','val','test'):
        dataset=PeakDataset(args,split,embeddings=False)
        dataset.cache.mkdir(parents=True,exist_ok=True)
        start=time.monotonic(); generated=0; index=0
        for x,y,mark,u in DataLoader(dataset,batch_size=args.embedding_batch_size,shuffle=False,num_workers=args.num_workers):
            needed=[]
            for local in range(len(x)):
                file=dataset.cache/f'{index+local}.npy'
                valid=False
                if file.exists():
                    try:
                        saved=np.load(file,mmap_mode='r')
                        valid=saved.shape==(args.d_llm,len(dataset.columns)) and np.isfinite(saved).all()
                    except (ValueError,OSError,EOFError): pass
                if not valid: needed.append(local)
            if needed:
                embedding=generator.generate_embeddings(x[needed],mark[needed])
                for pos,local in enumerate(needed):
                    file=dataset.cache/f'{index+local}.npy'
                    temporary=file.with_suffix('.npy.tmp')
                    array=embedding[pos].cpu().numpy()
                    if not np.isfinite(array).all(): raise FloatingPointError('Nonfinite GPT-2 embedding')
                    with temporary.open('wb') as stream:
                        np.save(stream,array)
                    temporary.replace(file)
                generated+=len(needed)
            index+=len(x)
            seconds=time.monotonic()-start
            progress=dict(split=split,completed=index,total=len(dataset),generated_this_run=generated,
                          seconds=round(seconds,2),estimated_remaining_seconds=(len(dataset)-index)*seconds/generated if generated else None,
                          cache=str(dataset.cache),status='complete' if index==len(dataset) else 'running')
            progress_path=Path(args.embedding_progress) if args.embedding_progress else dataset.cache/'progress.json'
            progress_path.parent.mkdir(parents=True,exist_ok=True)
            temporary=progress_path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(progress,indent=2),encoding='utf-8'); temporary.replace(progress_path)
            print(f'{split}: {index}/{len(dataset)} cached; newly generated={generated}',flush=True)
        print(f'{split}: {len(dataset)} embeddings')


if __name__=='__main__': main()
