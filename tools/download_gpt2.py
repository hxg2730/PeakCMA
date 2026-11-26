"""Download official GPT-2 Base files locally; no executable remote code."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

TARGET = Path('models/pretrained/gpt2')


def download(name):
    path = TARGET/name
    if path.exists() and path.stat().st_size > 0:
        print('Already downloaded:', name, flush=True)
        return
    url = 'https://huggingface.co/openai-community/gpt2/resolve/main/'+name
    with requests.get(url,stream=True,timeout=(30,120)) as response:
        response.raise_for_status()
        count=0; last=0
        with path.with_suffix(path.suffix+'.part').open('wb') as out:
            for chunk in response.iter_content(1024*1024):
                if not chunk: continue
                out.write(chunk); count+=len(chunk)
                if count-last>=50*1024*1024:
                    print(f'{name}: {count/1024**2:.0f} MiB',flush=True); last=count
        path.with_suffix(path.suffix+'.part').replace(path)
    print(f'Downloaded {name}: {count} bytes',flush=True)


if __name__=='__main__':
    TARGET.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download,['config.json','tokenizer_config.json','vocab.json','merges.txt','tokenizer.json','model.safetensors']))
