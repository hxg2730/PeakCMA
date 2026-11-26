"""Restart only this experiment's processes from atomic epoch checkpoints."""
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1]; os.chdir(root)
out=root/'Results/weather_full'
status=json.loads((out/'status.json').read_text())
config=json.loads((out/'configuration.json').read_text())
if status.get('stage')!='2024_baseline': raise RuntimeError('Unexpected stage; refusing automatic restart')
for key,expected in [('pid','tools/run_weather_full.py'),('child_pid','train.py')]:
    pid=status[key]; proc=Path('/proc')/str(pid)
    command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    if Path(os.readlink(proc/'cwd')).resolve()!=root or expected not in command:
        raise RuntimeError('Process ownership validation failed')
for key in ('pid','child_pid'): os.kill(status[key],signal.SIGTERM)
time.sleep(2)
folder=out/'training/Weather/96_192/2024/baseline'
for name in ('last_training.pth','best_model.pth'):
    shutil.copyfile(folder/name,folder/(name+'.before_ram'))
env=os.environ.copy(); env.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
command=[sys.executable,'-u','tools/run_weather_full.py','--device','cuda','--precision',config['precision'],
         '--prompt_batch_size','64','--embedding_batch_size','4','--threads','8','--seeds',
         *map(str,config['seeds']),'--epochs',str(config['epochs']),'--preload_embeddings']
with (out/'driver.log').open('a') as log:
    log.write('\nRestarting with host RAM embedding cache; all samples and optimisation preserved.\n'); log.flush()
    process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                             env=env,start_new_session=True)
(out/'driver.pid').write_text(str(process.pid)+'\n')
record=dict(old_driver=status['pid'],old_child=status['child_pid'],new_driver=process.pid,
             changed='host RAM cache only; epoch/model/optimizer/RNG restored',unix_time=time.time(),command=command)
(out/'ram_restart.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
