"""Persist stage/epoch timestamps without changing the training process."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

root=Path('Results/weather_full'); previous=None
with (root/'timeline.jsonl').open('a',encoding='utf-8') as log:
    while True:
        status=json.loads(subprocess.check_output([sys.executable,'tools/full_status.py'],text=True))
        signature=(status.get('status'),status.get('stage'),status.get('cache',{}).get('global_completed',0)//1000,
                   status.get('training',{}).get('epoch'))
        if signature!=previous:
            status['observed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
            log.write(json.dumps(status)+'\n'); log.flush(); previous=signature
        if status.get('status') in ('complete','failed'): break
        time.sleep(5)
