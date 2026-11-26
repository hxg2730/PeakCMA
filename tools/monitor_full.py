"""Emit lightweight progress until the detached full experiment finishes."""
import json
import subprocess
import sys
import time

while True:
    result=json.loads(subprocess.check_output([sys.executable,'tools/full_status.py'],text=True))
    print(json.dumps(result),flush=True)
    if result['status'] in ('complete','failed'):
        sys.exit(0 if result['status']=='complete' else 1)
    time.sleep(30)
