"""Compute per-variable baseline-relative indices for matched runs."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('baseline',type=Path,help='baseline metrics.json')
p.add_argument('peak',type=Path,help='peak metrics.json')
a=p.parse_args()
b=json.loads(a.baseline.read_text()); r=json.loads(a.peak.read_text())
result={}
for metric in ('neighbourhood','amplitude','timing','rest'):
    values=[]
    for channel in (k for k in r if k.isdigit()):
        numerator=r[channel][metric]; denominator=b[channel][metric]
        if numerator is not None and denominator is not None and denominator>0:
            values.append(100*numerator/denominator)
    result[metric]=sum(values)/len(values) if values else None
print(json.dumps(result,indent=2))
