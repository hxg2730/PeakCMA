"""Precompute a later seed; stop before the sequential driver can reach it."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--handoff_stage', required=True)
    a = p.parse_args()
    project = Path(__file__).resolve().parents[1]
    os.chdir(project)
    root = Path('Results/weather_full')
    os.environ.update(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', PYTHONIOENCODING='utf-8')
    stages = json.loads((root / 'commands.json').read_text())['stages']
    selected = [(name, args) for name, args in stages if name.startswith(str(a.seed) + '_')]
    assert len(selected) == 2
    stage_names = [name for name, _ in stages]
    barrier = stage_names.index(a.handoff_stage)
    assert barrier < stage_names.index(selected[0][0])
    record = root / f'parallel_seed{a.seed}.json'
    state = dict(pid=os.getpid(), seed=a.seed, handoff_stage=a.handoff_stage,
                 started_unix=time.time(), status='running', stages=selected,
                 mathematical_configuration='Exact arguments from commands.json; independent process RNG')
    def save():
        temp = record.with_suffix('.json.tmp')
        temp.write_text(json.dumps(state, indent=2)); temp.replace(record)
    def handoff():
        current = json.loads((root / 'status.json').read_text())
        if current['status'] != 'running': return True
        return stage_names.index(current['stage']) >= barrier
    save()
    for name, args in selected:
        if handoff():
            state.update(status='handed_off', finished_unix=time.time()); save(); return
        with (root / (name + '.log')).open('a') as log:
            child = subprocess.Popen([sys.executable, *args], stdout=log, stderr=subprocess.STDOUT)
            state.update(stage=name, child_pid=child.pid); save()
            while child.poll() is None:
                if handoff():
                    # Only terminate our own child. Atomic epoch checkpoints retain completed work.
                    child.terminate()
                    child.wait(timeout=60)
                    state.update(status='handed_off', finished_unix=time.time()); save(); return
                time.sleep(1)
            if child.returncode:
                state.update(status='failed', exit_code=child.returncode); save()
                raise RuntimeError(name + ' failed')
    state.update(status='complete', finished_unix=time.time()); save()


if __name__ == '__main__':
    main()
