"""Package completed, verified Weather experiments without caches or credentials."""
import hashlib
import json
from pathlib import Path
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / 'Results/weather_full'


def main():
    status = json.loads((ROOT / 'status.json').read_text(encoding='utf-8'))
    report = json.loads((ROOT / 'final_report.json').read_text(encoding='utf-8'))
    stats = json.loads((ROOT / 'full_statistics.json').read_text(encoding='utf-8'))
    assert status['status'] == stats['status'] == 'complete'
    seeds = report['configuration']['seeds']
    assert len(stats['seed_rows']) == len(seeds) == 6
    files = set()
    for seed in seeds:
        run = report['runs'][str(seed)]
        weight = PROJECT / run['weight']
        assert run['reload_prediction_verified']
        assert hashlib.sha256(weight.read_bytes()).hexdigest() == run['sha256']
        files.add(weight)
        folder = ROOT / f'training/Weather/96_192/{seed}'
        for mode in ('baseline', 'peak'):
            for name in ('best_model.pth', 'history.json', 'metrics.json', 'data_manifest.json'):
                path = folder / mode / name
                assert path.is_file(), path
                files.add(path)
    # Fixed allowlist: never package SSH files, credentials, environments or old trials.
    files.update(p for p in ROOT.iterdir() if p.is_file()
                 and p.suffix in ('.json', '.jsonl', '.log', '.csv', '.md')
                 and p.name not in ('artifact_manifest.json', 'package_manifest.json')
                 and not p.name.startswith(('partial_', 'individual_', 'remote_run')))
    for name in ('README.md', 'LICENSE.txt', 'requirements.txt', 'train.py'):
        path = PROJECT / name
        if path.exists(): files.add(path)
    for directory in ('models', 'data_provider', 'storage', 'utils', 'layers', 'tests', 'scripts', 'tools'):
        for path in (PROJECT / directory).rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.sh') and '__pycache__' not in path.parts:
                files.add(path)
    for name in ('Weather.csv', 'Weather.source.json'):
        files.add(PROJECT / 'dataset' / name)
    manifest = {str(path.relative_to(PROJECT)).replace('\\', '/'):
                dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                for path in sorted(files)}
    (ROOT / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    files.add(ROOT / 'artifact_manifest.json')
    target = ROOT / 'PeakCMA_Weather_full_experiments.zip'
    temp = target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for path in sorted(files):
            archive.write(path, str(path.relative_to(PROJECT)).replace('\\', '/'))
    temp.replace(target)
    result = dict(file=target.name, bytes=target.stat().st_size,
                  sha256=hashlib.sha256(target.read_bytes()).hexdigest(), files=len(files),
                  seeds=seeds, all_export_predictions_verified=True,
                  remote_only=['Embeddings/PeakCMA', 'last_training.pth', 'test_predictions.npz'])
    (ROOT / 'package_manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
