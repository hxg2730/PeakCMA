"""Wait for the existing full run, download its package and verify every artifact."""
import getpass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import time
import zipfile
import paramiko

PROJECT = Path(__file__).resolve().parents[1]
LOCAL = PROJECT / 'Results/weather_full'
# 远端连接信息全部来自环境变量，仓库中不保存任何服务器地址或账号。
REMOTE_HOST = os.environ.get('PEAKCMA_SSH_HOST')
REMOTE_PORT = int(os.environ.get('PEAKCMA_SSH_PORT', '22'))
REMOTE_USER = os.environ.get('PEAKCMA_SSH_USER', 'root')
REMOTE_ROOT = os.environ.get('PEAKCMA_REMOTE_ROOT')
if not (REMOTE_HOST and REMOTE_ROOT):
    raise SystemExit('请设置 PEAKCMA_SSH_HOST 与 PEAKCMA_REMOTE_ROOT（可选 PEAKCMA_SSH_PORT / PEAKCMA_SSH_USER）')
REMOTE = str(PurePosixPath(REMOTE_ROOT) / 'Results/weather_full')


def collect(client):
    sftp = client.open_sftp()
    print('COLLECTOR_READY', flush=True)
    while True:
        with sftp.open(REMOTE + '/status.json') as stream:
            status = json.load(stream)
        if status['status'] == 'failed': raise RuntimeError(status)
        try:
            with sftp.open(REMOTE + '/package_manifest.json') as stream:
                package = json.load(stream)
            assert status['status'] == 'complete'
            break
        except FileNotFoundError:
            time.sleep(30)
    filename = package['file']
    assert PurePosixPath(filename).name == filename
    target = LOCAL / filename
    temp = target.with_suffix('.zip.part')
    progress = [0]
    def report(done, total):
        if done - progress[0] >= 32 * 1024**2 or done == total:
            print(f'DOWNLOAD {done}/{total}', flush=True)
            progress[0] = done
    sftp.get(REMOTE + '/' + filename, str(temp), callback=report)
    with temp.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert actual == package['sha256'], 'Package SHA-256 mismatch'
    temp.replace(target)
    with zipfile.ZipFile(target) as archive:
        manifest = json.loads(archive.read('Results/weather_full/artifact_manifest.json'))
        for name, expected in manifest.items():
            with archive.open(name) as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            assert actual == expected['sha256'], 'Artifact SHA-256 mismatch: ' + name
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            assert not path.is_absolute() and '..' not in path.parts
            # Only extract experiment outputs. The package carries the complete source separately.
            if path.parts[:2] == ('Results', 'weather_full'):
                archive.extract(member, PROJECT)
    (LOCAL / 'package_manifest.json').write_text(json.dumps(package, indent=2), encoding='utf-8')
    result = dict(status='complete', package_sha256_verified=True,
                  artifact_sha256_verified=len(manifest), extracted_results=True,
                  package=filename, bytes=target.stat().st_size)
    (LOCAL / 'collection_status.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('COLLECTION_COMPLETE ' + json.dumps(result), flush=True)
    sftp.close()


def main():
    # Keep the supplied credential in this process's memory only, for transient reconnects.
    password = getpass.getpass('SSH password: ')
    while True:
        client = paramiko.SSHClient()
        client.load_host_keys(str(PROJECT / '.ssh/known_hosts'))
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        try:
            client.connect(REMOTE_HOST, port=REMOTE_PORT, username=REMOTE_USER, password=password,
                           allow_agent=False, look_for_keys=False, timeout=10, auth_timeout=30)
            client.get_transport().set_keepalive(30)
            collect(client)
            return
        except (TimeoutError, ConnectionError, EOFError, paramiko.SSHException) as error:
            if isinstance(error, (paramiko.AuthenticationException, paramiko.BadHostKeyException)):
                raise
            state = dict(status='waiting_for_connection', checked_unix=time.time(),
                         error=type(error).__name__, retry_seconds=30,
                         training_status='Unverified while SSH is unreachable')
            (LOCAL / 'collection_status.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
            print('RECONNECT_PENDING ' + json.dumps(state), flush=True)
        finally:
            client.close()
        time.sleep(30)


if __name__ == '__main__':
    main()
