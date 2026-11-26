"""Interactive SSH bridge: credential read with getpass, never persisted."""
import getpass
import json
import sys
import time
from pathlib import Path
import os
import paramiko

# 远端连接信息全部来自环境变量，仓库中不保存任何服务器地址或账号。
REMOTE_HOST = os.environ.get('PEAKCMA_SSH_HOST')
REMOTE_PORT = int(os.environ.get('PEAKCMA_SSH_PORT', '22'))
REMOTE_USER = os.environ.get('PEAKCMA_SSH_USER', 'root')
if not REMOTE_HOST:
    raise SystemExit('请设置环境变量 PEAKCMA_SSH_HOST（可选 PEAKCMA_SSH_PORT / PEAKCMA_SSH_USER）')

client=paramiko.SSHClient()
client.load_host_keys(str(Path('.ssh/known_hosts')))
client.set_missing_host_key_policy(paramiko.RejectPolicy())
password=getpass.getpass('SSH password: ')
client.connect(REMOTE_HOST,port=REMOTE_PORT,username=REMOTE_USER,password=password,
               allow_agent=False,look_for_keys=False,timeout=30,auth_timeout=60)
del password
client.get_transport().set_keepalive(30)
print('SSH_READY',flush=True)
for line in sys.stdin:
    try:
        request=json.loads(line)
        if request['action']=='exec':
            channel=client.get_transport().open_session()
            channel.exec_command(request['command'])
            start=time.monotonic()
            while True:
                if channel.recv_ready(): print(channel.recv(65536).decode('utf-8',errors='replace'),end='',flush=True)
                if channel.recv_stderr_ready(): print(channel.recv_stderr(65536).decode('utf-8',errors='replace'),end='',flush=True)
                if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready(): break
                if time.monotonic()-start>request.get('timeout',1200): raise TimeoutError('Remote command timeout')
                time.sleep(.05)
            print('\nSSH_DONE '+str(channel.recv_exit_status()),flush=True)
        elif request['action'] in ('upload','download'):
            sftp=client.open_sftp()
            last=[0]
            def progress(done,total):
                if done-last[0]>=32*1024**2 or done==total:
                    print(f'TRANSFER {done}/{total}',flush=True); last[0]=done
            if request['action']=='upload': sftp.put(request['local'],request['remote'],callback=progress)
            else: sftp.get(request['remote'],request['local'],callback=progress)
            sftp.close(); print('TRANSFER_DONE',flush=True)
        elif request['action']=='close': break
    except Exception as error:
        print('SSH_ERROR '+type(error).__name__+': '+str(error),flush=True)
client.close()
