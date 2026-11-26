"""Read-only SSH progress stream; password stays in memory only."""
import getpass
import json
import sys
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
               allow_agent=False,look_for_keys=False,timeout=30)
del password
client.get_transport().set_keepalive(30)
REMOTE_ROOT=os.environ.get('PEAKCMA_REMOTE_ROOT')
if not REMOTE_ROOT: raise SystemExit('请设置环境变量 PEAKCMA_REMOTE_ROOT（远端项目根目录）')
_,output,error=client.exec_command('cd '+REMOTE_ROOT+' && .venv/bin/python -u tools/monitor_full.py')
for line in output:
    if '--compact' in sys.argv:
        value=json.loads(line)
        training=value.get('training',{})
        slim={'status':value.get('status'),'stage':value.get('stage'),
              'epoch':training.get('epoch'),'validation':training.get('validation'),
              'best_validation_mse':training.get('best_validation_mse'),
              'completed_pairs':value.get('completed_pairs'), 'gpu':value.get('gpu')}
        print(json.dumps(slim),flush=True)
    else: print(line,end='',flush=True)
print(error.read().decode('utf-8',errors='replace'),end='',flush=True)
status=output.channel.recv_exit_status()
client.close()
raise SystemExit(status)
