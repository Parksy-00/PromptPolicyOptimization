"""One authorized execution: wait for the named suite, serve GPU0, run frozen CLI, clean up owned server."""
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
STATE=json.loads((HERE/'execution.json').read_text())
ROOT=Path(STATE['root']); RUN=Path(STATE['run_dir']); SERVER=Path(STATE['server_dir'])
server_process=None; run_process=None

def now():return datetime.now(timezone.utc).isoformat()

def save(status=None,**fields):
    if status:STATE['status']=status
    STATE.update(updated_at=now(),**fields)
    temporary=HERE/'execution.json.tmp'
    temporary.write_text(json.dumps(STATE,indent=2)+'\n')
    temporary.replace(HERE/'execution.json')

def read(path):return json.loads(Path(path).read_text())

def terminate_owned(process):
    if process is None:return
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=30)

def interrupt(signum,frame):raise KeyboardInterrupt(f'signal {signum}')

signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
save(worker_pid=os.getpid())
try:
    dependency=Path(STATE['dependency_report'])
    while True:
        summary=read(dependency/'summary.json')
        if summary['status'] in {'complete','finished_with_errors'} and (dependency/'owned_server_shutdown.json').exists():break
        if summary['status']=='interrupted':raise RuntimeError('Prior suite interrupted; do not start a dependent GPU job automatically')
        save('waiting_for_existing_suite',dependency_status=summary['status'],dependency_updated_at=summary['updated_at'])
        time.sleep(30)
    save('waiting_for_gpu_release',dependency_status=summary['status'])
    while True:
        used=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if used<2000:break
        save(gpu0_memory_used_mib=used)
        time.sleep(15)
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',8018))==0:raise RuntimeError('Port8018 is already occupied; do not share an unowned server')
    manifest=read(STATE['server_template'])
    command=list(manifest['command']);command[command.index('--port')+1]='8018'
    settings=STATE['scientific_parameters']
    assert manifest['model_path']==settings['model']['name_or_path']
    assert manifest['revision']==settings['model']['revision']
    save('starting_server',server_command=command)
    with (SERVER/'server.log').open('ab') as log:
        server_process=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',VLLM_USE_V1='1',TOKENIZERS_PARALLELISM='false'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    manifest.update(pid=server_process.pid,command=command,base_url='http://127.0.0.1:8018/v1',started_at=now(),purpose='User-approved GEPA fixed-parent likelihood verification/refinement; native3 questions x1 rollout;5 iterations.')
    (SERVER/'server.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save(server_pid=server_process.pid)
    deadline=time.monotonic()+1800
    while True:
        if server_process.poll() is not None:raise RuntimeError(f'Owned server exited with {server_process.returncode}')
        try:
            with urllib.request.urlopen('http://127.0.0.1:8018/v1/models',timeout=5) as response:models=json.load(response)
            matches=[m for m in models['data'] if m['id']=='qwen3-8b-pinned']
            assert len(matches)==1 and matches[0]['root']==manifest['model_path'] and matches[0]['max_model_len']==40960
            break
        except (OSError,AssertionError,KeyError):
            if time.monotonic()>deadline:raise RuntimeError('Server not ready after30 minutes; no automatic restart')
            save('starting_server');time.sleep(5)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(Path(STATE['frozen_project'])/'src'),PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
    command=['uv','run','--no-sync','python','-m','prompt_policy',STATE['config'],'--output',str(RUN)]
    with (HERE/'run.log').open('ab') as log:
        run_process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save('running',run_pid=run_process.pid,run_command=command,started_at=now())
    while run_process.poll() is None:
        fields={}
        if (RUN/'progress.json').exists():fields['progress']=read(RUN/'progress.json')
        save('running',**fields);time.sleep(30)
    if run_process.returncode!=0:raise RuntimeError(f'GEPA CLI exited with {run_process.returncode}; see run.log and manifest')
    completed=read(RUN/'manifest.json');result=read(RUN/'result.json');selection=read(RUN/'selection.json')
    assert completed['status']=='complete' and selection['prompt']==result['selected_prompt']
    assert result['search_cost']['task_calls']<=1830
    for label in ['initial','selected']:
        assert result['test'][label]['questions']==300 and result['test'][label]['rollouts']==300
    for relative,expected in completed['source']['files_sha256'].items():
        actual=hashlib.sha256((RUN/'source_snapshot'/relative).read_bytes()).hexdigest()
        frozen=hashlib.sha256((Path(STATE['frozen_project'])/relative).read_bytes()).hexdigest()
        assert actual==frozen==expected
    audit={'status':'passed','frozen_source_verified':True,'test_questions_per_condition':300,'search_task_calls':result['search_cost']['task_calls']}
    (RUN/'completion_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    save('complete',finished_at=now(),result={'validation_accuracy':result['validation_accuracy'],'initial_test':result['test']['initial']['accuracy'],'selected_test':result['test']['selected']['accuracy']},completion_audit=audit)
except BaseException as error:
    save('interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=traceback.format_exc(),finished_at=now())
    traceback.print_exc()
finally:
    try:
        if run_process is not None and run_process.poll() is None:terminate_owned(run_process)
        terminate_owned(server_process)
        save(owned_server_shutdown_at=now() if server_process is not None else None)
    except BaseException:save(cleanup_error=traceback.format_exc())
