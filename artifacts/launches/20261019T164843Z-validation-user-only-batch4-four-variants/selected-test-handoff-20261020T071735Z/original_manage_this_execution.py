"""Finite four-condition execution record; invokes only the existing frozen CLI."""
import csv, hashlib, json, math, os, signal, subprocess, time, traceback, urllib.request
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

HERE=Path(__file__).resolve().parent
STATE=json.loads((HERE/'suite.json').read_text())
REPORT=Path(STATE['report_dir']); FROZEN=Path(STATE['frozen_project'])
ACTIVE={}; RESTARTS={0:0,1:0}; STOP=False

def now(): return datetime.now(timezone.utc).isoformat()
def write(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)
def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return default

def rows(out,name):
    path=out/(name+'.jsonl')
    if not path.exists():return []
    data=[]
    for line in path.open():
        try:data.append(json.loads(line))
        except json.JSONDecodeError:break  # a concurrent append can expose an unfinished last line
    return data

def event(kind,**details):
    record={'at':now(),'event':kind,**details}
    with (HERE/'events.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)

def stop_process(pid,expected):
    try:
        command=Path(f'/proc/{pid}/cmdline').read_bytes().decode()
        if not command:return
        assert expected in command and os.getpgid(pid)==pid
        os.killpg(pid,signal.SIGTERM)
    except (FileNotFoundError,ProcessLookupError):pass

def server_ready(server):
    try:
        with urllib.request.urlopen(server['base_url']+'/models',timeout=3) as response:
            models=json.load(response)['data']
        return len(models)==1 and models[0]['id']=='qwen3-8b-pinned' and models[0]['max_model_len']==40960
    except Exception:return False

def ensure_server(gpu):
    server=STATE['servers'][gpu]
    if server_ready(server):return True
    if RESTARTS[gpu]>=1:return False
    RESTARTS[gpu]+=1
    manifest=read(server['manifest'])
    history=HERE/'server_history';history.mkdir(exist_ok=True)
    (history/f"gpu{gpu}-pid{manifest['pid']}.json").write_bytes(Path(server['manifest']).read_bytes())
    stop_process(manifest['pid'],'vllm.entrypoints.openai.api_server');time.sleep(3)
    with (Path(server['directory'])/'server.log').open('ab') as log:
        process=subprocess.Popen(manifest['command'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),VLLM_USE_V1='1',TOKENIZERS_PARALLELISM='false'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    manifest.update(pid=process.pid,started_at=now(),infrastructure_restart=RESTARTS[gpu])
    write(server['manifest'],manifest);server['pid']=process.pid
    event('server_restarted_for_remaining_conditions',gpu=gpu,pid=process.pid)
    for _ in range(60):
        if server_ready(server):return True
        if process.poll() is not None:return False
        time.sleep(2)
    return False

def audit(c):
    out=Path(c['run_dir']);manifest=read(out/'manifest.json');result=read(out/'result.json')
    assert manifest['status']=='complete' and result['iterations_completed']==5
    assert manifest['source']['files_sha256']==STATE['source_files_sha256']
    captured=rows(out,'trajectories');decisions=rows(out,'decisions')
    assert len(captured)==60 and len(decisions)==5
    schedule=read(out/'plan.json')['train_minibatch_schedule']
    assert len({t['example_id'] for t in captured})==20
    assert len({(t['example_id'],t['repeat_index']) for t in captured})==60
    parent=read(out/'config.json')['initial_prompt']
    for iteration,decision in enumerate(decisions):
        group=[t for t in captured if t['iteration']==iteration]
        assert len(group)==12 and {t['example_id'] for t in group}==set(schedule[iteration])
        assert {t['prompt'] for t in group}=={parent} and decision['parent_prompt']==parent
        if decision['accepted']:assert decision['selected_validation_accuracy']>=decision['parent_validation_accuracy']
        parent=decision['selected_prompt']
    assert parent==result['selected_prompt']
    assert result['total_cost']['task_calls']==len(captured)+sum(v['rollouts'] for v in rows(out,'validation_scores') if v['phase']!='base')+sum(t['rollouts'] for k,t in result['test'].items() if not t.get('reused_initial'))
    expected_val={json.loads(line)['id'] for line in Path(read(out/'config.json')['benchmark']['validation']).open()}
    val_records=rows(out,'validation');scores=rows(out,'validation_scores')
    for summary in scores:
        records=[r for r in val_records if r['iteration']==summary['iteration'] and r['phase']==summary['phase'] and r['prompt']==summary['prompt']]
        assert len(records)==summary['rollouts']==summary['questions']==300
        assert {r['example_id'] for r in records}==expected_val
        assert sum(r['reward'] for r in records)/300==summary['accuracy']
    references={(t['example_id'],t['repeat_index']):t for t in captured}
    likelihood_records=0;lp={}
    path=out/'likelihood.jsonl'
    if path.exists():
        for line in path.open():
            r=json.loads(line);t=references[(r['example_id'],r['repeat_index'])]
            assert t['editor_eligible'] and r['response_ids']==t['response_ids']
            assert len(r['token_logprobs'])==len(t['response_ids']) and all(math.isfinite(v) for v in r['token_logprobs'])
            assert math.isclose(math.fsum(r['token_logprobs'])/len(r['token_logprobs']),r['mean_logprob'],abs_tol=1e-12)
            lp[(r['prompt'],r['example_id'],r['repeat_index'])]=r['token_logprobs'];likelihood_records+=1
    by_iteration={i:next(t['prompt'] for t in captured if t['iteration']==i) for i in range(5)}
    for edit in rows(out,'edits'):
        for t in edit['trajectories']:
            key=(t['example_id'],t['repeat_index']);positions=t['selected_token_indices']
            if not positions:continue
            old=lp[(by_iteration[edit['iteration']],*key)];new=lp[(edit['instruction'],*key)]
            assert math.isclose(math.fsum(old[i] for i in positions)/len(positions),t['parent_mean_logprob'],abs_tol=1e-12)
            assert math.isclose(math.fsum(new[i] for i in positions)/len(positions),t['candidate_mean_logprob'],abs_tol=1e-12)
    tests=rows(out,'test');expected_test={json.loads(line)['id'] for line in Path(read(out/'config.json')['benchmark']['test']).open()}
    for label,test in result['test'].items():
        assert test['questions']==test['rollouts']==300
        if test.get('reused_initial'):continue
        records=[r for r in tests if r['condition']==label]
        assert len(records)==300 and {r['example_id'] for r in records}==expected_test
        assert sum(r['reward'] for r in records)/300==test['accuracy']
    exports=REPORT/c['name']/'editor_calls';exports.mkdir(parents=True,exist_ok=True)
    for index,call in enumerate(rows(out,'optimizer_calls')):
        stem=f"{index:02d}-round{call['iteration']+1}-{call['phase']}"
        write(exports/(stem+'-messages.json'),call['messages'])
        (exports/(stem+'-response.txt')).write_text(call['text']+'\n')
    return {'status':'passed','captured_rollouts':60,'distinct_questions':20,'iterations':5,'likelihood_records_verified':likelihood_records,'all_validation_and_test_rows':True,'frozen_source_verified':True}

def refresh():
    summaries=[]
    for c in STATE['conditions']:
        out=Path(c['run_dir']);result=read(out/'result.json',{})
        decisions=rows(out,'decisions');validations=rows(out,'validation_scores');edits=rows(out,'edits');groups=rows(out,'proposal_groups');calls=rows(out,'optimizer_calls');evidence=rows(out,'editor_evidence')
        unique={}
        for e in edits:
            for t in e['trajectories']:unique[(e['iteration'],t['example_id'],t['repeat_index'])]=t
        compositions=Counter('mixed' if g['successes'] and g['failures'] else 'success_only' if g['successes'] else 'failure_only' for g in groups if g['included'])
        baseline=next((v['accuracy'] for v in validations if v['phase']=='base'),None)
        final=result.get('validation_accuracy',decisions[-1]['selected_validation_accuracy'] if decisions else baseline)
        denominator=sum(t['total_tokens'] for t in unique.values())
        summary={'condition':c['name'],'gpu':c['gpu'],'status':c['status'],'run_id':c['run_id'],'iterations':len(decisions),'accepted_updates':sum(d['accepted'] for d in decisions),'base_validation':baseline,'final_validation':final,'initial_test':result.get('test',{}).get('initial',{}).get('accuracy'),'selected_test':result.get('test',{}).get('selected',{}).get('accuracy'),'proposal_measured':sum(e['phase']=='proposal' for e in edits),'proposal_verified':sum(e['phase']=='proposal' and e['verified'] for e in edits),'refinement_measured':sum(e['phase']=='refinement' for e in edits),'refinement_verified':sum(e['phase']=='refinement' and e['verified'] for e in edits),'invalid_editor_outputs':sum(bool(call['error']) for call in calls),'retained_rollouts':sum(e['included'] for e in evidence),'captured_rollouts':sum(e['captured'] for e in evidence),'reward_compositions':dict(compositions),'selected_token_fraction':sum(t['selected_tokens'] for t in unique.values())/denominator if denominator else None,'progress':read(out/'progress.json',{}),'error':c.get('error'),'selected_prompt':result.get('selected_prompt'),'audit':c.get('audit')}
        summaries.append(summary)
    write(REPORT/'summary.json',{'suite_id':STATE['suite_id'],'status':STATE['status'],'updated_at':now(),'conditions':summaries})
    columns=['condition','status','iterations','accepted_updates','base_validation','final_validation','initial_test','selected_test','proposal_verified','proposal_measured','refinement_verified','refinement_measured','invalid_editor_outputs','retained_rollouts','captured_rollouts','selected_token_fraction','run_id']
    with (REPORT/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(summaries)
    def accuracy(v):return f'{v:.2%}' if v is not None else 'pending'
    lines=['# Validation-gated prompt optimization: four likelihood variants','',f"Status: {STATE['status']}; updated KST: {datetime.now(timezone(timedelta(hours=9))).isoformat()}",'','Pinned Qwen3-8B; DAPO seed0 train150/validation300/test300; task16K; batch4 x3;5 rounds; independent prompt chains.','',f"Baseline validation reused only after exact identity/evidence checks from `{STATE['base_run_id']}` (230/300).",'','| Condition | Status | Rounds | Adopted | Base val | Final val | Initial test | Selected test |','| --- | --- | --- | --- | --- | --- | --- | --- |']
    for s in summaries:lines.append(f"| {s['condition']} | {s['status']} | {s['iterations']}/5 | {s['accepted_updates']} | {accuracy(s['base_validation'])} | {accuracy(s['final_validation'])} | {accuracy(s['initial_test'])} | {accuracy(s['selected_test'])} |")
    lines+=['','Each question verifies only its own retained fixed response IDs. Validation selects one winner per round; test is frozen and never used for selection.','Same problem schedule does not imply identical stochastic responses across conditions. Centered groups without both reward classes are uninformative. Branch/response masks are fixed under the parent. No scientific run is retried, shortened or silently repaired.','']
    for s in summaries:
        lines += [f"## {s['condition']}",'',f"Run: `{s['run_id']}`",'',f"Proposal verified/measured: {s['proposal_verified']}/{s['proposal_measured']}; refinement: {s['refinement_verified']}/{s['refinement_measured']}; invalid editor outputs: {s['invalid_editor_outputs']}.",f"Retained/captured responses: {s['retained_rollouts']}/{s['captured_rollouts']}; reward compositions: {s['reward_compositions']}; selected-token fraction: {s['selected_token_fraction']}.",'',f"Progress: `{json.dumps(s['progress'])}`",'']
        condition=next(c for c in STATE['conditions'] if c['name']==s['condition'])
        chain=rows(Path(condition['run_dir']),'decisions')
        if chain:
            lines += ['| Round | Accepted | Parent val | Selected val | Validated candidates |','| --- | --- | --- | --- | --- |']
            for d in chain:
                lines.append(f"| {d['iteration']+1} | {d['accepted']} | {accuracy(d['parent_validation_accuracy'])} | {accuracy(d['selected_validation_accuracy'])} | {len(d['validation_candidates'])} |")
            lines.append('')
        if s['error']:lines += ['Error: '+s['error'],'']
        if s['selected_prompt']:lines += ['Final instruction:','','```text',s['selected_prompt'],'```','']
        if s['audit']:lines += ['Integrity audit: `'+json.dumps(s['audit'])+'`','']
    lines+=['[Machine-readable summary](summary.json) · [Comparison CSV](comparison.csv) · [Suite configuration](suite.json)','', 'Original prompts, raw responses/IDs, per-token likelihoods, candidate validation and adoption decisions remain in each run. Source and exact configurations were frozen before launch.']
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n')
    write(HERE/'suite.json',STATE);write(REPORT/'suite.json',STATE)

def start(c):
    assert hashlib.sha256(Path(c['execution_config']).read_bytes()).hexdigest()==c['config_file_sha256']
    command=['uv','run','--no-sync','python','-m','prompt_policy',c['execution_config'],'--output',c['run_dir']]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(FROZEN/'src'),PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
    log_path=HERE/(c['name']+'.log')
    with log_path.open('ab') as log:process=subprocess.Popen(command,cwd=STATE['root'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    c.update(status='running',pid=process.pid,command=command,log=str(log_path),started_at=now())
    ACTIVE[c['gpu']]=(c,process,time.time());event('condition_started',condition=c['name'],gpu=c['gpu'],pid=process.pid)

def interrupted(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
try:
    STATE.update(status='running',controller_pid=os.getpid(),started_at=now())
    for gpu in (0,1):
        pending=next(c for c in STATE['conditions'] if c['gpu']==gpu and c['status']=='planned')
        if ensure_server(gpu):start(pending)
        else:pending.update(status='failed',error='Server readiness failed before launch')
    refresh()
    while any(c['status'] in {'planned','running'} for c in STATE['conditions']):
        for gpu in (0,1):
            entry=ACTIVE.get(gpu)
            if entry:
                c,process,started=entry;out=Path(c['run_dir']);progress=out/'progress.json'
                last=progress.stat().st_mtime if progress.exists() else started
                if process.poll() is None and time.time()-last>5400:
                    event('stalled_condition_terminated',condition=c['name'],stale_seconds=time.time()-last)
                    stop_process(process.pid,'prompt_policy');c['error']='No scientific progress for90 minutes; terminated without retry'
                code=process.poll()
                if code is None:continue
                c.update(returncode=code,finished_at=now());manifest=read(out/'manifest.json',{})
                if code==0 and manifest.get('status')=='complete':
                    try:c['audit']=audit(c);c['status']='complete';write(out/'completion_audit.json',c['audit'])
                    except Exception:c.update(status='audit_failed',error=traceback.format_exc())
                else:
                    c.update(status='failed',error=c.get('error') or manifest.get('error') or f'CLI exited{code}')
                    if manifest.get('status')=='running':manifest.update(status='interrupted',error=c['error']);write(out/'manifest.json',manifest)
                event('condition_finished',condition=c['name'],status=c['status'],returncode=code)
                del ACTIVE[gpu]
            if gpu not in ACTIVE:
                pending=next((c for c in STATE['conditions'] if c['gpu']==gpu and c['status']=='planned'),None)
                if pending:
                    if ensure_server(gpu):start(pending)
                    else:pending.update(status='failed',error='Server unavailable after one infrastructure restart')
        try:refresh()
        except Exception:event('report_refresh_error',error=traceback.format_exc())
        if any(c['status'] in {'planned','running'} for c in STATE['conditions']):time.sleep(30)
    STATE['status']='complete' if all(c['status']=='complete' for c in STATE['conditions']) else 'finished_with_errors'
except BaseException:
    STATE.update(status='interrupted',controller_error=traceback.format_exc())
    for c,process,_ in ACTIVE.values():
        stop_process(process.pid,'prompt_policy');c.update(status='interrupted',error=STATE['controller_error'])
    for c in STATE['conditions']:
        if c['status']=='planned':c['status']='cancelled'
finally:
    shutdown=[]
    for server in STATE['servers']:
        try:stop_process(server['pid'],'vllm.entrypoints.openai.api_server');shutdown.append({'gpu':server['gpu'],'pid':server['pid'],'signal':'SIGTERM'})
        except Exception:shutdown.append({'gpu':server['gpu'],'error':traceback.format_exc()})
    STATE.update(finished_at=now(),server_shutdown=shutdown)
    write(REPORT/'owned_server_shutdown.json',shutdown)
    try:refresh()
    except Exception:write(HERE/'final_report_error.json',{'error':traceback.format_exc()})
    event('suite_finished',status=STATE['status'])
