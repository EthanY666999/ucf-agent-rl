"""Authorized P1/P2 insertion at the S06 boundary; one GPU owner, durable journals.

The pilot limit is a command-line stop at 150 with cfg.steps=400.  The selected
arm resumes its SAME directory/configuration after S07 and S10.  No pilot test
split is evaluated and no pilot checkpoint is pruned here.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


class AdmissionError(RuntimeError):
    """Source or admission evidence changed; no experiment may launch."""


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def process_identity(pid, proc_root=Path('/proc')):
    try:
        # comm may contain spaces; starttime is field 22, tail starts at field 3.
        return (proc_root / str(pid) / 'stat').read_text().rpartition(') ')[2].split()[19]
    except FileNotFoundError:
        return None


def same_process(record, proc_root=Path('/proc')):
    if not record.get('pid') or record.get('start_ticks') is None:
        return False
    return process_identity(record['pid'], proc_root) == str(record['start_ticks'])


def verify_ready(state, check_git=True):
    state = Path(state)
    ready = read(state / 'ready.json')
    plan = read(state / 'pilot_plan.json')
    assert ready['passed'] is True and plan['schema_version'] == 1
    assert sha(state / 'pilot_plan.json') == ready['pilot_plan_sha256']
    assert ready['files_sha256'], 'Empty readiness manifest'
    for path, digest in ready['files_sha256'].items():
        assert sha(path) == digest, 'Readiness file changed: ' + path
    cpu = ready['cpu_acceptance']
    assert sha(cpu['path']) == cpu['sha256']
    assert read(cpu['path'])['passed'] is True
    admissions=ready['admission_receipts']
    assert admissions, 'Inherited environment/family admission receipts are required'
    admitted_paths=set()
    for item in admissions:
        assert item['role'] and sha(item['path']) == item['sha256']
        assert read(item['path'])['passed'] is True, item['role']
        admitted_paths.add(str(Path(item['path'])))
    source=Path(plan['source_root']); runs=Path(plan.get('runs_root','/data/runs'))
    inherited=runs/'queue_q35_hard/ucf_v22_20261109'
    required={str(runs/'hard_queue/S02_SFT_seed42/verification.json'),
              str(runs/'datasets/sokoban/cpu_acceptance.json'),
              str(source/'artifacts/hard_cpu_acceptance.json'),
              str(runs/'stage0/sokoban_acceptance/acceptance.json')}
    required.update(str(inherited/name) for name in ('cpu_functional.json','htsim_first_fence.json','prefix_causality.json'))
    assert required <= admitted_paths, 'Missing inherited admission records: '+str(required-admitted_paths)
    # Direct train.py launch must preserve run_ucf.py's family checks. Pin the
    # original runtime too: S07/S10 use that code and its inherited admissions.
    for runtime in {source,Path(plan['original_root'])}:
        for anchor in (runtime/'SOURCE_MANIFEST_q35.json',runtime/'artifacts/frozen_q35.json'):
            assert str(anchor) in ready['files_sha256'], 'Readiness must bind runtime source anchor: '+str(anchor)
        frozen=read(runtime/'artifacts/frozen_q35.json')
        for name,digest in frozen['source_sha256'].items():
            assert sha(runtime/name)==digest, 'Frozen pilot source changed: '+name
        for name in ('cpu_acceptance','gpu_acceptance','calibration','phase1_gate'):
            value=read(runtime/'artifacts/q35_gates'/f'{name}.json')
            assert value['passed'] is True and value['model_family']=='Qwen3.5', name
    if check_git:
        actual = subprocess.check_output(['git', '-C', str(Path(plan['source_root']).parent),
                                          'rev-parse', 'HEAD'], text=True).strip()
        assert actual == ready['source_commit']
    ids = [e['id'] for e in plan['entries']]
    assert ids == ['P1', 'P2', 'S07', 'S10'], ids
    assert len({e['output'] for e in plan['entries']}) == 4
    for entry in plan['entries']:
        cfg = read(entry['config_path'])
        assert cfg['steps'] == 400 and cfg['eval_every'] == 25
        assert str(cfg['model']).endswith('/S02_SFT_seed42/sft/final_model')
        if entry['id'] in ('P1', 'P2'):
            assert cfg['seed'] == 42 and cfg['solvability_gate'] is True
            assert cfg.get('early_exploration_floor', False) == (entry['id'] == 'P2')
            assert cfg['ucf_corrections'] == 'v22_question_loo_rejection256'
            assert not cfg.get('sokoban_exact_fail') and not cfg.get('sokoban_bfs_progress')
            assert cfg.get('ucf_selection_weight', 'v2.2') == 'v2.2'
        if entry['id'] == 'S10':
            assert cfg['seed'] == 43 and cfg['ucf_controller'] == 'v2'
            assert not cfg.get('ucf_corrections') and not cfg.get('fork_point')
    return plan


def verify_s06(plan):
    out = Path(plan['s06_output'])
    receipt = read(out / 'verification.json')
    assert receipt.get('passed') is True and receipt.get('steps') == 400
    assert read(out / 'status.json')['status'] == 'complete'
    return receipt


def archive_control_fence(plan, state):
    fence = plan['boundary_fence']
    path = Path(fence['path'])
    archive = Path(state) / 'H03_control_fence_released.json'
    if path.exists():
        assert sha(path) == fence['sha256'], 'Do not overwrite a changed/manual failure marker'
        assert read(path)['handoff_token'] == fence['token']
        assert not archive.exists(), 'Fence both live and archived; inspect before release'
        path.replace(archive)
    else:
        assert archive.exists() and sha(archive) == fence['sha256']


def pilot_verified(out):
    """Receipt for a completed 150-step pilot; never call full-run verify here."""
    out = Path(out)
    status = read(out / 'status.json')
    assert status['status'] == 'paused_checkpointed' and status['step'] == 150
    assert status['total_steps'] == 400 and (out / 'checkpoint.pt').is_file()
    assert not (out / 'final_test.json').exists() and not (out / 'final_test.jsonl').exists()
    assert not (out / 'supplemental500').exists(), 'Pilot must not evaluate final test data'
    metrics = [json.loads(line) for line in (out / 'train_metrics.jsonl').read_text().splitlines() if line]
    assert [r['step'] for r in metrics] == list(range(1, 151)), 'Missing/duplicate pilot training steps'
    for row in metrics:
        for key in ['loss', 'kl', 'grad_norm', 'step_seconds']:
            assert math.isfinite(row[key]), (row['step'], key)
    for step in range(0, 151, 25):
        value = read(out / f'validation_{step:04d}.json')
        assert value.get('split') == 'validation' and value['n_tasks'] == 100
        assert value['n_trajectories'] == 400
    return dict(passed=True, steps=150, training_budget=150, resumable_total_steps=400,
                final_test_used=False, checkpoint_retained=True,
                checkpoint_bytes=(out / 'checkpoint.pt').stat().st_size,
                metrics_sha256=sha(out / 'train_metrics.jsonl'),
                validation_sha256={str(s): sha(out / f'validation_{s:04d}.json') for s in range(0,151,25)})


def command_for(entry, pilot, resume=False):
    command = list(entry['command'])
    assert '--stop-after-step' not in command and '--resume' not in command
    if pilot:
        assert any(Path(x).name == 'train.py' for x in command), 'Pilot uses train.py stop contract'
        command += ['--stop-after-step', '150']
    if resume:
        command.append('--resume')
    return command


def reporting(plan, state, entry, upto):
    target = Path(state) / 'reports' / entry['id'] / f'step_{upto:04d}.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return read(target)
    command = [plan['python'], str(Path(plan['source_root']) / 'scripts/pilot_p1p2_report.py'),
               'report', '--run', entry['output'], '--reference', plan['reporting']['reference_path'],
               '--upto', str(upto), '--output', str(target)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    subprocess.run(command, env=env, check=True, stdout=subprocess.DEVNULL)
    result = read(target)
    # A durable event is emitted only after report + validation exist, once per step.
    with (Path(state) / 'report_events.jsonl').open('a') as f:
        f.write(json.dumps(dict(arm=entry['id'],step=upto,path=str(target),unix=time.time()),ensure_ascii=False)+'\n')
    print(target.with_suffix('.md').read_text(), flush=True)
    return result


def report_available(plan, state, entry):
    out = Path(entry['output'])
    for step in range(25, 151, 25):
        if not (out / f'validation_{step:04d}.json').exists():
            break
        try:
            reporting(plan, state, entry, step)
        except Exception as exc:
            atomic(Path(state)/'report_errors'/f'{entry["id"]}_{step:04d}.json',
                   dict(error=repr(exc),promotion_allowed=False,unix=time.time()))
            # A transient partial write can be retried; never restart the GPU task.
            break


def release_entry_marker(entry, state):
    """Only release a previously fingerprinted control fence, never an unknown failure."""
    path = Path(entry['output']) / 'needs_manual.json'
    if not path.exists():
        return
    digest = entry.get('authorized_existing_marker_sha256')
    assert digest and sha(path) == digest, 'Unrecognized marker for ' + entry['id']
    archive = Path(state) / 'released_markers' / (entry['id'] + '.json')
    archive.parent.mkdir(exist_ok=True)
    assert not archive.exists()
    path.replace(archive)


def verify_full(plan, entry):
    """Existing trusted full-run validator and storage policy, executed without GPU."""
    original = Path(plan['original_root'])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='',
               PYTHONPATH=str(original)+':'+str(original/'scripts'), OMP_NUM_THREADS='1')
    script = ('from pathlib import Path; import sys; from q35_run_receipt import verify; '
              'from q35_p0_supervisor import relocate_verified_model; '
              'p=Path(sys.argv[1]); verify(p,prune_checkpoint=True); relocate_verified_model(p)')
    subprocess.run([plan['python'],'-c',script,entry['output']],cwd=original,env=env,check=True)


def prior_pilot_receipt(state, entry):
    """A promoted arm may already be past 150 when its controller restarts."""
    promotion = Path(state) / 'S05b_promotion.json'
    out = Path(entry['output'])
    if not promotion.exists() or not (out / 'pilot150_verification.json').exists():
        return False
    value = read(promotion)
    if value['selected_arm'] != entry['id'] or value['output'] != entry['output']:
        return False
    receipt = read(out / 'pilot150_verification.json')
    assert receipt.get('passed') is True and receipt['steps'] == 150
    # The resumed metrics file necessarily changes; frozen 150-step evaluations do not.
    for step, digest in receipt['validation_sha256'].items():
        assert sha(out / f'validation_{int(step):04d}.json') == digest
    assert (Path(state) / 'reports' / entry['id'] / 'step_0150.json').exists()
    return True


def tree_hashes(root):
    root = Path(root)
    records = {}
    for path in sorted(root.rglob('*')):
        assert not path.is_symlink(), 'Paused migration refuses internal symlinks: ' + str(path)
        if path.is_file():
            records[str(path.relative_to(root))] = dict(bytes=path.stat().st_size, sha256=sha(path))
    return records


def relocate_paused_run(plan, state, entry):
    """Hash-verified copy, then preserve the logical CLI/config path via a symlink.

    Called only while this supervisor owns the GPU lock. All stages are journaled;
    no source bytes are removed until both complete copies match the manifest.
    """
    state = Path(state)
    logical = Path(entry['output'])
    target = Path(plan['promotion_storage_root']) / logical.name
    journal = state / 'storage' / (entry['id'] + '.json')
    backup = logical.with_name(logical.name + '.verified_migration_source')
    assert logical.is_absolute() and target.is_absolute() and logical != target
    assert logical not in target.parents and target not in logical.parents
    if journal.exists():
        record = read(journal)
        assert record['logical'] == str(logical) and record['destination'] == str(target)
        if record['status'] == 'complete':
            assert logical.is_symlink() and logical.resolve() == target.resolve()
            return True
        if record['status'] == 'linked_verified' and not backup.exists():
            assert logical.is_symlink() and logical.resolve() == target.resolve()
            assert tree_hashes(target) == record['files']
            record.update(status='complete',source_bytes_released=sum(v['bytes'] for v in record['files'].values()),
                          completed_unix=time.time())
            atomic(journal,record)
            return True
    else:
        assert logical.is_dir() and not logical.is_symlink() and not backup.exists()
        status = read(logical / 'status.json')
        assert status['status'] == 'paused_checkpointed' and status['step'] == 150
        assert read(logical / 'pilot150_verification.json')['passed'] is True
        record = dict(logical=str(logical), destination=str(target), status='planned',
                      files=tree_hashes(logical), authorized_phase='paused_checkpointed_150')
        atomic(journal, record)
    for attempt_path in (state / 'attempts').glob('*.json'):
        attempts = read(attempt_path)
        assert not attempts or not same_process(attempts[-1]), 'Refuse to move a live training run'
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        needed = sum(v['bytes'] for v in record['files'].values()) + int(plan.get('minimum_free_bytes',30*2**30))
        if shutil.disk_usage(target.parent).free < needed:
            atomic(state / 'promotion_pending.json', dict(reason='Insufficient destination disk; evidence retained',
                   required_bytes=needed, available_bytes=shutil.disk_usage(target.parent).free))
            return False
        assert logical.is_dir() and not logical.is_symlink()
        shutil.copytree(logical, target)
    # A crash during copy may leave a partial destination. Never overwrite unknown
    # bytes automatically: preserve both copies and let the controller continue.
    if tree_hashes(target) != record['files']:
        atomic(state / 'promotion_pending.json', dict(reason='Migration copy differs; both copies retained',
               destination=str(target), logical=str(logical)))
        return False
    source = backup if backup.exists() else logical
    assert not source.is_symlink() and tree_hashes(source) == record['files']
    record['status'] = 'copied_verified'; atomic(journal, record)
    if not backup.exists():
        logical.rename(backup)
    if not logical.exists() and not logical.is_symlink():
        logical.symlink_to(target, target_is_directory=True)
    assert logical.is_symlink() and logical.resolve() == target.resolve()
    assert tree_hashes(backup) == record['files'] and tree_hashes(target) == record['files']
    record['status'] = 'linked_verified'; atomic(journal, record)
    shutil.rmtree(backup)
    record.update(status='complete', source_bytes_released=sum(v['bytes'] for v in record['files'].values()),
                  completed_unix=time.time())
    atomic(journal, record)
    return True


def recovery_step(plan, entry, pilot):
    """Only resume a durable metric prefix; do not discard evidence after the checkpoint."""
    out=Path(entry['output'])
    command=[plan['python'],'-c',
             'import torch,sys; s=torch.load(sys.argv[1],map_location="cpu",weights_only=False,mmap=True); print(int(s["step"]))',
             str(out/'checkpoint.pt')]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1')
    step=int(subprocess.check_output(command,env=env,text=True).strip())
    metrics=[json.loads(x) for x in (out/'train_metrics.jsonl').read_text().splitlines() if x]
    assert [r['step'] for r in metrics]==list(range(1,step+1)), 'Metrics are not exactly the checkpoint prefix; preserve evidence, no automatic resume'
    assert step <= (150 if pilot else 400), 'Checkpoint exceeds this run phase'
    return step


def run_entry(plan, state, entry, gpu_lock, pilot, promote=False):
    out = Path(entry['output'])
    journal = Path(state) / 'attempts' / (entry['id'] + ('_400' if promote else '') + '.json')
    attempts = read(journal) if journal.exists() else []
    if pilot and prior_pilot_receipt(state, entry):
        return True
    if (out/'status.json').exists():
        current = read(out/'status.json')
        if pilot and current.get('status') == 'paused_checkpointed' and current.get('step') == 150:
            receipt = pilot_verified(out)
            atomic(out/'pilot150_verification.json',receipt)
            report_available(plan,state,entry)
            return True
        if not pilot and current.get('status') == 'complete':
            verify_full(plan,entry)
            return True
    release_entry_marker(entry,state)
    for attempt in range(len(attempts), 2):
        if attempts and same_process(attempts[-1]):
            raise RuntimeError('Previous training child still alive; refusing duplicate launch')
        resume = promote or bool(attempts)
        if resume and not (out/'checkpoint.pt').exists():
            atomic(Path(state)/'failures'/f'{entry["id"]}.json',dict(reason='No recoverable checkpoint',preserved=True))
            return False
        if resume:
            checkpoint_step=recovery_step(plan,entry,pilot)
        else:
            checkpoint_step=None
        if out.exists() and not resume and (out/'train_metrics.jsonl').exists():
            raise RuntimeError('Unjournaled existing run; refusing overwrite')
        out.mkdir(parents=True,exist_ok=True)
        minimum = int(plan.get('minimum_free_bytes',30*2**30))
        if shutil.disk_usage(out).free < minimum:
            atomic(Path(state)/'failures'/f'{entry["id"]}.json',dict(reason='Insufficient disk reserve',minimum_bytes=minimum,preserved=True))
            return False
        cfg=read(entry['config_path'])
        try:
            verify_ready(state)
        except Exception as exc:
            raise AdmissionError('Readiness changed before '+entry['id']+': '+repr(exc)) from exc
        atomic(out/'launch_config.json',cfg)
        runtime=Path(entry['runtime_root'])
        atomic(out/'hardware.json',dict(model_family='Qwen3.5',code_commit=subprocess.check_output(
            ['git','-C',str(runtime),'rev-parse','HEAD'],text=True).strip(),
            source_manifest_sha256=sha(runtime/'SOURCE_MANIFEST_q35.json'),
            config_difference=entry.get('config_difference',{}),pilot_phase=pilot,comparison='S05 historical v2; P1/P2 v2.2'))
        command=command_for(entry,pilot,resume)
        record=dict(command=command,resume=resume,checkpoint_step=checkpoint_step,started=time.time(),phase='150' if pilot else '400')
        attempts.append(record);atomic(journal,attempts)
        atomic(Path(state)/'status.json',dict(status='running',experiment=entry['id'],phase=record['phase'],pid=os.getpid()))
        env=dict(os.environ,PYTHONPATH=str(runtime)+':'+str(runtime/'ucf'))
        env.pop('CUDA_VISIBLE_DEVICES',None)
        with (Path(state)/(entry['id']+('_400' if promote else '')+'.log')).open('a') as log:
            child=subprocess.Popen(command,cwd=runtime,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(gpu_lock.fileno(),))
            record.update(pid=child.pid,start_ticks=process_identity(child.pid));atomic(journal,attempts)
            while child.poll() is None:
                if pilot:report_available(plan,state,entry)
                time.sleep(plan.get('poll_seconds',10))
            record.update(returncode=child.returncode,ended=time.time());atomic(journal,attempts)
        if record['returncode']==0:
            try:
                if pilot:
                    atomic(out/'pilot150_verification.json',pilot_verified(out))
                    report_available(plan,state,entry)
                else:verify_full(plan,entry)
                return True
            except Exception as exc:
                record['verification_error']=repr(exc);atomic(journal,attempts)
        # One legitimate checkpoint recovery; no new SFT restart or deletion.
    atomic(Path(state)/'failures'/f'{entry["id"]}.json',dict(reason='Two attempts or verification failure',attempts=attempts,preserved=True))
    return False


def guarded_run(plan, state, entry, gpu_lock, pilot, promote=False):
    """Keep failed evidence and permit the other independent pilot to run."""
    try:
        result=run_entry(plan,state,entry,gpu_lock,pilot,promote)
        if not result:
            terminal_failure_marker(entry,state,promote)
        return result
    except AdmissionError:
        raise
    except Exception as exc:
        # A live orphan still owns the inherited GPU lock; do not proceed to a
        # second training child even if this controller was restarted manually.
        journal=Path(state)/'attempts'/(entry['id']+('_400' if promote else '')+'.json')
        attempts=read(journal) if journal.exists() else []
        if attempts and same_process(attempts[-1]):
            raise
        atomic(Path(state)/'failures'/(entry['id']+('_400' if promote else '')+'.json'),
               dict(reason=repr(exc),preserved=True,unix=time.time()))
        terminal_failure_marker(entry,state,promote)
        return False


def terminal_failure_marker(entry,state,promote=False):
    path=Path(entry['output'])/'needs_manual.json'
    if not path.exists():
        atomic(path,dict(status='failed_preserved',id=entry['id'],phase='400' if promote else '150_or_original',
                        reason='Pilot handoff controller exhausted recovery or could not validate; do not restart from SFT',
                        evidence=str(Path(state)/'failures'),queue_may_continue_independent_jobs=True))


def decide(plan, state, results):
    from pilot_p1p2_report import choose_upgrade
    reports=[]
    for ident in ('P1','P2'):
        path=Path(state)/'reports'/ident/'step_0150.json'
        # Both arms have been attempted to a terminal state. A failed arm cannot
        # qualify; the other completed, verified arm can still qualify on its own.
        if not results.get(ident):
            reports.append(dict(decision_150=dict(status='fail',eligible_for_400=False,reason='pilot_incomplete_or_failed')))
        elif not path.exists():
            reports.append(dict(decision_150=dict(status='pending',eligible_for_400=False,reason='report_missing')))
        else:reports.append(read(path))
    chosen=choose_upgrade(*reports)
    path=Path(state)/'decision.json'
    if path.exists():
        assert read(path)==chosen, 'Frozen pilot decision changed; inspect before promotion'
    else:atomic(path,chosen)
    return chosen


def paired_final_report(plan,state,entry):
    """Final-test pairing is permitted only after the already-selected arm finishes 400."""
    state=Path(state);target=state/'S05b_vs_S05_paired.json'
    if target.exists():
        return
    original=Path(plan['original_root'])
    right=Path(plan.get('runs_root','/data/runs'))/'hard_queue/S05_ucf_seed42'
    command=[plan['python'],str(original/'scripts/paired_report.py'),'--left',entry['output'],
             '--right',str(right),'--output',str(target)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             PYTHONPATH=str(original)+':'+str(original/'scripts'))
    try:
        subprocess.run(command,cwd=original,env=env,check=True)
        assert target.exists()
    except Exception as exc:
        atomic(state/'paired_report_error.json',dict(error=repr(exc),command=command,
               selection_used_test=False,queue_continues=True))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--state',type=Path,required=True)
    args=parser.parse_args();state=args.state;state.mkdir(parents=True,exist_ok=True)
    own=(state/'controller.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=verify_ready(state)
    atomic(state/'status.json',dict(status='waiting_for_S06_boundary',pid=os.getpid(),old_supervisor=plan['old_supervisor']))
    while same_process(plan['old_supervisor']):time.sleep(plan.get('poll_seconds',10))
    verify_s06(plan)
    handles=[]
    for path in [state.parent/'supervisor.lock',Path(plan.get('runs_root','/data/runs'))/'gpu80.lock']:
        handle=path.open('a');fcntl.flock(handle,fcntl.LOCK_EX);handles.append(handle)
    # Close the time-of-check gap after waiting on the active GPU owner.
    plan=verify_ready(state);verify_s06(plan)
    atomic(state/'handoff_received.json',dict(passed=True,unix=time.time(),S06_verified=True,training_signalled=False))
    entries={e['id']:e for e in plan['entries']};results={}
    for ident in ('P1','P2'):
        results[ident]=guarded_run(plan,state,entries[ident],handles[-1],pilot=True)
        atomic(state/'entry_results.json',results)
    choice=decide(plan,state,results)
    for ident in ('S07','S10'):
        results[ident]=guarded_run(plan,state,entries[ident],handles[-1],pilot=False)
        atomic(state/'entry_results.json',results)
    if choice.get('chosen_arm') and all(results.get(k) for k in ('S07','S10')):
        ident=choice['chosen_arm'];entry=entries[ident]
        atomic(state/'S05b_promotion.json',dict(experiment='S05b',selected_arm=ident,output=entry['output'],
            resume_from_step=150,total_steps=400,after=['S07','S10'],decision_sha256=sha(state/'decision.json')))
        try:
            storage_ready=relocate_paused_run(plan,state,entry)
        except Exception as exc:
            storage_ready=False
            atomic(state/'promotion_pending.json',dict(reason=repr(exc),evidence_retained=True))
        results['S05b']=guarded_run(plan,state,entry,handles[-1],pilot=False,promote=True) if storage_ready else False
        atomic(state/'entry_results.json',results)
        if results['S05b']:
            paired_final_report(plan,state,entry)
    archive_control_fence(plan,state)
    atomic(state/'status.json',dict(status='returning_to_original_queue',results=results,decision=choice))
    for handle in reversed(handles):fcntl.flock(handle,fcntl.LOCK_UN);handle.close()
    # exec releases this controller process without leaving a second launcher.
    own.close()
    os.execv(plan['python'],[plan['python'],str(Path(plan['source_root'])/'scripts/pilot_return_queue.py'),'--state',str(state)])


if __name__=='__main__':
    main()
