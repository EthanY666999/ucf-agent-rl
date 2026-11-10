"""Revised serial queue. Live admission/F02 is never signalled or restarted."""
import fcntl,json,os,shutil,subprocess,time
from pathlib import Path
from q35_p0_plan import ROOT,RUNS,PYTHON
from hard_queue_plan import compile_plan
from q35_p0_supervisor import session,status,relocate_verified_model
from q35_run_receipt import read,atomic,sha,verify,summarize,rows
STATE=RUNS/'queue_q35_hard'

def event(**kw):
    with (STATE/'events.jsonl').open('a') as f:f.write(json.dumps(dict(unix=time.time(),**kw))+'\n')

def execute(command,where,success,lock):
    if status(where/'status.json')==success:return True
    journal=STATE/'attempts'/(str(where.relative_to(RUNS)).replace('/','__')+'.json');journal.parent.mkdir(exist_ok=True)
    attempts=read(journal) if journal.exists() else []
    while len(attempts)<2:
        resume=bool(attempts) and (where/'checkpoint.pt').exists()
        if attempts and not resume and where.exists():
            saved=where.with_name(where.name+'_attempt1_'+str(int(time.time())));where.rename(saved)
            event(kind='preserved_failed_attempt',path=str(saved))
        if str(ROOT/'sft.py') in command:where.parent.mkdir(parents=True,exist_ok=True)
        else:where.mkdir(parents=True,exist_ok=True)
        rec=dict(command=command,resume=resume,started=time.time());attempts.append(rec);atomic(journal,attempts)
        with (STATE/(journal.stem+'.log')).open('a') as log:
            proc=subprocess.Popen(command+(['--resume'] if resume else []),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),))
            rec['pid']=proc.pid;atomic(journal,attempts);rec['returncode']=proc.wait();rec['ended']=time.time();atomic(journal,attempts)
        if rec['returncode']==0 and status(where/'status.json')==success:return True
        event(kind='attempt_failed',run=str(where),attempt=len(attempts))
    return False

def eval_receipt(out,cfg):
    assert status(out/'status.json')=='evaluation_complete'
    scores={'main100':summarize(rows(out/'final_test.jsonl'),100)}
    sup=[r for i in range(1,6) for r in rows(out/f'supplemental500/batch{i}.jsonl')];scores['supplemental500']=summarize(sup,500)
    from hard_metrics import completed_metrics
    completed_metrics(out,cfg)
    atomic(out/'verification.json',dict(passed=True,evaluation_only=True,evaluation=scores,record_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.suffix in ['.json','.jsonl'] and p.name!='verification.json'}))

def ready_cpu(env):
    if env=='frozenlake':return True
    p=RUNS/f'datasets/{env}/cpu_acceptance.json';tests=ROOT/'artifacts/hard_cpu_acceptance.json'
    return p.exists() and read(p).get('passed') and tests.exists() and read(tests).get('passed')

def gpu_gate(env,cfg,lock):
    dest=RUNS/f'stage0/{env}_acceptance';dest.mkdir(exist_ok=True)
    if env=='flh':
        revised=dest/'acceptance_authorized_20261108.json'
        if revised.exists():
            from q35_f03_continuation import verify_authorized_base_gate
            return verify_authorized_base_gate(revised)
    if (dest/'acceptance.json').exists():return read(dest/'acceptance.json').get('passed',False)
    # All smoke probes use the base model and validation-only data, never final test.
    acceptance_issues=[]
    for method in ['grpo','tree_random','ucf']:
        trial=dict(cfg,model='/data/models/Qwen3.5-2B',branch_interval=2,ucf_beta=.25,ucf_variant='ucf',ucf_controller='v2',ucf_c0=1.)
        conf=dest/(method+'_launch.json');atomic(conf,trial);out=dest/method
        cmd=[PYTHON,'-u',str(ROOT/'train.py'),'--method',method,'--config',str(conf),'--output-dir',str(out),'--smoke']
        if status(out/'status.json')!='smoke_complete' and shutil.disk_usage(dest).free<35*2**30:
            acceptance_issues.append(dict(method=method,reason='Insufficient disk for smoke checkpoint and final model; preserve evidence'))
            continue
        if not execute(cmd,out,'smoke_complete',lock):
            atomic(dest/'acceptance.json',dict(passed=False,status='待人工',method=method));return False
        # Smoke weights never initialize an experiment. Verify evidence before checkpoint cleanup.
        metrics=[json.loads(x) for x in (out/'train_metrics.jsonl').read_text().splitlines()]
        assert len(metrics)==1 and metrics[0]['max_parameter_change']>=0
        import math
        assert all(math.isfinite(metrics[0][k]) for k in ['loss','kl','grad_norm'])
        assert read(out/'status.json')['status']=='smoke_complete'
        import gzip
        with gzip.open(out/'rollouts_0001.jsonl.gz','rt') as f:traces=[json.loads(x) for x in f]
        think=sum(token in [248068,248069] for r in traces for token,mask in zip(r['ids'],r['mask']) if mask)
        if think:
            acceptance_issues.append(dict(method=method,reason='Generated thinking markers violate the unchanged acceptance protocol',count=think))
        atomic(out/'smoke_checks.json',dict(passed=think==0,finite=True,thinking_markers=think,nonzero_parameter_update=metrics[0]['max_parameter_change']>0,zero_update_note='all-equal reward groups may have zero advantages; existing Stage0 checks cover nonzero update correctness'))
        if think:
            # Retain failed acceptance evidence and weights; continue independent smoke methods.
            continue
        model=out/'final_model'
        if model.exists() and not model.is_symlink():
            atomic(out/'verification.json',dict(passed=True,smoke_only=True,final_model_sha256={p.name:sha(p) for p in model.glob('*.safetensors')}))
            target=Path('/home/vipuser/q35-smoke-models')/env/method
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copytree(model,target)
            assert all(sha(target/p.name)==sha(p) for p in model.iterdir() if p.is_file())
            shutil.rmtree(model);model.symlink_to(target,target_is_directory=True)
        cp=out/'checkpoint.pt'
        if cp.exists():
            atomic(out/'checkpoint_pruning.json',dict(reason='One-step smoke complete, final validation and finite update verified; final model retained',bytes=cp.stat().st_size))
            cp.unlink()
    atomic(dest/'acceptance.json',dict(passed=not acceptance_issues,status='待人工' if acceptance_issues else 'passed',issues=acceptance_issues,methods=['grpo','tree_random','ucf'],cpu_receipt_sha256=sha(RUNS/f'datasets/{env}/cpu_acceptance.json')))
    return not acceptance_issues

def manual(entry,reason):
    out=Path(entry['output']);out.mkdir(parents=True,exist_ok=True)
    atomic(out/'needs_manual.json',dict(status='待人工',reason=reason));event(kind='needs_manual',id=entry['id'],reason=reason)

def predecessor_ready(entry):
    previous=entry.get('strict_predecessor')
    if not previous:return True
    receipt=Path(previous)/'verification.json'
    return receipt.exists() and read(receipt).get('passed') is True

def main():
    from q35_queue_pause import honor_user_pause
    if honor_user_pause('hard_queue_supervisor'):
        return
    STATE.mkdir(exist_ok=True);(STATE/'configs').mkdir(exist_ok=True)
    own=(STATE/'supervisor.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic(STATE/'status.json',dict(status='waiting_for_F02_400_and_final_evaluation',pid=os.getpid()))
    while session('q35_admission'):time.sleep(10)
    f02=RUNS/'stage2/F02_noSFT_GRPO4_seed42'
    # The live run must finish, including final evaluation. No bypass on a mere step400 checkpoint.
    assert status(f02/'status.json')=='complete','F02 incomplete: preserve it, do not advance'
    lock=(RUNS/'gpu80.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    verify(f02,prune_checkpoint=True);relocate_verified_model(f02)
    from hard_metrics import completed_metrics,backfill_flat
    backfill_flat(f02);completed_metrics(f02,read(f02/'config.json'))
    plan=compile_plan(read(ROOT/'configs/frozenlake_q35_2b.json'),read(RUNS/'stage0/cost/result.json'));atomic(STATE/'plan.json',plan)
    pending=list(plan['entries']);deferred=[]
    while pending or deferred:
        # Only Sokoban may be skipped for acceptance readiness; resume at the next boundary.
        if deferred and ready_cpu('sokoban'):
            pending=deferred+pending;deferred=[]
        if not pending:
            atomic(STATE/'status.json',dict(status='waiting_sokoban_acceptance',remaining=[e['id'] for e in deferred]));time.sleep(60);continue
        e=pending.pop(0);out=Path(e['output']);env=e['environment']
        if (out/'verification.json').exists() and read(out/'verification.json')['passed']:continue
        predecessor=e.get('strict_predecessor')
        if not predecessor_ready(e):
            atomic(STATE/'status.json',dict(status='needs_manual_strict_order',experiment=e['id'],blocked_by=predecessor,reason='Authorized serial prefix cannot be skipped'))
            return
        if (out/'needs_manual.json').exists():
            if predecessor:
                atomic(STATE/'status.json',dict(status='needs_manual_strict_order',experiment=e['id'],reason='Preserve failed run; do not skip authorized serial prefix'))
                return
            continue
        if not ready_cpu(env):
            if env=='sokoban':deferred.append(e);continue
            pending.insert(0,e);atomic(STATE/'status.json',dict(status='waiting_environment_acceptance',environment=env));time.sleep(60);continue
        cfg=e['config']
        if cfg.get('ucf_corrections')=='v22_question_loo_rejection256':
            from q35_v21_gate import corrected_cpu_ready
            if not corrected_cpu_ready():
                manual(e,'Corrected UCF CPU gates not passed; preserve original S05')
                return
        if cfg.get('fork_point')=='first_fence':
            from q35_v21_gate import gpu_admission
            if not gpu_admission(e,lock):
                manual(e,'UCF-v2.2-ff offline/CPU/GPU acceptance not passed; no launch, no bypass of serial prefix')
                atomic(STATE/'status.json',dict(status='needs_manual_v21_gate',experiment=e['id']))
                return
        if cfg.get('dataset_sha256')=='PENDING_DATASET':cfg['dataset_sha256']=sha(Path(cfg['dataset_path']))
        if cfg.get('frozen_maps_sha256')=='PENDING_MAPS':cfg['frozen_maps_sha256']=sha(Path(cfg['frozen_maps_path']))
        # Do not launch a dependent job without its own environment's verified SFT.
        sft_root=RUNS/'hard_queue/S02_SFT_seed42' if env=='sokoban' else RUNS/'stage2/F03_SFT_seed42'
        if any(x in e['requires'] for x in ['F03_sft_and_evaluation','S02_sft']) and not (sft_root/'verification.json').exists():
            manual(e,'Required SFT and evaluation not verified');continue
        if env!='frozenlake' and not gpu_gate(env,cfg,lock):
            manual(e,'Environment GPU acceptance not passed; see stage0 acceptance evidence');continue
        if e['environment']=='flh' and e['method'] in ['grpo','tree_random','ucf']:
            from q35_f03_continuation import f03_rl_smoke_gate
            if not f03_rl_smoke_gate(e,lock,execute):
                manual(e,'F03 RL loading/prompt/generated-marker smoke requires manual review');continue
        if shutil.disk_usage(RUNS).free<55*2**30 and e['method']!='eval' and e['id']!='F03':
            manual(e,'Insufficient room after verified completed checkpoint cleanup');continue
        out.mkdir(parents=True,exist_ok=True)
        conf=STATE/'configs'/(e['id']+'.json')
        if conf.exists():assert read(conf)==cfg
        else:atomic(conf,cfg)
        atomic(out/'launch_config.json',cfg)
        atomic(out/'hardware.json',dict(gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),model_family='Qwen3.5',config_difference=e['config_difference'],source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST_q35.json')))
        atomic(STATE/'status.json',dict(status='running',experiment=e['id'],pid=os.getpid(),deferred=[x['id'] for x in deferred]))
        def cmd(key):return [str(conf) if x==e['config_path'] else x for x in e[key]]
        if e['id']=='F03':
            from q35_f03_continuation import evaluate_f03_once
            ok=evaluate_f03_once(cmd('evaluation_command'),out,lock)
        elif e['method']=='sft':
            ok=execute(cmd('training_command'),out/'sft','complete',lock)
            if ok:ok=execute(cmd('evaluation_command'),out/'evaluation','evaluation_complete',lock)
        else:ok=execute(cmd('command'),out,'evaluation_complete' if e['method']=='eval' else 'complete',lock)
        if not ok:manual(e,'Authorized extra compatibility evaluation failed; no SFT retrain and no recovery quota consumed' if e['id']=='F03' else 'Initial attempt and one retry/recovery failed; evidence retained');continue
        try:
            if e['method']=='eval':eval_receipt(out,cfg)
            else:
                verify(out,sft=e['method']=='sft',prune_checkpoint=e['id']!='F03');relocate_verified_model(out,sft=e['method']=='sft')
                if e['id']=='F03':
                    from q35_f03_continuation import prune_verified_checkpoint
                    prune_verified_checkpoint(out/'sft',out/'verification.json')
        except Exception as exc:manual(e,'Final record validation: '+repr(exc));continue
        event(kind='verified',id=e['id'])
        if e['id'] in ['S13','S15']:
            from q35_s13_s15_compare import write_report
            write_report()
    atomic(STATE/'status.json',dict(status='revised_primary_queue_complete',all_experiments_complete=False,remaining=plan['remaining_legacy_after_this_plan']))
if __name__=='__main__':main()
