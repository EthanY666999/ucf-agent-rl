"""Build a hash-bound paired continuation plan after the GPU admission and fork."""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path('/home/vipuser/decision-point-ucf-v3-capture-candidate-20261110')
STATE=Path('/data/runs/ucf_v3_budget10240_20261110')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def hash_existing_sources(files):
    missing=[str(p) for p in files if not p.is_file()]
    assert not missing,'Required runtime sources missing: '+repr(missing)
    return {str(p):sha(p) for p in files}
def main():
    proof=read(STATE/'admission/acceptance.json');assert proof['passed']
    assert proof['original_sampling_bitwise_unchanged'] and proof['max_original_capture_logprob_delta']<=1e-4
    assert proof['update']['max_parameter_change']>0
    assert read(STATE/'continuation_cpu_tests.json')['returncode']==0
    fork=read(STATE/'checkpoint_forks.json');assert fork['tensor_optimizer_rng_equal']
    outputs={x['method']:x for x in fork['outputs']}
    assert len({x['tensor_optimizer_rng_sha256'] for x in fork['outputs']})==1
    files=list((ROOT/'code').rglob('*.py'))+list((ROOT/'code').rglob('*.json'))
    files += [ROOT/'storage/train_scoped.py',ROOT/'storage/scoped_inference.py',
              ROOT/'control/v3_manager.py',ROOT/'control/review_pilot.py']
    files=[p for p in files if '__pycache__' not in p.parts and 'artifacts' not in p.parts]
    hashes=hash_existing_sources(files)
    for path,h in read(STATE/'admission_sources.json').items():
        assert sha(path)==h,'An admitted source changed after the GPU check'
    entries=[]
    for step in (25,50,100,150,200):
        for method in ('ucf','grpo'):
            o=outputs[method];out=Path(o['path']).parent
            cfg=read(o['config_path'])
            assert cfg['max_response_tokens']==10240 and cfg['ucf_probability_source']=='original_sampling_logits'
            admitted=read(STATE/'admission/config.json')
            excluded={'train_batch_size','mini_batch_tasks','ucf_log_dir','run_label'}
            assert {k:v for k,v in cfg.items() if k not in excluded}=={k:v for k,v in admitted.items() if k not in excluded}
            entries.append(dict(id=out.name,kind='train',method=method,stop_step=step,
                output=str(out),config=o['config_path'],config_sha256=o['config_sha256'],
                requires_gpu_acceptance=str(STATE/'admission/acceptance.json'),
                requires_scoped_gpu_acceptance=str(STATE/'admission')))
    plan=dict(revision='V3_budget10240_shared_RL_step16_paired_exploration',
        python='/data/venvs/qwen35/bin/python',runtime=str(ROOT/'code'),state=str(STATE/'training'),
        manager_script=str(ROOT/'control/v3_manager.py'),train_entrypoint=str(ROOT/'storage/train_scoped.py'),
        scoped_helper=str(ROOT/'storage/scoped_inference.py'),source_sha256=hashes,
        locks=['/data/runs/queue_q35_hard/supervisor.lock','/data/runs/gpu80.lock'],entries=entries,
        initialization=dict(step=16,shared_rl_warmstart=True,task_sft=False,
            source_checkpoint_sha256=fork['source']['sha256'],not_fresh_raw_start=True),
        first_boundary_new_updates=9,no_automatic_400_step_promotion=True,final_test_used=False,
        intended_review='Review fixed validation and original-four truncation after each paired boundary. Later boundaries may be revised only after a saved optimizer boundary.',
        unix=time.time())
    path=STATE/'plan.json';path.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(path=str(path),sha256=sha(path),entries=len(entries))))

if __name__=='__main__':main()
