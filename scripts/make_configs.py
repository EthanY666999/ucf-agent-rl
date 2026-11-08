"""Update the active 4B plan; preserve existing 9B configs and all device settings."""
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def main():
    base=json.loads((ROOT/'configs/protocol_2b.json').read_text())
    previous=json.loads((ROOT/'configs/plan.json').read_text())
    deferred={item['id']:item for section in ['main','conditional','deferred_unchanged']
              for item in previous.get(section,[]) if item['id'].startswith('9b_')}
    plan=[]
    for scale in ['4b']:
        model=f'models/Qwen3.5-{scale.upper()}'
        envs=['sokoban','flh']
        for env in envs:
            methods=['base_eval','sft','grpo4','tree_random_k2','ucf_v2.2']
            for label in methods:
                plan.append(make(base,scale,env,label,42,model))
        if scale=='4b':
            for env in envs:
                for label in ['starpo_s','gigpo']:plan.append(make(base,scale,env,label,42,model))
        for env in envs:
            labels=['grpo4','tree_random_k2','ucf_v2.2']
            for label in labels:plan.append(make(base,scale,env,label,43,model))
    # Conditional no-SFT experiments are outside the main ordered list.
    extras=[make(base,s,e,'nosft_ucf_v2.2-ff',42,f'models/Qwen3.5-{s.upper()}')
            for s in ['4b'] for e in ['sokoban','flh']]
    for env in envs:
        for name in ['ucf_v2_seed42.json','ucf_v2_seed43.json','nosft_ucf_v2.1_seed42.json']:
            old=ROOT/'configs/4b'/env/name
            if old.exists():old.unlink()
    write(ROOT/'configs/plan.json',dict(schema_version=1,execution='manual serial; no automatic queue',
        active_scope='4B only',main=plan,conditional=extras,
        deferred_unchanged=list(deferred.values()),
        deferred_note='9B arrangements are for the collaborator to decide after 4B; retained entries are not an active queue',
        webshop='future work'))


def make(base,scale,env,label,seed,model):
    c=copy.deepcopy(base)
    method={'grpo4':'grpo','tree_random_k2':'tree_random','base_eval':'grpo',
            'ucf_v2.2':'ucf','nosft_ucf_v2.2-ff':'ucf'}.get(label,label)
    stage='evaluate' if label=='base_eval' else ('sft' if label=='sft' else 'train')
    start='base' if label in ['base_eval','sft','nosft_ucf_v2.2-ff'] else 'sft'
    ident=f'{scale}_{env}_{label}_seed{seed}'
    sft_output=f'runs/{scale}/{env}/sft_seed42'
    c.update(model=model if start=='base' else sft_output+'/final_model',bootstrap_model=model,
        model_family='Qwen3.5',environment=env,seed=seed,run_label=ident,eval_every=25,
        flat_group_size=4,micro_batch_size=1,micro_batch_token_limit=32768,
        vllm_memory_fraction=.12 if scale=='4b' else .15,vllm_max_sequences=8,
        max_response_tokens=4096 if start=='base' and stage!='sft' else 10240,
        max_turns={'sokoban':10,'flh':20,'alfworld':50}[env],
        optimizer='adamw')
    for key in ['ucf_beta','ucf_variant','ucf_controller','ucf_c0']:
        c.pop(key,None)
    if env!='alfworld':
        c.update(dataset_path=f'data/{env}/data.json',dataset_sha256=sha(ROOT/f'data/{env}/data.json'),
            frozen_maps_path=f'data/{env}/maps.json',frozen_maps_sha256=sha(ROOT/f'data/{env}/maps.json'),
            eval_sets_dir=f'data/{env}',evaluate_extra_sets=True)
    else:
        c.update(history_length=2,action_format='first triple-backtick block',
                 evaluation_splits=['valid_seen','valid_unseen'])
    if method=='tree_random':c['branch_interval']=2
    if method=='gigpo':c.update(gigpo_gamma=.95,gigpo_omega=1.)
    if method=='starpo_s':c.update(starpo_keep_fraction=.5,clip_low=.2,clip_high=.28,kl_coefficient=0.)
    if method=='ucf':c.update(ucf_beta=.25,ucf_variant='ucf',ucf_c0=1.,
        ucf_controller='leave_one_task_out',
        ucf_version='v2.2-ff' if 'nosft' in label else 'v2.2')
    if 'nosft' in label:
        c['turn_stop_protocol']='first_complete_action_block_token_boundary'
        c['ucf_fork_point']='immediately_after_first_opening_fence'
    gates=[]
    if scale=='9b':gates.append('distributed_training_backend_required')
    if env=='alfworld':gates.append('alfworld_training_integration_required')
    if method=='ucf':gates.append(c['ucf_version']+'_2b_acceptance_and_source_import_required')
    sft_pending=env=='flh' and (stage=='sft' or start=='sft')
    if sft_pending:gates.append('flh_sft_start_pending_2b_H01_decision')
    c['_release']=dict(id=ident,stage=stage,method=method,scale=scale,start=start,
        output=sft_output if stage=='sft' else f'runs/{scale}/{env}/{label}_seed{seed}',
        sft_data={'sokoban':'data/sokoban/sft_messages.jsonl','flh':None}[env],
        gates=gates,gpu_status='NOT GPU validated; 80GiB single GPU insufficient for RL',
        algorithm_source='protocol_2b.json; only model/runtime/path fields change, plus declared environment and method settings')
    if env=='flh':
        c['_release']['sft_start_status']='pending_2b_H01_decision'
        c['_release']['sft_start_note']='2B currently uses original FrozenLake F03; align 4B data/protocol after H01. No hard Golden Trace default; model/output paths are placeholders.'
    if method=='ucf':
        c['_release']['implementation_status']='planned; no verified v2.2 import yet'
        c['_release']['baseline_and_p_hat']='leave-one-trajectory-out, unchanged'
    path=ROOT/f'configs/{scale}/{env}/{label}_seed{seed}.json'
    write(path,c)
    return dict(id=ident,config=str(path.relative_to(ROOT)),stage=stage,method=method,gates=gates,
        same_sft_start=sft_output if start=='sft' else None)


if __name__=='__main__':main()
