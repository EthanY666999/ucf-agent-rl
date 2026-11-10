"""User revision 2026-11-07: exact serial order, isolated from live F02."""
import copy,hashlib,json
from pathlib import Path
from q35_p0_plan import compile_plan as legacy_plan,ROOT,RUNS,PYTHON,SFT
DATA=RUNS/'datasets'
SOKO_SFT=RUNS/'hard_queue/S02_SFT_seed42/sft/final_model'
# An acceptance smoke is part of each environment's entry, never another training experiment.
ORDER=[('F03','frozenlake','sft',True,42),
 ('H01','flh','eval',False,42),('H02','flh','eval',True,42),
 ('H03','flh','grpo',True,42),('H04','flh','tree_random',True,42),('H05','flh','ucf',True,42),
 ('S01','sokoban','eval',False,42),('S02','sokoban','sft',True,42),('S03','sokoban','grpo',True,42),('S04','sokoban','tree_random',True,42),('S05','sokoban','ucf',True,42),
 ('F04','frozenlake','grpo',True,42),('F05','frozenlake','tree_random',True,42),('F06','frozenlake','ucf',True,42),
 ('F10','frozenlake','ucf',False,42),('F09','frozenlake','tree_random',False,42),
 ('H06','flh','starpo_s',True,42),('H07','flh','gigpo',True,42),('S06','sokoban','gigpo',True,42),('S07','sokoban','starpo_s',True,42),
 ('H08','flh','grpo',True,43),('H09','flh','tree_random',True,43),('H10','flh','ucf',True,43),('H11','flh','starpo_s',True,43),('H12','flh','gigpo',True,43),
 ('S08','sokoban','grpo',True,43),('S09','sokoban','tree_random',True,43),('S10','sokoban','ucf',True,43),
 ('H13','flh','ucf_noIS',True,42),('H14','flh','ucf_uniform',True,42),('H15','flh','ucf_tau',True,42),('H16','flh','ucf_v1',True,42),
 ('H17','flh','ucf_noIS',True,43),('H18','flh','ucf_uniform',True,43),('H19','flh','ucf_tau',True,43),
 ('F07','frozenlake','starpo_s',True,42),('F08','frozenlake','gigpo',True,42),
 ('F11','frozenlake','ucf',True,43),('F12','frozenlake','tree_random',True,43),('F13','frozenlake','grpo',True,43),('F14','frozenlake','starpo_s',True,43),('F15','frozenlake','gigpo',True,43),
 ('F16','frozenlake','ucf',True,44),('F17','frozenlake','tree_random',True,44),('F18','frozenlake','grpo',True,44),('F19','frozenlake','starpo_s',True,44),('F20','frozenlake','gigpo',True,44),
 ('F28','frozenlake','patr',True,42),('F29','frozenlake','grpo8',True,42),('F30','frozenlake','patr',False,42),('F31','frozenlake','grpo8',False,42),('F32','frozenlake','starpo_s',False,42),('F33','frozenlake','gigpo',False,42),('F34','frozenlake','ucf',False,43),('F35','frozenlake','tree_random',False,43),('F36','frozenlake','grpo',False,43)]

# User 2026-11-08: finish the unchanged Sokoban trio, then F03 and FLH.
ORDER=[x for x in ORDER if x[0] in ['S01','S02','S03','S04','S05']]+[x for x in ORDER if x[0] not in ['S01','S02','S03','S04','S05']]

# User 2026-11-09 revision: FLH precedes the new no-SFT pair.
_insert=next(i for i,x in enumerate(ORDER) if x[0]=='H05')+1
ORDER[_insert:_insert]=[('S13','sokoban','grpo',False,42),('S15','sokoban','ucf',False,42)]
# User 2026-11-09: after the already-running H02 evaluation, prioritize Sokoban GiGPO.
_s06=next(x for x in ORDER if x[0]=='S06')
ORDER=[x for x in ORDER if x[0]!='S06']
ORDER.insert(next(i for i,x in enumerate(ORDER) if x[0]=='H03'),_s06)
STRICT_PREFIX=['F03','H01','H02','S06','H03','H04','H05','S13','S15']

def compile_plan(base,cost):
    old={e['id']:e for e in legacy_plan(base,cost)['entries']};entries=[]
    for ident,env,method,sft,seed in ORDER:
        if env=='frozenlake':
            if ident in old:
                e=old[ident];e['environment']=env;entries.append(e);continue
        cfg=copy.deepcopy(base);model=str(SOKO_SFT if env=='sokoban' else SFT) if sft else base['model']
        data=DATA/env/'data.json';out=RUNS/'hard_queue'/f'{ident}_{method}_seed{seed}'
        if ident=='S02':out=SOKO_SFT.parent.parent
        cfg.update(model=model,model_family='Qwen3.5',environment=env,seed=seed,steps=400,eval_every=25,flat_group_size=4,max_turns=20 if env=='flh' else 10,max_response_tokens=10240 if sft else cost['nosft_max_response_tokens'],evaluate_extra_sets=True,eval_sets_dir=str(DATA/env),dataset_path=str(data),dataset_sha256=hashlib.sha256(data.read_bytes()).hexdigest() if data.exists() else 'PENDING_DATASET',run_label=out.name)
        maps=DATA/env/'maps.json'
        cfg.update(frozen_maps_path=str(maps),frozen_maps_sha256=hashlib.sha256(maps.read_bytes()).hexdigest() if maps.exists() else 'PENDING_MAPS')
        if env=='frozenlake':
            for k in ['dataset_path','dataset_sha256','frozen_maps_path','frozen_maps_sha256']:cfg.pop(k,None)
            cfg.update(max_turns=base['max_turns'],eval_sets_dir='/home/vipuser/decision-point-q35-hard/code/eval_sets')
        if method in ['tree_random','patr']:cfg['branch_interval']=2
        if method=='grpo8':cfg['flat_group_size']=8
        if method=='starpo_s':cfg.update(starpo_keep_fraction=.5,clip_low=.2,clip_high=.28,kl_coefficient=0.)
        if method=='gigpo':cfg.update(gigpo_gamma=.95,gigpo_omega=1.)
        variant='ucf'
        if method.startswith('ucf'):
            if method in ['ucf_noIS','ucf_uniform']:variant=method.split('_')[1]
            cfg.update(ucf_beta=.25,ucf_variant=variant,ucf_c0=1.,ucf_controller='v1' if method=='ucf_v1' else 'v2')
            if method=='ucf_tau':cfg['ht_weight_cap']=10
        config=out/'launch_config.json';e=dict(id=ident,environment=env,priority='revised_user_order',method='ucf' if method.startswith('ucf') else ('grpo' if method=='grpo8' else method),seed=seed,output=str(out),config_path=str(config),config=cfg,requires=['stage0_acceptance',env+'_acceptance']+(['S02_sft' if env=='sokoban' else 'F03_sft_and_evaluation'] if sft and method!='sft' else []),config_difference={k:dict(base=base.get(k),value=v) for k,v in cfg.items() if base.get(k)!=v})
        if method=='sft':
            e.update(training_command=[PYTHON,'-u',str(ROOT/'sft.py'),'--model',base['model'],'--data',str(DATA/env/'sft_messages.jsonl'),'--output',str(out/'sft'),'--optimizer','adamw'],evaluation_command=[PYTHON,'-u',str(ROOT/'train.py'),'--method','grpo','--config',str(config),'--output-dir',str(out/'evaluation'),'--eval-only'])
        else:
            command=[PYTHON,'-u',str(ROOT/'ucf/run_ucf.py' if method.startswith('ucf') else ROOT/'train.py')]
            command+=['--variant',variant] if method.startswith('ucf') else ['--method','grpo' if method in ['eval','grpo8'] else method]
            command+=['--config',str(config),'--output-dir',str(out)]
            if method=='eval':command+=['--eval-only']
            e['command']=command
        if ident in ['S13','S15']:
            assert cfg['model']=='/data/models/Qwen3.5-2B'
            cfg['max_response_tokens']=4096
            e['config_difference']['max_response_tokens']=dict(base=base.get('max_response_tokens'),value=4096)
        entries.append(e)
    by_id={e['id']:e for e in entries}
    # Completed F02/S01 are historical old-protocol results; never rerun them.
    for ident,env,method,sft,seed in ORDER:
        if not sft and ident != 'S01':
            e=by_id[ident]
            e['config']['response_protocol']='first_complete_action_block'
            e['config_difference']['response_protocol']=dict(base=None,value='first_complete_action_block')
            e['protocol_caveat']='New noSFT first-complete-action-block protocol; F02 is historical, not a paired control'
    # Explicitly approved future UCF correction; the active S05 stays original.
    for ident,e in by_id.items():
        if e['method']=='ucf' and ident!='S05':
            e['config']['ucf_corrections']='v22_question_loo_rejection256'
            e['config_difference']['ucf_corrections']=dict(base=None,value='v22_question_loo_rejection256')
            e['implementation_version']='UCF-v2.2; opening-fence conditional node; S05 historical v2 unchanged'
    for ident in ['S15','F10','F34']:
        e=by_id[ident]
        e['config']['fork_point']='first_fence'
        e['command']+=['--fork_point','first_fence']
        e['config_difference']['fork_point']=dict(base=None,value='first_fence')
        e['implementation_version']='UCF-v2.2-ff'
        e['comparison_caveat']='noSFT UCF-v2.2-ff versus historical S05 changes BOTH initialization and implementation, plus protocol. Versus future SFT v2.2, initialization, response protocol and permitted pre-fence reasoning differ; not a strict remove-SFT-only ablation'
        e['requires']+=['v21_offline_S13_new_protocol','v21_one_step_GPU']
    for previous,current in zip(['S05']+STRICT_PREFIX,STRICT_PREFIX):
        by_id[current]['strict_predecessor']=by_id[previous]['output']
    return dict(revision='user_20261109_S06_GiGPO_next_after_active_H02',entries=entries,existing_admission=['F01','F02'],f02_must_finish_400_and_final_eval=True,sokoban_unready='defer only unaccepted Sokoban; retry earliest pending Sokoban at next experiment boundary after acceptance',remaining_legacy_after_this_plan=['F37 optional official RAGEN reference','F38 optional gradient diagnostic','S11','S12','S14 noSFT-Random K=2: original later P2 position, not promoted','S16','S17'],legacy_ablations_replaced_by=['H13','H14','H15','H16'],test_used_for_selection=False)
