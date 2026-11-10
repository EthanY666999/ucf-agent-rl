"""Compile the numbered P0 queue without launching jobs or choosing parameters.

F01/F02 are owned by admit_f01. Entries here are immutable launch inputs for
the continuation supervisor; Sokoban remains blocked on its own acceptance.
"""
import argparse
import copy
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUNS=Path('/data/runs')
PYTHON='/data/venvs/qwen35/bin/python'
SFT=RUNS/'stage2/F03_SFT_seed42/sft/final_model'
ROWS=[
 ('F03',2,'SFT','sft',True,42),
 ('F04',2,'SFT_GRPO4','grpo',True,42),
 ('F05',3,'SFT_RandomK2','tree_random',True,42),
 ('F06',3,'SFT_UCFv2','ucf',True,42),
 ('F07',3,'SFT_StarPOS','starpo_s',True,42),
 ('F08',3,'SFT_GiGPO','gigpo',True,42),
 ('F09',4,'noSFT_RandomK2','tree_random',False,42),
 ('F10',4,'noSFT_UCFv2','ucf',False,42),
 ('F11',5,'SFT_UCFv2','ucf',True,43),
 ('F12',5,'SFT_RandomK2','tree_random',True,43),
 ('F13',5,'SFT_GRPO4','grpo',True,43),
 ('F14',5,'SFT_StarPOS','starpo_s',True,43),
 ('F15',5,'SFT_GiGPO','gigpo',True,43),
]


def compile_plan(base,cost):
    assert cost['passed'] and cost['steps']==5
    cap=2048 if cost['trajectory_length_p95']<=2048 else 4096
    assert cost['nosft_max_response_tokens']==cap
    entries=[]
    for ident,stage,label,method,sft,seed in ROWS:
        out=RUNS/f'stage{stage}/{ident}_{label}_seed{seed}'
        cfg=copy.deepcopy(base)
        cfg.update(model=str(SFT) if sft else base['model'],model_family='Qwen3.5',
                   seed=seed,steps=400,eval_every=25,flat_group_size=4,
                   max_response_tokens=10240 if sft else cap,
                   evaluate_extra_sets=True,eval_sets_dir='/home/vipuser/decision-point-q35-hard/code/eval_sets',
                   run_label=out.name)
        if method=='tree_random':cfg['branch_interval']=2
        if method=='starpo_s':
            cfg.update(starpo_keep_fraction=.5,clip_low=.2,clip_high=.28,kl_coefficient=0.)
        if method=='gigpo':cfg.update(gigpo_gamma=.95,gigpo_omega=1.)
        if method=='ucf':cfg.update(ucf_beta=.25,ucf_variant='ucf',ucf_c0=1.,ucf_controller='v2')
        entry=dict(id=ident,priority='P0',stage=stage,method=method,seed=seed,
                   output=str(out),requires=['stage0_acceptance']+(['F03_sft_and_evaluation'] if sft and method!='sft' else []))
        config=out/'launch_config.json'
        if method=='sft':
            # SFT is always a fresh copy of the base model, never F02's weights.
            cfg['model']=str(SFT)
            entry.update(training_command=[PYTHON,'-u',str(ROOT/'sft.py'),'--model',base['model'],
                                           '--data',str(ROOT.parent/'golden-trace/sft_messages.jsonl'),
                                           '--output',str(out/'sft'),'--optimizer','adamw'],
                         evaluation_command=[PYTHON,'-u',str(ROOT/'train.py'),'--method','grpo',
                                             '--config',str(config),'--output-dir',str(out/'evaluation'),'--eval-only'],
                         sft_protocol=dict(epochs=1,trajectories=1000,seed=42,batch_trajectories=16,
                                           micro_batch=1,learning_rate=2e-5,optimizer='adamw'))
        else:
            command=[PYTHON,'-u',str(ROOT/'ucf/run_ucf.py' if method=='ucf' else ROOT/'train.py')]
            if method!='ucf':command+=['--method',method]
            entry['command']=command+['--config',str(config),'--output-dir',str(out)]
        entry.update(config_path=str(config),config=cfg,
                     config_difference={k:dict(base=base.get(k),value=v) for k,v in cfg.items() if base.get(k)!=v})
        entries.append(entry)
    return dict(scope='80G Qwen3.5-2B only',launches_processes=False,
                existing_admission=['F01','F02'],entries=entries,
                next_environment=dict(ids=['S01','S02','S03','S04','S05'],priority='P0',
                                      status='pending_stage0.9_implementation_and_acceptance'),
                deferred=['P1','P2'],test_used_for_selection=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cost',type=Path,default=RUNS/'stage0/cost/result.json');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    plan=compile_plan(json.loads((ROOT/'configs/frozenlake_q35_2b.json').read_text()),json.loads(a.cost.read_text()))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
