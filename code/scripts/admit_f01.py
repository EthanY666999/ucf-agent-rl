"""Stage-0 receipt and F01 admission. No implicit downgrade of failed checks."""
import json,subprocess,hashlib,time,fcntl,shutil
from pathlib import Path
ROOT=Path('/data/decision-point/code');STAGE=Path('/data/runs/stage0')
def read(p):return json.loads(p.read_text())
def main():
 while subprocess.run(['tmux','has-session','-t','q35_repair_0431'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:time.sleep(10)
 while subprocess.run(['tmux','has-session','-t','q35_crosscard_resume'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:time.sleep(10)
 assert (STAGE/'final_cpu.exit').read_text().strip()=='0'
 replay=read(STAGE/'controller_replay_final/replay.json');assert all(r['passed'] and r['old_exact_steps']==400 for r in replay.values())
 cost=read(STAGE/'cost/result.json');assert cost['passed'] and cost['steps']==5
 cross=read(STAGE/'crosscard/result.json');assert cross['steps']==10
 from crosscard_receipt import compare
 cross_receipt=compare(cross['pairs'],[json.loads(line) for line in (STAGE/'crosscard/train_metrics.jsonl').read_text().splitlines()])
 (STAGE/'crosscard/comparison_receipt.json').write_text(json.dumps(cross_receipt,indent=2))
 assert cross_receipt['passed'],('crosscard_comparison_failed',cross_receipt)
 checks={}
 for method,directory in [('grpo','grpo'),('tree_random','tree_random_smoke_k2'),('patr','patr_smoke_k2'),('gigpo','gigpo'),('starpo_s','starpo_s'),('ucf','ucf_smoke_capacity')]:
  result=read(STAGE/'gpu_smoke'/directory/'result.json');assert result['passed'],method;checks[method]=result
 assert checks['ucf']['forced_decoder_fixture']['passed']
 gate=ROOT/'artifacts/q35_gates';gate.mkdir(exist_ok=True)
 common=dict(passed=True,model_family='Qwen3.5',generated_unix=time.time())
 (gate/'cpu_acceptance.json').write_text(json.dumps(dict(common,tests_log=str(STAGE/'final_cpu.log'),replay=str(STAGE/'controller_replay_final/replay.json')),indent=2))
 (gate/'gpu_acceptance.json').write_text(json.dumps(dict(common,methods={k:dict(step_seconds=v['step_seconds'],peak_allocated_gib=v['peak_allocated_gib'],think_tokens=v['generated_think_tokens']) for k,v in checks.items()}),indent=2))
 (gate/'calibration.json').write_text(json.dumps(dict(common,mode='per_batch',target=2.,rho_min=.03,infeasible='reachable_support_upper_bound',c0_used_for_training=False),indent=2))
 (gate/'phase1_gate.json').write_text(json.dumps(dict(common,role='protocol_registration_only',beta=.25,phase1_diagnostic_deferred=True,gradient_quality_used_for_selection=False),indent=2))
 admission=dict(common,cost=cost,crosscard_comparison=str(STAGE/'crosscard/comparison_receipt.json'),gpu_gate=str(gate/'gpu_acceptance.json'),cpu_gate=str(gate/'cpu_acceptance.json'))
 (STAGE/'acceptance.json').write_text(json.dumps(admission,indent=2))
 # Keep the stage-0 weights remotely, without consuming the 100GB data volume's
 # space needed for atomic formal checkpoint saves. Logical run paths stay valid.
 cp=STAGE/'cost/checkpoint.pt'
 if cp.exists() and not cp.is_symlink():
  dest=Path('/home/vipuser/q35-stage0-weights/cost_checkpoint.pt');dest.parent.mkdir(exist_ok=True)
  assert not dest.exists()
  shutil.move(str(cp),str(dest));cp.symlink_to(dest)
  (STAGE/'cost/storage_relocation.json').write_text(json.dumps(dict(logical_path=str(cp),physical_path=str(dest),reason='preserve pilot weights while reserving atomic checkpoint capacity on /data'),indent=2))
 out=Path('/data/runs/stage1/F01_base_eval');out.mkdir(parents=True,exist_ok=True)
 cfg=read(ROOT/'configs/frozenlake_q35_2b.json');cfg.update(max_response_tokens=cost['nosft_max_response_tokens'],flat_group_size=4,evaluate_extra_sets=True,eval_sets_dir=str(ROOT/'eval_sets'),model_family='Qwen3.5')
 (out/'launch_config.json').write_text(json.dumps(cfg,indent=2))
 (out/'hardware.json').write_text(json.dumps(dict(gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True),model_family='Qwen3.5',code_manifest_sha256=hashlib.sha256((ROOT/'SOURCE_MANIFEST_q35.json').read_bytes()).hexdigest(),environment_freeze='/data/setup_logs/qwen35_freeze_v2_20261107.txt',config_difference={'max_response_tokens':{'old':10240,'new':cfg['max_response_tokens'],'reason':'user preregistered cost-p95 rule'}}),indent=2))
 lock=open('/data/runs/gpu80.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 with (out/'run.log').open('a') as log:
  subprocess.run(['/data/venvs/qwen35/bin/python','-u',str(ROOT/'train.py'),'--method','grpo','--config',str(out/'launch_config.json'),'--output-dir',str(out),'--eval-only'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 # Continue immediately into F02 only after all F01 evaluation outputs exist.
 assert read(out/'status.json')['status']=='evaluation_complete'
 for name in ['final_test.json','supplemental500/summary.json','filtered409.json','map8.json']:assert (out/name).exists(),name
 f02=Path('/data/runs/stage2/F02_noSFT_GRPO4_seed42');f02.mkdir(parents=True,exist_ok=True)
 cfg.update(steps=400,eval_every=25,seed=42,flat_group_size=4,run_label='F02_noSFT_GRPO4_seed42')
 (f02/'launch_config.json').write_text(json.dumps(cfg,indent=2))
 (f02/'hardware.json').write_text((out/'hardware.json').read_text())
 command=['/data/venvs/qwen35/bin/python','-u',str(ROOT/'train.py'),'--method','grpo','--config',str(f02/'launch_config.json'),'--output-dir',str(f02)]
 for attempt in range(2):
  if attempt and not (f02/'checkpoint.pt').exists():break
  with (f02/'run.log').open('a') as log:
   rc=subprocess.run(command+(['--resume'] if attempt else []),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
  if rc==0 and read(f02/'status.json').get('status')=='complete':return
 (f02/'needs_manual.json').write_text(json.dumps(dict(status='待人工',reason='F02 failed; checkpoint retry attempted if available',preserve_all_evidence=True),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
