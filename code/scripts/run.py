"""Safe, portable entry point; leaves the archived trainer and algorithm unchanged."""
import argparse,fcntl,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--method',required=True,choices=['grpo','patr','tree_random'])
p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true')
p.add_argument('--config',default='configs/frozenlake_05b.json')
p.add_argument('--stop-after-step',type=int)
a=p.parse_args()
cfg_path=(ROOT/a.config).resolve()
cfg=json.loads(cfg_path.read_text())
if cfg['eval_samples']!=4:raise SystemExit('Archived evaluation_metrics requires exactly four samples per task.')
out=ROOT/'runs'/('smoke' if a.smoke else 'formal')/f'{a.method}_seed{cfg["seed"]}'
# Lock before checking output to avoid concurrent starts. All methods share one GPU.
lock=(ROOT/'artifacts/.run.lock').open('a')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit('Another training process holds the workspace lock.')
if a.resume:
 if not (out/'checkpoint.pt').exists():raise SystemExit(f'Cannot resume: {out}/checkpoint.pt is missing')
else:
 if out.exists() and any(out.iterdir()):raise SystemExit(f'Refusing to overwrite {out}; use --resume or a new seed/workspace.')
model=ROOT/cfg['model']
if not (model/'config.json').is_file():raise SystemExit('Base model missing. Run python scripts/prepare_model.py first.')
deadline=ROOT/'artifacts/deadline.json'
if deadline.exists():
 import time
 if json.loads(deadline.read_text())['unix']<=time.time():raise SystemExit('An expired artifacts/deadline.json would stop training. Review/remove your local deadline.')
cmd=[sys.executable,'-u',str(ROOT/'train.py'),'--method',a.method,'--config',str(cfg_path)]
for flag in ['smoke','resume']:
 if getattr(a,flag):cmd.append('--'+flag)
if a.stop_after_step is not None:cmd+=['--stop-after-step',str(a.stop_after_step)]
print('Output:',out,flush=True)
# exec preserves PID so SIGINT/SIGTERM reach the trainer; inherited lock lasts until exit.
os.set_inheritable(lock.fileno(),True)
os.chdir(ROOT);os.execv(sys.executable,cmd)
