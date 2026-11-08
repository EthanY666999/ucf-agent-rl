"""Evaluate the exported, reloaded SFT checkpoint before any independent RL arm."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from run_experiment import ROOT,load_config


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    cfg,meta=load_config(ROOT/a.config)
    if meta['stage']!='sft':raise SystemExit('Provide the SFT stage config')
    if meta['gates']:raise SystemExit('Blocked release: '+', '.join(meta['gates']))
    out=ROOT/meta['output'];cfg['model']=str(out/'final_model')
    command=[sys.executable,str(ROOT/'code/train.py'),'--config',str(ROOT/'runtime'/f'{meta["id"]}.sft_eval.json'),
        '--method','grpo','--eval-only','--output-dir',str(out/'evaluation')]
    print(json.dumps(dict(reloaded_model=cfg['model'],command=command),indent=2))
    if not a.execute:return
    import fcntl
    runtime=ROOT/'runtime';runtime.mkdir(exist_ok=True)
    with (runtime/'one_experiment.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Another experiment holds the GPU lease')
        subprocess.run([sys.executable,str(ROOT/'scripts/verify_release.py')],check=True)
        subprocess.run([sys.executable,str(ROOT/'scripts/patch_vllm.py'),'--check'],check=True)
        if (out/'evaluation').exists():raise SystemExit('SFT evaluation exists; preserve it and inspect before retrying')
        Path(command[3]).write_text(json.dumps(cfg,indent=2)+'\n')
        subprocess.run(command,cwd=ROOT/'code',check=True)
        status=json.loads((out/'evaluation/status.json').read_text())
        if status['status']!='evaluation_complete':raise SystemExit('Reload evaluation incomplete')
        from model_identity import identity
        files=['status.json','config.json','evaluation/status.json','evaluation/final_test.json']
        evidence={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in files}
        (out/'sft_acceptance.json').write_text(json.dumps(dict(passed=True,
            model_identity=identity(out/'final_model'),evidence_sha256=evidence,
            evaluation='evaluation/final_test.json',used_for_model_selection=False),indent=2)+'\n')


if __name__=='__main__':main()
