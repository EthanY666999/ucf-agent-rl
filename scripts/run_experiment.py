"""One experiment at a time. Checks are CPU-only until an explicit --execute."""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from release_admission import ucf_release_gate
PATH_KEYS=['model','bootstrap_model','dataset_path','frozen_maps_path','eval_sets_dir']


def load_config(path):
    cfg=json.loads(path.read_text())
    metadata=cfg.pop('_release')
    for key in PATH_KEYS:
        if key in cfg:cfg[key]=str((ROOT/cfg[key]).resolve())
    for path_key,hash_key in [('dataset_path','dataset_sha256'),('frozen_maps_path','frozen_maps_sha256')]:
        if path_key in cfg:
            assert hashlib.sha256(Path(cfg[path_key]).read_bytes()).hexdigest()==cfg[hash_key], f'{path_key} hash mismatch'
    return cfg,metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--sft-audit',action='store_true')
    a=p.parse_args();path=(ROOT/a.config).resolve()
    cfg,meta=load_config(path)
    ucf_gate=ucf_release_gate(cfg,meta['method'])
    print(json.dumps(dict(experiment=meta['id'],stage=meta['stage'],gates=meta['gates'],
        ucf_release_gate=ucf_gate,parameters=cfg,smoke=a.smoke,
        will_allocate_gpu=a.execute and not a.sft_audit and not meta['gates'] and not ucf_gate),indent=2),flush=True)
    if not a.execute:return
    if ucf_gate:raise SystemExit(ucf_gate)
    # Never silently bypass missing ports, the 2B UCF acceptance or the SFT decision.
    gates=[g for g in meta['gates'] if not (a.sft_audit and g=='distributed_training_backend_required')]
    if gates:
        raise SystemExit('Not runnable in this release: '+', '.join(gates)+'. See docs; stop and report.')
    subprocess.run([sys.executable,str(ROOT/'scripts/verify_release.py')],check=True)
    if a.sft_audit and meta['stage']!='sft':raise SystemExit('--sft-audit requires an SFT config')
    if a.smoke and meta['stage']=='sft':raise SystemExit('SFT smoke uses --sft-audit; full SFT is one epoch')
    out=ROOT/meta['output']
    if a.smoke:out=ROOT/'runs/smoke'/meta['id']
    if out.exists() and any(out.iterdir()) and not a.resume and not a.sft_audit:
        raise SystemExit('Output already exists; use a matching --resume. Never overwrite evidence.')
    if a.resume and not (out/'checkpoint.pt').exists():raise SystemExit('No checkpoint to resume')
    if not Path(cfg['model'],'config.json').is_file():raise SystemExit('Local model is missing; download weights on collaborator hardware first')
    if meta['start']=='sft':
        from model_identity import verify_sft
        starting_model_identity=verify_sft(Path(cfg['model']).parent)
    else:starting_model_identity=None
    # ALFWorld and 9B have explicit gates above. Existing engine is single device.
    if not a.sft_audit:
        subprocess.run([sys.executable,str(ROOT/'scripts/patch_vllm.py'),'--check'],check=True)
        if meta['stage']=='train':
            import torch
            total=torch.cuda.get_device_properties(0).total_memory
            if total<120*1024**3:
                raise SystemExit('Current full-parameter 4B RL needs a larger single GPU; 80GiB is insufficient. Stop and discuss backend/hardware.')
    runtime=ROOT/'runtime';runtime.mkdir(exist_ok=True)
    with (runtime/'one_experiment.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Another experiment holds this release GPU lease')
        (ROOT/'code/artifacts').mkdir(exist_ok=True)
        config_out=runtime/(meta['id']+('.smoke' if a.smoke else '')+'.json')
        config_out.write_text(json.dumps(cfg,indent=2)+'\n')
        receipt=dict(config_sha256=hashlib.sha256(config_out.read_bytes()).hexdigest(),
            release_manifest_sha256=hashlib.sha256((ROOT/'RELEASE_MANIFEST.json').read_bytes()).hexdigest(),
            starting_model_identity=starting_model_identity,smoke=a.smoke)
        revision=subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True,capture_output=True)
        receipt['release_commit']=revision.stdout.strip() if revision.returncode==0 else None
        (runtime/(meta['id']+'.launch_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
        if meta['stage']=='sft':
            data=ROOT/meta['sft_data']
            if data.suffix=='.gz':
                materialized=runtime/(meta['id']+'.jsonl')
                with gzip.open(data,'rb') as src,materialized.open('wb') as dst:
                    import shutil;shutil.copyfileobj(src,dst)
                data=materialized
            command=[sys.executable,str(ROOT/'code/sft.py'),'--model',cfg['model'],'--data',str(data),'--output',str(out),'--optimizer',cfg['optimizer']]
            if a.sft_audit:command.append('--audit-only')
        else:
            command=[sys.executable,str(ROOT/'code/train.py'),'--config',str(config_out),'--method',meta['method'],'--output-dir',str(out)]
            if meta['stage']=='evaluate':command.append('--eval-only')
            if a.smoke:command.append('--smoke')
        if a.resume:command.append('--resume')
        env=dict(os.environ,PYTHONPATH=str(ROOT/'code'),TOKENIZERS_PARALLELISM='false')
        result=subprocess.run(command,cwd=ROOT/'code',env=env)
        if out.exists() and not a.sft_audit:
            (out/'release_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
            (out/'release_manifest.json').write_bytes((ROOT/'RELEASE_MANIFEST.json').read_bytes())
        if result.returncode:raise SystemExit('Experiment failed; evidence retained. Stop and ask before retrying.')
        if meta['stage']=='sft' and not a.sft_audit:
            # Exported SFT must be evaluated independently before subsequent RL.
            print('SFT exported. Run scripts/evaluate_sft.py before starting any RL arm.')


if __name__=='__main__':main()
