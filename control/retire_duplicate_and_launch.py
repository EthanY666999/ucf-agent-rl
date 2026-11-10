"""Remove only the published, redundant step16 source; preserve both new arms."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

STATE=Path('/data/runs/ucf_v3_budget10240_20261110')
def read(p):return json.loads(Path(p).read_text())
def atomic(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--published-commit',required=True)
    parser.add_argument('--published-manifest-sha256',required=True);args=parser.parse_args()
    manifest=STATE/'planned_duplicate_removal.json'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()==args.published_manifest_sha256
    plan=read(STATE/'plan.json');proof=read(STATE/'checkpoint_forks.json');m=read(manifest)
    assert proof['tensor_optimizer_rng_equal'] and len(m['replacements'])==2
    assert read(STATE/'admission_exit.json')['returncode']==0
    assert not (STATE/'training_launch.json').exists(),'Already launched; inspect live state instead'
    locks=[]
    for path in plan['locks']:
        f=open(path,'a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    source=Path(m['path'])
    assert str(source)=='/home/vipuser/ucf-v3-runs-20261110/V3_S_ucf_nosft_scoped_seed42/checkpoint.pt'
    assert source.is_file() and not source.is_symlink() and source.stat().st_nlink==1
    assert source.stat().st_size==m['bytes'] and source.stat().st_mtime<=proof['unix']
    assert read(source.parent/'status.json')['step']==16
    inodes=[]
    for r in m['replacements']:
        path=Path(r['path']);st=path.stat();inodes.append((st.st_dev,st.st_ino))
        assert st.st_size==r['bytes'] and st.st_mtime<=proof['unix']
        init=read(path.parent/'initialization.json');status=read(path.parent/'status.json')
        assert init['source_checkpoint_sha256']==m['sha256'] and init['tensor_optimizer_rng_sha256']==r['tensor_optimizer_rng_sha256']
        assert status['status']=='paused_checkpointed' and status['step']==16
    assert len(set(inodes))==2 and (source.stat().st_dev,source.stat().st_ino) not in inodes
    before={p:shutil.disk_usage(p).free for p in ('/home/vipuser','/data')}
    record=dict(manifest=m,manifest_sha256=args.published_manifest_sha256,published_commit=args.published_commit,
        verification='Fork transaction fully hashed both serialized checkpoints and verified all model/optimizer/RNG payloads; stable file sizes/mtimes and paused statuses rechecked under exclusive GPU/queue locks',
        exact_early_checkpoint_not_indefinitely_archived=True,source_weights_downloaded_to_mac=False,
        before_free_bytes=before,unix=time.time())
    atomic(STATE/'duplicate_retirement_planned_execution.json',record)
    source.unlink()
    after={p:shutil.disk_usage(p).free for p in before}
    record.update(deleted=True,after_free_bytes=after,released_system_bytes=after['/home/vipuser']-before['/home/vipuser'])
    atomic(STATE/'duplicate_retirement_receipt.json',record)
    atomic(source.parent/'CHECKPOINT_LOCATION.json',dict(source_step=16,source_sha256=m['sha256'],
        retired=True,replacements=m['replacements'],receipt=str(STATE/'duplicate_retirement_receipt.json'),
        note='Active continuation checkpoints advance; original step16 is not archived indefinitely. Historical evidence is retained.'))
    for e in plan['entries'][:2]:assert shutil.disk_usage(e['output']).free>=32*2**30
    env=dict(os.environ,UCF_INHERITED_QUEUE_LOCK_FD=str(locks[0].fileno()),UCF_INHERITED_GPU_LOCK_FD=str(locks[1].fileno()))
    env.pop('CUDA_VISIBLE_DEVICES',None)
    with (STATE/'training_manager.log').open('a') as log:
        p=subprocess.Popen([plan['python'],'-u',plan['manager_script'],str(STATE/'plan.json')],
            cwd=Path(plan['runtime']).parent,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,pass_fds=tuple(f.fileno() for f in locks))
    atomic(STATE/'training_launch.json',dict(pid=p.pid,unix=time.time(),source_step=16,
        max_response_tokens=10240,task_sft=False,shared_rl_warmstart=True,plan_sha256=hashlib.sha256((STATE/'plan.json').read_bytes()).hexdigest()))
    print(json.dumps(dict(manager_pid=p.pid,released_gib=record['released_system_bytes']/2**30,
        free_gib={k:v/2**30 for k,v in after.items()})))

if __name__=='__main__':main()
