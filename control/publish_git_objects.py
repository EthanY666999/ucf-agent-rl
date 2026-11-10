"""Upload small git objects through GitHub REST when git's HTTP transport stalls."""
import base64,json,subprocess,tempfile,time,sys
from pathlib import Path
repo=Path('/Users/ethan/Documents/ChatGPT/ACL2027/work/ucf-v3-20261110/publication')
api='repos/EthanY666999/ucf-agent-rl';branch='experiments/ucf-v3-no-sft';base=sys.argv[1]
local_base=sys.argv[2] if len(sys.argv)>2 else base
def git(*args):return subprocess.check_output(['git',*args],cwd=repo)
def request(endpoint,data=None):
 for attempt in range(3):
  try:
   if data is None:cmd=['gh','api',api+'/'+endpoint]
   else:
    with tempfile.NamedTemporaryFile(mode='w',suffix='.json',delete=False) as f:json.dump(data,f);name=f.name
    cmd=['gh','api','-X','POST',api+'/'+endpoint,'--input',name]
   try:r=subprocess.run(cmd,capture_output=True,timeout=35,check=True)
   finally:
    if data is not None:Path(name).unlink(missing_ok=True)
   return json.loads(r.stdout)
  except (subprocess.CalledProcessError,subprocess.TimeoutExpired):
   if attempt==2:raise
   time.sleep(2)
head=request('git/ref/heads/'+branch)['object']['sha'];assert head==base,(head,base)
base_tree=request('git/commits/'+base)['tree']['sha'];elements=[]
# REST-created commits can have the same tree but a different local commit ID.
# When HTTP git fetch is unavailable, prove tree equality before diffing a
# local counterpart. The remote lease and commit parent remain the remote SHA.
assert git('rev-parse',local_base+'^{tree}').decode().strip()==base_tree
cache_path=repo.parent/'blob_upload_cache.json';cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
for row in git('diff','--name-status',local_base,'HEAD').decode().splitlines():
 status,path=row.split('\t');assert status in ('A','M'),'No implicit deletions or renames'
 content=git('show','HEAD:'+path);assert len(content)<99_000_000
 obj=git('rev-parse','HEAD:'+path).decode().strip()
 if path.endswith('.gz'):
  if obj not in cache:
   actual=request('git/blobs',dict(content=base64.b64encode(content).decode(),encoding='base64'))['sha'];assert actual==obj
   cache[obj]=path;cache_path.write_text(json.dumps(cache,indent=2)+'\n');print('uploaded',path,len(content),flush=True)
  elements.append(dict(path=path,mode='100644',type='blob',sha=obj))
 else:elements.append(dict(path=path,mode='100644',type='blob',content=content.decode()))
tree=request('git/trees',dict(base_tree=base_tree,tree=elements))['sha']
commit=request('git/commits',dict(message=git('log','-1','--format=%B').decode().strip(),tree=tree,parents=[base],author=dict(name='EthanY666999',email='238026754+EthanY666999@users.noreply.github.com'),committer=dict(name='EthanY666999',email='238026754+EthanY666999@users.noreply.github.com')))
result=dict(base=base,tree=tree,sha=commit['sha'],files=len(elements),branch=branch,ref_not_updated=True)
(repo.parent/'pending_publish.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
