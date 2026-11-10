"""CPU checks for packaging, environment compatibility and tree/advantage contracts."""
import hashlib,json,sys,unittest,os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from patr_core import make_datasets,make_env,route_patr,training_weights
from rllm.agents.frozenlake_agent import FrozenLakeAgent
class Candidate:
 def __init__(self,name,reward,env):
  self.lineage=name;self.reward=reward;self.env=env;self.actions=[];self.turns=5;self.order=();self.terminal=''
 def clone(self,suffix):
  import copy
  b=copy.deepcopy(self);b.lineage+=suffix;return b
class ReleaseTests(unittest.TestCase):
 def setUp(self):self.cfg=json.loads((ROOT/os.environ.get('TEST_CONFIG','configs/frozenlake_05b.json')).read_text())
 def test_frozen_source_identity(self):
  family=os.environ.get('SOURCE_MODEL_FAMILY',self.cfg['model']).lower()
  name='SOURCE_MANIFEST_q35.json' if ('qwen3.5' in family or family=='q35') else 'SOURCE_MANIFEST.json'
  m=json.loads((ROOT/name).read_text())
  for n,h in m['files'].items():self.assertEqual(hashlib.sha256((ROOT/n).read_bytes()).hexdigest(),h,n)
 def test_fixed_dataset_and_environment(self):
  d=make_datasets();self.assertEqual({k:len(v) for k,v in d.items()},{'train':10000,'test':100,'validation':100})
  env=make_env(d['validation'][0],self.cfg)
  agent=FrozenLakeAgent(max_steps=10);agent.update_from_env(env.render(),0.,False,{})
  self.assertNotEqual(int(agent.update_from_model('```Right```').action),0)
  self.assertEqual(int(agent.update_from_model('```Right``').action),0)
 def test_group_weights(self):
  b=[SimpleNamespace(task_id=0,reward=r,ntokens=n) for r,n in [(0,2),(1,4)]]
  rows=training_weights(b,1)
  self.assertLess(rows[0][1],0);self.assertGreater(rows[1][1],0)
  self.assertAlmostEqual(sum(x.ntokens*w for x,a,w in rows),1.)
  for x in b:x.reward=0
  self.assertTrue(all(a==0 for x,a,w in training_weights(b,1)))
 def test_random_has_no_score_pruning(self):
  env=make_env(make_datasets()['validation'][0],self.cfg)
  cfg=dict(self.cfg,scorer_progress_weight=0.)
  bs=[Candidate(str(i),float(i==0),env) for i in range(4)]
  for method in [False,True]:
   kept,done,events=route_patr(bs,cfg,np.random.default_rng(42),method)
   self.assertEqual(sum(x['kind']=='expand' for x in events),2)
   if method:self.assertFalse(any(x['kind']=='prune' for x in events))
if __name__=='__main__':unittest.main()
