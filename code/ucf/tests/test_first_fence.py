# CPU unit tests mock only the vLLM request-parameter value object; no engine is started.
import pytest
@pytest.fixture(autouse=True)
def cpu_sampling_params(monkeypatch):
    import sys
    from types import SimpleNamespace
    def params(**kwargs):return SimpleNamespace(**dict({'allowed_token_ids':None},**kwargs))
    monkeypatch.setitem(sys.modules,'vllm',SimpleNamespace(SamplingParams=params))

import copy, math, sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'ucf')]
import patr_core as core
import ucf_first_fence as ff
from first_fence import locate, action_masses
from ucf_select import stratum, feature, select_all
from ucf_weights import fork_rows
FENCE=999
MAP={10:1,11:1,20:2,21:2,30:3,31:3,40:4,41:4}
class Tok:
 eos_token_id=2
 def encode(self,text,**kw):
  if text=='```':return [FENCE]
  word=text[3:-3];return [FENCE,{'Left':10,'Down':20,'Right':30,'Up':40}[word.strip()]+word.startswith(' '),FENCE]
 def decode(self,ids,**kw):
  table={FENCE:'```',2:'',90:'reason ',91:'more ',10:'Left',11:' Left',20:'Down',21:' Down',30:'Right',31:' Right',40:'Up',41:' Up'}
  return ''.join(table.get(x,'?') for x in ids)
def fixture():
 env=NS(desc=np.array([list('SF'),list('FG')],dtype='S1'),s=0,nrow=2,ncol=2)
 pre=core.Branch(0,env,NS(_parse_model_response=lambda text:('','1' if text.endswith('```Left```') else '3')),[100],[0],[0.],lineage='0')
 b=copy.deepcopy(pre);b.ids += [90,91,FENCE,30,FENCE,2];b.mask += [1]*6;b.old_logprobs += [-.3,-.4,-.2,-1.,-.1,-.1]
 b.trace=[{'response':'reason more ```Right```'}]
 n=dict(branch=pre,j=0,turn=1,state=0,boundary=1,S=.3,parent_action=3)
 return pre,b,n
class FirstFence(unittest.TestCase):
 def test_first_not_last_and_no_fence(self):
  self.assertEqual(locate([5,90,FENCE,30,FENCE],1,5,FENCE),3)
  self.assertIsNone(locate([FENCE,90,30],1,3,FENCE))
 def test_suffix_cannot_change_node_probability_or_selection(self):
  pre,b,n=fixture();cfg=dict(max_response_tokens=100,max_context_tokens=100)
  others=[copy.deepcopy(b) for _ in range(3)]
  for j,o in enumerate(others,1):o.lineage=str(j);o.reward=j%2;o.terminal='success' if o.reward else 'hole'
  results=[]
  for suffix in [[30,FENCE,2],[20,91,40,FENCE,30],[2]]:
   x=copy.deepcopy(b);x.ids=x.ids[:4]+suffix;x.mask=[0]+[1]*(len(x.ids)-1);x.old_logprobs=[0.]+[-.2]*(len(x.ids)-1)
   def probe(cs,*args):
    for ns in cs.values():
     for v in ns:v.update(H_cons=.4,pi={0:.1,1:.2,2:.2,3:.3,4:.2})
    return {}
   with patch.object(ff,'initial',return_value=([x]+others,[copy.deepcopy(n)],{},{})),patch.object(ff,'probe_first_fence',side_effect=probe):
    first,cs,_,_=ff.initial_ucf(None,[{'uid':'x'}],None,Tok(),cfg,42)
   v=cs[0][0]
   with tempfile.TemporaryDirectory() as td:
    _,rows=select_all(cs,first,[{'uid':'x'}],.5,42,1,Path(td)/'s')
   results.append((v['branch'].ids,v['boundary'],v['pi'],rows))
  self.assertEqual(results[0],results[1]);self.assertEqual(results[0],results[2])
 def test_stratum_depends_only_on_first_successor_not_later_suffix(self):
  pre,_,_=fixture();n=dict(branch=pre,pi={0:.1,1:.2,2:.2,3:.3,4:.2})
  self.assertEqual(stratum(n,[FENCE,30,90,2],MAP,FENCE),stratum(n,[FENCE,30,FENCE,10,FENCE],MAP,FENCE))
  self.assertNotEqual(stratum(n,[FENCE,30],MAP,FENCE)['original_class'],stratum(n,[FENCE,20],MAP,FENCE)['original_class'])
 def test_all_variants_no_fence_factor_and_sampling_frequency(self):
  raw={t:math.log(p) for t,p in zip(MAP,[.04,.06,.1,.2,.15,.05,.2,.1])}
  p=action_masses(raw,MAP);self.assertAlmostEqual(p[0],.1);self.assertAlmostEqual(p[1],.1)
  pre,_,_=fixture();n=dict(branch=pre,pi=p);st=stratum(n,[FENCE,30],MAP,FENCE)
  allowed=st['allowed_tokens'];probs=np.array([math.exp(raw[t]) for t in allowed]);probs/=probs.sum()
  draws=np.random.default_rng(20261109).choice(allowed,200000,p=probs)
  freq=np.array([(draws==t).mean() for t in allowed]);self.assertLess(float(np.max(np.abs(freq-probs)/np.sqrt(probs*(1-probs)/len(draws)))),4)
 def test_prompt_scoring_includes_every_variant_unrestricted(self):
  pre,b,n=fixture();n['branch']=copy.deepcopy(pre);n['branch'].ids=b.ids[:4]
  seen=[]
  def request(engine,jobs):
   result={}
   for key,ids,params in jobs:
    seen.append((ids,params));x=key[1]
    result[key]=NS(prompt_logprobs=[None]*(len(ids)-1)+[{x:NS(logprob=math.log(.1))}])
   return result,dict(input_tokens=40,generated_tokens=8,request_seconds=0.,seconds=0.)
  with patch.object(ff,'requests',side_effect=request):ff.probe_first_fence({0:[n]},None,Tok(),{'max_context_tokens':100},42)
  self.assertEqual({ids[-1] for ids,_ in seen},set(MAP))
  self.assertTrue(all(ids[:-1]==b.ids[:4] and p.allowed_token_ids is None and p.prompt_logprobs==1 for ids,p in seen))
  self.assertAlmostEqual(n['pi'][0],.2);self.assertEqual(n['exact_logprobs'][30],math.log(.1))
 def test_cf_prefix_copy_unrestricted_old_lp_mask_and_last_parser(self):
  pre,o,n=fixture();n['branch']=copy.deepcopy(pre)
  for attr in ['ids','mask','old_logprobs']:setattr(n['branch'],attr,list(getattr(o,attr)[:4]))
  n.update(boundary=4,turn_start=1,pi={0:.1,1:.2,2:.2,3:.3,4:.2},rho=.2,baseline=1/3,exact_logprobs={x:-2. for x in MAP})
  cfg=dict(max_turns=10,max_response_tokens=100,max_context_tokens=100,temperature=1.,repeat_limit=6,ucf_beta=.25,ucf_variant='ucf')
  key=(0,0,1);seen=[]
  def requests(engine,jobs):
   for _,ids,params in jobs:seen.append((ids,params))
   if len(seen)==1:return {key:NS(outputs=[NS(token_ids=[20],logprobs=[{20:NS(logprob=-.5)}])])},dict(input_tokens=4,generated_tokens=1)
   ids=[FENCE,90,FENCE,10,FENCE,2]
   return {key:NS(outputs=[NS(token_ids=ids,logprobs=[{x:NS(logprob=-.1)} for x in ids],finish_reason='stop')])},dict(input_tokens=5,generated_tokens=len(ids))
  def apply(b,out,tok,c):
   b.ids.extend(out.token_ids);b.mask.extend([1]*len(out.token_ids));b.old_logprobs.extend(lp[t].logprob for t,lp in zip(out.token_ids,out.logprobs));b.actions.append(1);b.turns=1;b.reward=0;b.terminal='hole'
   self.assertTrue(out.text.startswith('reason more ```Down'))
  fake=NS(make_env=lambda t,c:copy.deepcopy(pre.env),apply_response=apply)
  with patch.object(ff,'requests',side_effect=requests),patch.object(ff,'stream',return_value=([],[],dict(input_tokens=0,generated_tokens=0),{})):
   rows,children,cost=ff.counterfactual(fake,{0:[n]},[o],[{'uid':'x'}],None,Tok(),cfg,42,1)
  self.assertEqual(seen[0][0],o.ids[:4]);self.assertEqual(seen[1][0],o.ids[:4]+[20])
  self.assertIn(20,seen[0][1].allowed_token_ids)
  self.assertEqual(rows[1].ids[:4],o.ids[:4]);self.assertEqual(rows[1].old_logprobs[:4],o.old_logprobs[:4])
  self.assertEqual(rows[1].old_logprobs[4],-2.);self.assertEqual(rows[1].mask[:4],[0]*4)
  self.assertTrue(all(rows[1].mask[4:]));self.assertEqual(cost['cf_parse_mismatch'],1.)
 def test_eval_delegates_without_training(self):
  original=lambda *a,**k:'eval'
  self.assertEqual(ff.make_collect(None,original)([],None,None,{},'ucf',None,evaluation=True),'eval')
if __name__=='__main__':unittest.main()
