import copy,math,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'ucf')]
from ucf_corrected import complement_stratum,sampling_restriction,first_complement_output,leave_question_out_constants,select_corrected
from ucf_select import feature
from response_protocol import configure_agent,parse_first_block,truncate_response,NAME
from vllm import SamplingParams
from test_first_fence import Tok,fixture,MAP,FENCE
class Complement(unittest.TestCase):
 def test_all_class_complements_including_residual(self):
  p={0:.08,1:.12,2:.2,3:.3,4:.3};classes={0:'STAY',1:'STAY',2:'A',3:'B',4:'C'}
  probs=np.array([.08,.06,.06,.1,.1,.15,.15,.15,.15]);tokens=[90]+list(MAP)
  for token in tokens:
   with patch('ucf_corrected.consequence_classes',return_value=classes):s=complement_stratum({'branch':NS(env=None),'pi':p},token,MAP)
   allowed=[i for i,t in enumerate(tokens) if classes[MAP.get(t,0)]!=s['original_class']]
   self.assertAlmostEqual(sum(probs[i] for i in allowed),s['pi_cf']);self.assertEqual(s['unsampled_mass'],0)
   params=SamplingParams(**sampling_restriction(s),max_tokens=1)
   if s['original_class']=='STAY':self.assertNotIn(90,params.allowed_token_ids)
   else:self.assertIsNone(params.allowed_token_ids);self.assertIsNone(params.logit_bias);self.assertEqual(params.n,256)
   q=probs[allowed]/probs[allowed].sum();draws=np.random.default_rng(20261109).choice(allowed,200000,p=q)
   freq=np.array([(draws==i).mean() for i in allowed]);self.assertLess(float(np.max(abs(freq-q)/np.sqrt(q*(1-q)/len(draws)))),4)
 def test_own_question_support_excluded(self):
  a={0:[.5]*4,1:[.5]*4,2:[.5]*4};b=copy.deepcopy(a);b[0]+=[.7,.2,.8]
  self.assertEqual(leave_question_out_constants(a)[0],leave_question_out_constants(b)[0])
  b=copy.deepcopy(a);b[1]+=[.7,.2,.8]
  self.assertNotEqual(leave_question_out_constants(a)[0],leave_question_out_constants(b)[0])
 def test_actual_feature_calibration_selection_own_suffix(self):
  first=[NS(task_id=ti,lineage=str(j),reward=j%2,terminal='success' if j%2 else 'hole') for ti in range(3) for j in range(4)]
  candidates={ti:[dict(branch=NS(task_id=ti),j=j,turn=1,S=.3,H_cons=.5) for j in range(4)] for ti in range(3)}
  results=[]
  for extra in [False,True]:
   cs=copy.deepcopy(candidates)
   if extra:cs[0].append(dict(branch=NS(task_id=0),j=0,turn=2,S=.8,H_cons=.9))
   weights={ti:[feature(n,first)[0] for n in ns] for ti,ns in cs.items()}
   constants=leave_question_out_constants(weights)
   with tempfile.TemporaryDirectory() as td:
    _,rows=select_corrected(cs,first,[{'uid':str(i)} for i in range(3)],constants,42,1,Path(td)/'selection')
   results.append(rows[0])
  self.assertEqual(results[0],results[1])

class Protocol(unittest.TestCase):
 def test_first_complete_block_not_last(self):
  self.assertEqual(parse_first_block(None,'reason ``` Right ``` later ```Left```')[1],'3')
  self.assertEqual(parse_first_block(None,'```unfinished </think> ```\nDown```')[1],'2')
  self.assertEqual(parse_first_block(None,'```grid\nabc``` then ```4```')[1],'4')
 def test_cut_ids_logprobs_without_retokenization(self):
  pre,_,_=fixture();ids=[90,FENCE,30,FENCE,91,FENCE,10,FENCE,2]
  out=NS(token_ids=ids,logprobs=[{t:NS(logprob=-float(i+1))} for i,t in enumerate(ids)],text='',finish_reason='length')
  cut,meta=truncate_response(pre,out,Tok())
  self.assertEqual(cut.token_ids,ids[:4]);self.assertEqual(cut.logprobs,out.logprobs[:4]);self.assertEqual(cut.finish_reason,'stop');self.assertEqual(meta['discarded_sampled_tokens'],5)
 def test_cf_copied_prefix_and_new_suffix(self):
  pre,_,_=fixture();pre.ids+=[90,FENCE];pre.mask+=[1,1];pre.old_logprobs+=[-.2,-.3]
  out=NS(token_ids=[20,FENCE,91,2],logprobs=[{t:NS(logprob=-.1)} for t in [20,FENCE,91,2]],text='',finish_reason='stop')
  cut,_=truncate_response(pre,out,Tok());self.assertEqual(cut.token_ids,[20,FENCE]);self.assertEqual(cut.text,'reason ```Down```')
 def test_sft_agent_unchanged(self):
  a=NS(_parse_model_response=lambda x:('old','1'));old=a._parse_model_response;configure_agent(a,{})
  self.assertIs(a._parse_model_response,old)
if __name__=='__main__':unittest.main()

class Rejection(unittest.TestCase):
 def test_ordered_first_accepted_and_exhaustion(self):
  classes={0:'STAY',1:'STAY',2:'A',3:'B',4:'C'}
  s={'allowed_tokens':None,'original_class':'B'}
  def output(i,t):return NS(index=i,token_ids=[t],logprobs=[{t:NS(logprob=-1.)}])
  outputs=[output(i,30) for i in range(256)];outputs[5]=output(5,90);outputs[8]=output(8,20)
  chosen,audit=first_complement_output(list(reversed(outputs)),s,MAP,classes)
  self.assertEqual(chosen.index,5);self.assertEqual(chosen.token_ids,[90]);self.assertEqual(len(audit),256)
  chosen,audit=first_complement_output([output(i,30) for i in range(256)],s,MAP,classes)
  self.assertIsNone(chosen);self.assertTrue(all(x['in_original_class'] for x in audit))
 def test_residual_frequency_from_actual_256_unrestricted_proposals(self):
  rng=np.random.default_rng(20261109);p=np.array([.08,.12,.2,.3,.3]);original=3
  observed=0;accepted=0;failures=0
  for _ in range(100):
   proposals=rng.choice(5,(2000,256),p=p);ok=proposals!=original;has=ok.any(1);idx=ok.argmax(1)
   chosen=proposals[np.arange(len(proposals)),idx][has]
   observed+=int((chosen==0).sum());accepted+=len(chosen);failures+=int((~has).sum())
  q=.08/.7;z=abs(observed-accepted*q)/math.sqrt(accepted*q*(1-q))
  self.assertLess(z,4);self.assertEqual(failures,0)
 def test_required_reward_and_candidate_suffix_causality(self):
  sys.path.insert(0,str(ROOT/'scripts'))
  from q35_v21_causality_audit import audit
  result=audit();self.assertTrue(result['passed']);self.assertTrue(result['checks']['other_three_rho_may_change'])
