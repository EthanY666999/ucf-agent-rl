import copy,hashlib,json,math,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import os
FENCE=int(os.environ.get("UCF_TEST_FENCE","73594"))
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import patr_core as core
from ucf_select import Controller,feature,select_all,stratum,calibrate
from ucf_weights import pseudo,fork_rows,training_weights_ucf
from ucf_probe import forced_logprobs

def branch(ti=0,j=0,r=0):return core.Branch(ti,None,None,[99,1,2,10,3,4],[0,1,1,0,1,1],[0,-.2,-.3,0,-.4,-.5],lineage=str(j),reward=r,terminal='success' if r else 'hole')
class TestUCF(unittest.TestCase):
 def test_independence_and_durable_selection(self):
  bs=[branch(j=j,r=j%2) for j in range(4)];n=dict(branch=branch(),j=0,turn=1,S=.3,H_cons=.2)
  old=feature(n,bs);bs[0].reward=1;bs[0].terminal='success';bs[0].ids.extend([8]*10)
  self.assertEqual(old,feature(n,bs));self.assertEqual(old[1],2/3)
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'select.jsonl';picked,rows=select_all({0:[n]},bs,[{'uid':'x'}],.5,42,1,p)
   self.assertEqual(json.loads(p.read_text()),rows[0]);self.assertEqual(rows[0]['selected'],rows[0]['u']<rows[0]['rho'])
 def test_controller(self):
  c=Controller(.4);self.assertEqual(c.c,.4);self.assertEqual(c.finish(0),.5);self.assertAlmostEqual(c.finish(100),.4)
  for n in [0,2,3,10]*10:
   old=c.c;c.finish(n);self.assertGreaterEqual(c.c/old,.8-1e-12);self.assertLessEqual(c.c/old,1.25+1e-12)
  c0=calibrate([[.2,.5,.4,.3,.8]]*4);self.assertAlmostEqual(sum(np.clip(c0*np.array([.2,.5,.4,.3,.8]),.03,1)),2)
  self.assertEqual(calibrate([[0,0,0]]),0.)
  c=calibrate([[.2,0],[.5,0]])
  self.assertAlmostEqual(sum(min(1,max(.03,c*w)) for ws in [[.2,0],[.5,0]] for w in ws)/2,1.03)
 def test_strata(self):
  mapping={10:1,11:1,20:2,21:2,30:3,31:3,40:4,41:4}
  for board,state in [(['SF','FG'],0),(['SH','FG'],0),(['SF','FG'],1)]:
   env=NS(desc=np.array([list(s) for s in board],dtype='S1'),s=state,nrow=2,ncol=2)
   n={'branch':NS(env=env),'pi':{0:.02,1:.2,2:.2,3:.3,4:.28}}
   for ids in [[FENCE,10],[FENCE,20],[FENCE,30],[FENCE,40],[999,10],[]]:
    s=stratum(n,ids,mapping,FENCE);self.assertAlmostEqual(s['pi_o']+s['pi_cf']+s['unsampled_mass'],1)
    for a in s['allowed_actions']:self.assertTrue(set(t for t,x in mapping.items() if x==a)<=set(s['allowed_tokens']))
    if not ids or ids[0]!=FENCE:self.assertEqual(s['original_class'],'STAY')
 def test_old_logprob_uses_actor(self):
  n=dict(exact_logprobs={42:(-.1,-3.)},forced_vllm_logprob_restricted=0.)
  self.assertEqual(forced_logprobs(n,FENCE,42),(-.1,-3.))
 def test_actor_full_vocab_no_grad(self):
  import torch
  from ucf_probe import actor_logprobs
  class Backbone(torch.nn.Module):
   def forward(self,input_ids,attention_mask,use_cache):
    self.grad_enabled=torch.is_grad_enabled()
    positions=torch.arange(input_ids.shape[1]).expand_as(input_ids)
    return NS(last_hidden_state=torch.stack([input_ids.float()/100000,positions.float()],-1))
  class Actor(torch.nn.Module):
   def __init__(self):
    super().__init__();self.model=Backbone();self.lm_head=torch.nn.Linear(2,73600,bias=False)
    with torch.no_grad():self.lm_head.weight.copy_(torch.arange(147200).reshape(73600,2)/147200)
  actor=Actor();actor.train();actor.lm_head.weight.grad=torch.ones_like(actor.lm_head.weight);prior=actor.lm_head.weight.grad.clone()
  prefixes=[[5,7],[4,8,9]];mapping={10:1,11:1,20:2,21:2,30:3,31:3,40:4,41:4}
  scored=actor_logprobs(actor,prefixes,mapping,batch_size=2,fence=FENCE)
  self.assertTrue(actor.training);self.assertFalse(actor.model.grad_enabled);self.assertTrue(torch.equal(prior,actor.lm_head.weight.grad))
  with torch.no_grad():
   for i,pre in enumerate(prefixes):
    hidden=torch.tensor([[pre[-1]/100000,len(pre)-1],[FENCE/100000,len(pre)]])
    expected=actor.lm_head(hidden).float().log_softmax(-1)
    for x in mapping:
     self.assertAlmostEqual(scored[i][x][0],float(expected[0,FENCE]),places=5)
     self.assertAlmostEqual(scored[i][x][1],float(expected[1,x]),places=5)
    self.assertLess(sum(math.exp(v[0]+v[1]) for v in scored[i].values()),1)
 def test_part_a_and_variants(self):
  for variant in ['ucf','noIS','uniform']:
   bs=[branch(j=j,r=j%2) for j in range(4)];cf=branch(j=5,r=1)
   n=dict(boundary=4,rho=.2,baseline=1/3,stratum=dict(pi_o=.7,pi_cf=.28,cf_fallback=False))
   extra=fork_rows(bs[0],cf,n,.5,variant);rows=training_weights_ucf(bs+extra,1)
   self.assertEqual(rows[:4],core.training_weights(bs,1));self.assertEqual(extra[0].mask,[0,0,0,0,1,1]);self.assertEqual(extra[0].ntokens,2)
   self.assertEqual(extra[0].ids[:4],bs[0].ids[:4]);self.assertEqual(extra[0].old_logprobs[:4],bs[0].old_logprobs[:4])
   self.assertAlmostEqual(rows[4][1],2*(-1/3)*(1 if variant=='noIS' else .7))
   self.assertAlmostEqual(rows[5][1],2*(2/3)*(1 if variant=='noIS' else .28))
   self.assertAlmostEqual(rows[4][2],.5/(4*4)*(1 if variant=='noIS' else 5))
   self.assertEqual(rows[4][2],rows[5][2])
 def test_fallback(self):
  bs=[branch(j=j,r=j%2) for j in range(4)];n=dict(boundary=4,rho=.2,baseline=1/3,stratum=dict(cf_fallback=True,pi_o=1.,pi_cf=0.))
  rows=fork_rows(bs[0],None,n,.5,'ucf');self.assertEqual(len(rows),1);self.assertAlmostEqual(rows[0].ucf['adv'],-2/3)
  self.assertAlmostEqual(training_weights_ucf(bs+rows,1)[-1][2],.5*5/16) # stratum weight 1; HT retained
 def test_pack_and_remapping(self):
  import torch,train
  bs=[branch(ti=i,j=j,r=j%2) for i in range(2) for j in range(4)]
  bs.append(pseudo(bs[4],4,'orig_fork',.5,.2,4,.5,'ucf'))
  chosen=[1,0];idmap={old:new for new,old in enumerate(chosen)}
  mapped=[]
  for b in bs:
   c=copy.copy(b);c.task_id=idmap[b.task_id];mapped.append(c)
  self.assertEqual(mapped[-1].task_id,0)
  rows=training_weights_ucf(mapped,2)
  from unittest.mock import patch
  full,tensor=torch.full,torch.tensor
  def cpu_full(*a,**kw):kw['device']='cpu';return full(*a,**kw)
  def cpu_tensor(*a,**kw):kw['device']='cpu';return tensor(*a,**kw)
  with patch('torch.full',cpu_full),patch('torch.tensor',cpu_tensor):ids,attn,mask,adv,w,old=train.pack(rows,0)
  self.assertEqual(int(mask.sum()),len(adv));self.assertEqual(len(old),sum(b.ntokens for b,_,_ in rows))
 def test_initial_no_dedup(self):
  from ucf_collect import initial_ucf
  import ucf_collect
  from unittest.mock import patch
  nodes=[dict(branch=branch(j=j),j=j,turn=1,parent_action=1,boundary=1) for j in range(4)]
  with patch.object(ucf_collect,'initial',return_value=([branch(j=j) for j in range(4)],nodes,{},{})) as mocked,patch.object(ucf_collect,'probe_exact',return_value={}),patch.object(ucf_collect,'token_map',return_value=({},{})),patch.object(ucf_collect,'fence_id',return_value=FENCE):
   _,c,_,_=initial_ucf(None,[{'uid':'x'}],None,None,{},42)
  self.assertEqual(len(c[0]),4);self.assertFalse(mocked.call_args.kwargs['with_probes'])
 def test_cf_pipeline(self):
  import ucf_collect
  from unittest.mock import patch
  class Tok:
   eos_token_id=2
   def __len__(self):return 100000
   def encode(self,text,**kw):
    if text=='```':return [FENCE]
    word=text[3:-3];a={'Left':10,'Down':20,'Right':30,'Up':40}[word.strip()]
    return [FENCE,a+int(word.startswith(' ')),FENCE]
   def decode(self,ids,**kw):
    pieces={FENCE:'```',2:'',10:'Left',11:' Left',20:'Down',21:' Down',30:'Right',31:' Right',40:'Up',41:' Up'}
    return ''.join(pieces[x] for x in ids)
  env=NS(desc=np.array([list('SF'),list('FG')],dtype='S1'),s=0,nrow=2,ncol=2)
  pre=core.Branch(0,env,None,[99],[0],[0.],lineage='0',turns=0)
  original=copy.deepcopy(pre);original.ids += [FENCE,10,FENCE,2];original.mask += [1]*4;original.old_logprobs += [-.1,-1.,-.2,-.2];original.reward=0
  n=dict(branch=pre,j=0,turn=1,state=0,boundary=1,parent_action=1,pi={0:.1,1:.2,2:.2,3:.3,4:.2},rho=.2,baseline=1/3,exact_logprobs={x:(-.1,-2.) for x in [10,11,20,21,30,31,40,41]})
  cfg=dict(max_turns=10,max_response_tokens=100,max_context_tokens=100,temperature=1.,repeat_limit=6,ucf_beta=.5,ucf_variant='ucf')
  key=(0,0,1);lp=lambda x:NS(logprob=x)
  answers=[({key:NS(outputs=[NS(token_ids=[30],logprobs=[{30:lp(-.7)}])])},dict(input_tokens=2,generated_tokens=1)),({key:NS(outputs=[NS(token_ids=[FENCE,2],logprobs=[{FENCE:lp(-.2)},{2:lp(-.2)}],finish_reason='stop')])},dict(input_tokens=3,generated_tokens=2))]
  observed=[]
  def apply(b,out,tok,c):
   observed.append(out);b.ids.extend(out.token_ids);b.mask.extend([1]*len(out.token_ids));b.old_logprobs.extend(x[t].logprob for x,t in zip(out.logprobs,out.token_ids));b.actions.append(3);b.turns=1;b.reward=1;b.terminal='success'
  fake=NS(make_env=lambda t,c:copy.deepcopy(env),apply_response=apply)
  with patch.object(ucf_collect,'requests',side_effect=answers),patch.object(ucf_collect,'stream',return_value=([],[],dict(input_tokens=0,generated_tokens=0),{})):
   rows,children,cost=ucf_collect.counterfactual(fake,{0:[n]},[original],[{'uid':'x'}],None,Tok(),cfg,42,0)
  self.assertEqual(len(rows),2);self.assertEqual(observed[0].token_ids,[FENCE,30,FENCE,2]);self.assertEqual(rows[1].old_logprobs[1:3],[-.1,-2.]);self.assertEqual(rows[1].mask,[0,1,1,1,1]);self.assertEqual(cost['cf_parse_mismatch'],0)
 def test_actor_runtime_binding(self):
  from ucf_runtime import configure_builder
  actor,ref,engine,tok=NS(),NS(),NS(),NS()
  train=NS(build_models=lambda cfg,seed:(actor,ref,engine,tok))
  configure_builder(train);configure_builder(train)
  self.assertEqual(train.build_models({},42),(actor,ref,engine,tok));self.assertIs(engine._ucf_actor,actor)
 def test_gradient_diagnostics_enable_checkpoint_mode(self):
  import torch,train
  from unittest.mock import patch
  sys.path.insert(0,str(ROOT/'scripts'))
  import accept_ucf,phase1_ucf
  class Actor(torch.nn.Module):
   def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.tensor(0.));self.config=NS(attention_dropout=0.)
  actor,ref=Actor().eval(),Actor().eval();b=branch();rows=[(b,1.,1.)]
  def selected(model,*args,**kwargs):
   if kwargs.get('gradient'):self.assertTrue(model.training)
   return model.p.reshape(1)
  packed=(torch.ones((1,2),dtype=torch.long),torch.ones((1,2)),torch.ones((1,2)),torch.ones(1),torch.ones(1),torch.zeros(1))
  cfg=dict(clip_low=.28,clip_high=.28,kl_coefficient=.001)
  with patch.object(accept_ucf,'training_weights_ucf',return_value=rows),patch.object(train,'microbatches',return_value=[rows]),patch.object(train,'pack',return_value=packed),patch.object(train,'selected_logprobs',side_effect=selected):
   result=accept_ucf.component_grad(actor,ref,[b],cfg,NS(pad_token_id=0),'A')
   self.assertFalse(actor.training);self.assertAlmostEqual(result['grad_norm'],1.)
   g=phase1_ucf.grad(actor,rows,cfg,NS(pad_token_id=0))
   self.assertFalse(actor.training);self.assertAlmostEqual(float(g[0]),1.)
 def test_entry(self):
  text=(ROOT/'run_ucf.py').read_text()
  for term in ['platform_verification','rental_expiry','deadline.json']:self.assertNotIn(term,text)
if __name__=='__main__':unittest.main()
