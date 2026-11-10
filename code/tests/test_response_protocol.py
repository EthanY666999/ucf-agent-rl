import ast,copy,gzip,json,subprocess,sys,unittest
from pathlib import Path
from types import SimpleNamespace as NS
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import patr_core as core
from response_protocol import NAME,first_block
class ActualProtocol(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  from transformers import AutoTokenizer
  cls.tok=AutoTokenizer.from_pretrained('/data/models/Qwen3.5-2B',local_files_only=True)
  run=Path('/data/runs/hard_queue/S03_grpo_seed42');cls.cfg=json.loads((run/'config.json').read_text())
  with gzip.open(run/'rollouts_0001.jsonl.gz','rt') as f:cls.row=json.loads(next(f))
  # Execute only the historical CPU apply_response function with current core
  # globals. No historical train module/model allocation is imported.
  source=subprocess.check_output(['git','show','95b78572:code/patr_core.py'],cwd=ROOT,text=True)
  node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='apply_response')
  namespace=vars(core).copy();exec(compile(ast.Module(body=[node],type_ignores=[]),'historical_apply_response','exec'),namespace)
  cls.old=staticmethod(namespace['apply_response'])
 def out(self,text,reason='stop'):
  ids=self.tok.encode(text,add_special_tokens=False)+[self.tok.eos_token_id]
  return NS(token_ids=ids,logprobs=[{t:NS(logprob=-.1)} for t in ids],text=text,finish_reason=reason)
 def branch(self,cfg):return core.start_branch(self.row['task'],0,self.tok,cfg,'0')
 def view(self,b):return {k:getattr(b,k) for k in ['ids','mask','old_logprobs','actions','trace','reward','turns','terminal']}
 def test_disabled_path_identical_with_actual_tokenizer(self):
  for text in ['```Right```','``` Down ```','reason ```Left``` then ```Up```','invalid','```Down``` extra']:
   for reason in ['stop','length']:
    a=self.branch(self.cfg);b=copy.deepcopy(a);out=self.out(text,reason)
    self.old(a,out,self.tok,self.cfg);core.apply_response(b,out,self.tok,self.cfg)
    self.assertEqual(self.view(a),self.view(b))
 def test_first_block_end_executed_even_if_unused_tail_hit_length(self):
  cfg=dict(self.cfg,response_protocol=NAME);b=self.branch(cfg);prompt=len(b.ids)
  out=self.out('draft ``` Right ``` later ```Left```','length')
  core.apply_response(b,out,self.tok,cfg)
  keep=self.tok.encode('draft ``` Right ```',add_special_tokens=False)
  self.assertEqual(b.ids[prompt:prompt+len(keep)],keep)
  self.assertEqual(b.mask[prompt:prompt+len(keep)],[1]*len(keep))
  self.assertEqual(b.ids[prompt+len(keep)],self.tok.eos_token_id);self.assertEqual(b.mask[prompt+len(keep)],0)
  self.assertEqual(b.actions,[3]);self.assertEqual(b.trace[0]['response'],'draft ``` Right ```')
  self.assertEqual(sum(b.mask),len(keep));self.assertEqual(len(b.ids),len(b.old_logprobs))
 def test_unfinished_first_draft_does_not_hide_first_complete_block(self):
  cfg=dict(self.cfg,response_protocol=NAME);b=self.branch(cfg)
  core.apply_response(b,self.out('draft ``` Right </think> now ```Down``` later ```Left```'),self.tok,cfg)
  self.assertEqual(b.actions,[2]);self.assertTrue(b.trace[0]['response'].endswith('```Down```'))
 def test_atomic_closing_token_retained_without_retokenizing(self):
  cfg=dict(self.cfg,response_protocol=NAME);b=self.branch(cfg);prompt=len(b.ids)
  out=self.out('Output ```Up```. discarded ```Down```')
  core.apply_response(b,out,self.tok,cfg)
  n=sum(b.mask);self.assertEqual(b.ids[prompt:prompt+n],out.token_ids[:n])
  self.assertEqual(b.actions,[4]);self.assertEqual(b.trace[0]['response'],'Output ```Up```.')
  self.assertEqual(b.trace[0]['protocol_cutoff']['boundary_token_overhang'],'.')
  self.assertEqual(b.old_logprobs[prompt:prompt+n],[-.1]*n)
  from response_protocol import protocol_metrics
  report=protocol_metrics([b]);self.assertEqual(report['boundary_token_extra_fraction'],1.)
 def test_no_complete_block_retains_invalid_or_length_behavior(self):
  cfg=dict(self.cfg,response_protocol=NAME)
  for reason in ['stop','length']:
   b=self.branch(cfg);out=self.out('```unfinished',reason);core.apply_response(b,out,self.tok,cfg)
   self.assertNotIn('protocol_cutoff',b.trace[0]);self.assertEqual(b.trace[0].get('action'),None if reason=='length' else 0)
if __name__=='__main__':unittest.main()
