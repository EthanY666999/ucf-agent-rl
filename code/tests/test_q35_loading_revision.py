import json,math,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from q35_loading_acceptance import compare_logprobs
from q35_thinking_policy import generated_policy,check_prompt_contract
import q35_after_s05_handoff as handoff
import q35_f03_continuation as continuation
from q35_run_receipt import atomic,sha

class NumericAcceptance(unittest.TestCase):
 def test_small_errors_pass(self):self.assertTrue(compare_logprobs([-1.,-2.],[-1.01,-2.01])['passed'])
 def test_local_weight_divergence_fails(self):self.assertFalse(compare_logprobs([-1.]*8,[-1.2]+[-1.]*7)['passed'])
 def test_systematic_drift_fails(self):self.assertFalse(compare_logprobs([-1.]*8,[-1.04]*8)['passed'])
 def test_nonfinite_and_incomplete_fail(self):
  for a,b in [([math.nan],[0.]),([0.],[]),([],[])]:
   with self.assertRaises(AssertionError):compare_logprobs(a,b)

class ThinkingPolicy(unittest.TestCase):
 def test_original_base_only_exception(self):
  self.assertTrue(generated_policy(22,'/data/models/Qwen3.5-2B')['passed'])
  self.assertFalse(generated_policy(22,'/data/runs/stage2/F03_SFT_seed42/sft/final_model')['passed'])
  self.assertFalse(generated_policy(1,'/unknown/model')['passed'])
  self.assertTrue(generated_policy(0,'/unknown/model')['passed'])
 def test_missing_empty_block_fails(self):
  class BadTokenizer:
   eos_token='EOS'
   def apply_chat_template(self,messages,tokenize=True,**kw):
    self.asserted=kw['enable_thinking'] is False
    return [1] if tokenize else '@@ASSISTANT@@EOSuser@@USER@@assistant'
   def decode(self,*args,**kwargs):return 'assistant'
  with self.assertRaisesRegex(AssertionError,'empty think'):check_prompt_contract(BadTokenizer())

class BoundaryAcceptance(unittest.TestCase):
 def test_step400_without_final_receipt_is_not_complete(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);atomic(p/'status.json',{'step':400,'status':'training'})
   self.assertFalse(handoff.verified(p))
   atomic(p/'verification.json',{'passed':True,'steps':399});self.assertFalse(handoff.verified(p))
   atomic(p/'verification.json',{'passed':True,'steps':400});self.assertTrue(handoff.verified(p))
 def test_only_owned_manual_marker_released(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'needs_manual.json';atomic(p,{'reason':'unrelated failure'})
   with self.assertRaises(AssertionError):handoff.release_markers({'created_markers':[str(p)],'authorized_manual_releases':{}})
   self.assertTrue(p.exists())
 def test_modified_manual_evidence_not_overridden(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'needs_manual.json';atomic(p,{'reason':'initial'})
   digest=sha(p);atomic(p,{'reason':'changed'})
   with self.assertRaises(AssertionError):handoff.release_markers({'created_markers':[],'authorized_manual_releases':{str(p):digest}})
 def test_revised_order(self):
  from hard_queue_plan import ORDER
  self.assertEqual([x[0] for x in ORDER[:13]],['S01','S02','S03','S04','S05','F03','H01','H02','H03','H04','H05','S13','S15'])

class CompatibilityAttempt(unittest.TestCase):
 def fixture(self,p):
  out=p/'stage2/F03_SFT_seed42';(out/'sft').mkdir(parents=True);atomic(out/'sft/status.json',{'status':'complete'})
  (p/'stage0').mkdir();atomic(p/'stage0/execution_ledger.json',{'user_revision_20261108':{'F03_evaluation_extra_attempts':1}})
  state=p/'queue';(state/'attempts').mkdir(parents=True)
  return out,state
 def test_no_second_extra_attempt(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);out,state=self.fixture(p);atomic(state/'attempts/F03_compatibility_extra_20261108.json',{'attempt':1})
   with patch.object(continuation,'RUNS',p),patch.object(continuation,'STATE',state),patch.object(continuation.subprocess,'Popen') as spawn:
    self.assertFalse(continuation.evaluate_f03_once(['train.py','--eval-only'],out,Mock()));spawn.assert_not_called()
 def test_extra_eval_preserves_normal_attempts_and_never_retrains(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);out,state=self.fixture(p);(out/'evaluation').mkdir();atomic(out/'evaluation/status.json',{'status':'failed'})
   normal=state/'attempts/normal.json';atomic(normal,[{'returncode':1},{'returncode':1}]);before=normal.read_bytes()
   def spawn(command,**kwargs):
    self.assertIn('--eval-only',command);self.assertNotIn('--resume',command)
    atomic(out/'evaluation/status.json',{'status':'evaluation_complete'});atomic(out/'evaluation/loading_logprob_check.json',{'passed':True})
    proc=Mock(pid=123);proc.wait.return_value=0;return proc
   with patch.object(continuation,'RUNS',p),patch.object(continuation,'STATE',state),patch.object(continuation.subprocess,'Popen',side_effect=spawn):
    self.assertTrue(continuation.evaluate_f03_once(['train.py','--eval-only'],out,Mock()))
   self.assertEqual(normal.read_bytes(),before)
   journal=json.loads((state/'attempts/F03_compatibility_extra_20261108.json').read_text())
   self.assertFalse(journal['SFT_retrained']);self.assertFalse(journal['consumes_experiment_recovery'])
   self.assertEqual(len(list(out.glob('evaluation_before_authorized_extra_*'))),1)

if __name__=='__main__':unittest.main()
