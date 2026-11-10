import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import q35_v21_gate as gate
from hard_queue_plan import compile_plan
class V21Gate(unittest.TestCase):
 def test_threshold_is_inclusive_and_not_widened(self):
  self.assertTrue(gate.offline_passed(dict(response_protocol='first_complete_action_block',source_experiment='S13',complete_steps=400,passed=True,mismatch_threshold=.02,mismatch_fraction=.02,training_only=True)))
  self.assertFalse(gate.offline_passed(dict(response_protocol='first_complete_action_block',source_experiment='S13',complete_steps=400,passed=True,mismatch_threshold=.03,mismatch_fraction=.025,training_only=True)))
  self.assertFalse(gate.offline_passed(dict(response_protocol='first_complete_action_block',source_experiment='S13',complete_steps=400,passed=True,mismatch_threshold=.02,mismatch_fraction=.0200001,training_only=True)))
 def test_cpu_receipt_must_match_current_source(self):
  with tempfile.TemporaryDirectory() as td:
   state=Path(td)
   for name in ['cpu_functional.json','htsim_first_fence.json','prefix_causality.json']:(state/name).write_text('{"passed":true}')
   with patch.object(gate,'STATE',state),patch.object(gate,'sha',return_value='current'):
    self.assertFalse(gate.corrected_cpu_ready())
    (state/'cpu_functional.json').write_text('{"passed":true,"source_manifest_sha256":"current"}')
    self.assertTrue(gate.corrected_cpu_ready())
 def test_failed_offline_never_launches_gpu(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);s=p/'hard_queue/S13_grpo_seed42';s.mkdir(parents=True);(s/'verification.json').write_text('{"passed":true}')
   state=p/'state';state.mkdir()
   for name in ['F02_offline.json','S13_offline.json']:(state/name).write_text(json.dumps(dict(passed=False,mismatch_threshold=.02,mismatch_fraction=.128,training_only=True)))
   for name in ['cpu_functional.json','htsim_first_fence.json']:(state/name).write_text('{"passed":true}')
   with patch.object(gate,'STATE',state),patch.object(gate,'RUNS',p),patch.object(gate.subprocess,'Popen') as launch:
    self.assertFalse(gate.gpu_admission({},None));launch.assert_not_called()
   self.assertFalse(json.loads((state/'admission.json').read_text())['GPU_started'])
 def test_only_two_authorized_runs_enable_v21(self):
  plan=compile_plan(json.loads((ROOT/'configs/frozenlake_q35_2b.json').read_text()),json.loads(Path('/data/runs/stage0/cost/result.json').read_text()))
  self.assertEqual([e['id'] for e in plan['entries'] if e['config'].get('fork_point')=='first_fence'],['S15','F10','F34'])
  for e in plan['entries']:
   if e['id'] in ['S15','F10','F34']:
    self.assertIn('--fork_point',e['command']);self.assertIn('BOTH',e['comparison_caveat'])
   elif e['method']=='ucf':self.assertNotIn('fork_point',e['config'])
if __name__=='__main__':unittest.main()
