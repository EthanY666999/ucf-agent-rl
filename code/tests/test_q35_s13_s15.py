import copy,json,tempfile,unittest,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from hard_queue_plan import compile_plan
from hard_queue_supervisor import predecessor_ready
from q35_s13_s15_compare import summarize
class QueueRevision(unittest.TestCase):
 def test_config_and_order(self):
  p=compile_plan(json.loads((ROOT/'configs/frozenlake_q35_2b.json').read_text()),json.loads(Path('/data/runs/stage0/cost/result.json').read_text()))
  es={e['id']:e for e in p['entries']}
  self.assertEqual([e['id'] for e in p['entries'][:13]],['S01','S02','S03','S04','S05','F03','H01','H02','H03','H04','H05','S13','S15'])
  for target,reference in [('S13','S03'),('S15','S05')]:
   a=es[target]['config'];b=es[reference]['config']
   self.assertEqual({k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)},({'model','max_response_tokens','run_label','fork_point','ucf_corrections','response_protocol'} if target=='S15' else {'model','max_response_tokens','run_label','response_protocol'}))
   self.assertEqual(a['model'],'/data/models/Qwen3.5-2B');self.assertEqual(a['max_response_tokens'],4096);self.assertEqual(a['eval_every'],25)
  self.assertEqual(es['S13']['strict_predecessor'],es['H05']['output']);self.assertEqual(es['S15']['strict_predecessor'],es['S13']['output']);self.assertEqual(es['H01']['strict_predecessor'],es['F03']['output'])
  self.assertNotIn('S14',es);self.assertTrue(any('S14' in x for x in p['remaining_legacy_after_this_plan']))
 def test_missing_failed_and_successful_predecessor(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);e={'strict_predecessor':td};self.assertFalse(predecessor_ready(e))
   (p/'verification.json').write_text(json.dumps({'passed':False}));self.assertFalse(predecessor_ready(e))
   (p/'verification.json').write_text(json.dumps({'passed':True}));self.assertTrue(predecessor_ready(e))
class Comparison(unittest.TestCase):
 def fixture(self):
  tasks=[{'uid':str(i),'map':'frozen'} for i in range(100)]
  records=[dict(task=t,lineage=str(j),reward=int(int(t['uid'])<20 or 20<=int(t['uid'])<50 and j==0),terminal='success' if int(t['uid'])<20 or 20<=int(t['uid'])<50 and j==0 else 'turn_limit',response_tokens=12+j) for t in tasks for j in range(4)]
  return tasks,records
 def test_known_counts(self):
  ts,rs=self.fixture();v=summarize(rs,ts);self.assertEqual((v['success_rate_pct'],v['pass_at_4_pct'],v['mixed_tasks'],v['all_failure_tasks'],v['mean_response_tokens']),(27.5,50,30,50,13.5))
 def test_duplicate_lineage_rejected(self):
  ts,rs=self.fixture();rs[1]['lineage']='0'
  with self.assertRaises(AssertionError):summarize(rs,ts)
 def test_changed_frozen_task_rejected(self):
  ts,rs=self.fixture();rs[0]['task']=dict(rs[0]['task'],map='changed')
  with self.assertRaises(AssertionError):summarize(rs,ts)
 def test_missing_trajectory_rejected(self):
  ts,rs=self.fixture()
  with self.assertRaises(AssertionError):summarize(rs[:-1],ts)
if __name__=='__main__':unittest.main()
