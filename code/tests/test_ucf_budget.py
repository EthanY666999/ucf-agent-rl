import sys,unittest,tempfile
from pathlib import Path
from types import SimpleNamespace as NS
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'ucf'))
from ucf_select import calibrate,select_all,RHO_MIN
class BudgetTests(unittest.TestCase):
 def test_capacity_saturates_positive_only(self):
  ws=[[.1,0],[.3]];c=calibrate(ws)
  mean=sum(min(1.,max(RHO_MIN,c*w)) for g in ws for w in g)/2
  self.assertAlmostEqual(mean,1.015)
 def test_empty_and_zero_support(self):
  self.assertEqual(calibrate([[],[]]),0.)
  self.assertEqual(calibrate([[0],[0]]),0.)
 def test_excluded_has_zero_probability(self):
  first=[NS(task_id=0,lineage=str(i),reward=0.,terminal='hole') for i in range(4)]
  n=dict(branch=first[0],j=0,turn=1,S=.9,H_cons=.9,stratum=dict(parse_mismatch=True))
  with tempfile.TemporaryDirectory() as d:
   picked,rows=select_all({0:[n]},first,[{'uid':'fixture'}],100,42,1,Path(d)/'select.jsonl')
  self.assertEqual(rows[0]['rho'],0.);self.assertFalse(rows[0]['selected']);self.assertEqual(picked,{0:[]})
