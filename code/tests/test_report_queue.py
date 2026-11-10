import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from q35_report_queue import diagnostics

class ReportQueueTest(unittest.TestCase):
    def test_committed_latest_rows_and_tail_diagnostics(self):
        rows=[dict(step=1,kl=20,ucf_preupdate_fullbatch_components=dict(B=dict(grad_norm=100))),dict(step=1,kl=.1,ucf_preupdate_fullbatch_components=dict(B=dict(grad_norm=1))),dict(step=2,kl=11,ucf_preupdate_fullbatch_components=dict(B=dict(grad_norm=5)))]
        r=diagnostics(rows)
        self.assertEqual(r['steps'],2);self.assertEqual(r['B_grad_norm']['median'],3.)
        self.assertEqual(r['B_grad_norm']['maximum'],5);self.assertEqual(r['B_grad_norm']['steps_above3'],1)
        self.assertEqual(r['steps_KL_above10'],1)
    def test_baseline_has_no_fabricated_B_gradient(self):
        self.assertIsNone(diagnostics([dict(step=1,kl=.01)])['B_grad_norm'])
