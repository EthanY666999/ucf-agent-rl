import json
import unittest
from pathlib import Path
from scripts.q35_p0_plan import compile_plan,SFT


class P0PlanTest(unittest.TestCase):
    def test_order_origins_and_no_unregistered_changes(self):
        base=json.loads((Path(__file__).resolve().parents[1]/'configs/frozenlake_q35_2b.json').read_text())
        for p95,cap in [(2048,2048),(2048.01,4096)]:
            plan=compile_plan(base,dict(passed=True,steps=5,trajectory_length_p95=p95,nosft_max_response_tokens=cap))
            self.assertEqual([r['id'] for r in plan['entries']],[f'F{i:02}' for i in range(3,16)])
            for r in plan['entries']:
                c=r['config'];sft=r['id'] not in ['F09','F10']
                self.assertEqual(c['model'],str(SFT) if sft else base['model'])
                self.assertEqual(c['max_response_tokens'],10240 if sft else cap)
                self.assertEqual(c['steps'],400);self.assertEqual(c['eval_every'],25)
                self.assertEqual(c['flat_group_size'],4)
                self.assertEqual(c['seed'],42 if int(r['id'][1:])<11 else 43)
                for k in ['learning_rate','train_batch_size','mini_batch_tasks','initial_branches','branch_factor','top_k_branches','temperature','max_turns']:
                    self.assertEqual(c[k],base[k],(r['id'],k))
                if r['method']=='tree_random':self.assertEqual(c['branch_interval'],2)
                if r['method']=='ucf':self.assertEqual(c['ucf_beta'],.25)
                if r['method']=='starpo_s':self.assertEqual(c['starpo_keep_fraction'],.5)
            self.assertIn(base['model'],plan['entries'][0]['training_command'])
            self.assertFalse(plan['launches_processes'])
