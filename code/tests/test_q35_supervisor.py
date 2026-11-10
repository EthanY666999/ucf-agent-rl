import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.q35_p0_supervisor import execute,manual
from scripts.q35_run_receipt import verify,summarize


class SupervisorTest(unittest.TestCase):
    def test_one_checkpoint_recovery_and_no_duplicate_completed_run(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'run';script=root/'child.py'
            script.write_text("import sys,json\nfrom pathlib import Path\np=Path(sys.argv[1]);p.mkdir(exist_ok=True)\nwith (p/'calls').open('a') as f:f.write('call\\n')\n(p/'checkpoint.pt').write_text('checkpoint')\n(p/'status.json').write_text(json.dumps(dict(status='complete' if '--resume' in sys.argv else 'failed')))\nraise SystemExit(0 if '--resume' in sys.argv else 1)\n")
            with (root/'lock').open('a') as lock,patch('scripts.q35_p0_supervisor.ROOT',root),patch('scripts.q35_p0_supervisor.STATE',root):
                command=[sys.executable,'-u',str(script),str(out)]
                self.assertTrue(execute(command,out,'complete',lock))
                self.assertTrue(execute(command,out,'complete',lock))
            self.assertEqual((out/'calls').read_text().splitlines(),['call','call'])
            self.assertEqual(len(json.loads((root/'run_attempts.json').read_text())),2)

    def test_second_failure_preserved_and_not_relaunched(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'run';script=root/'child.py'
            script.write_text("import sys\nfrom pathlib import Path\np=Path(sys.argv[1]);p.mkdir(exist_ok=True);(p/'checkpoint.pt').write_text('checkpoint')\nraise SystemExit(1)\n")
            with (root/'lock').open('a') as lock,patch('scripts.q35_p0_supervisor.ROOT',root),patch('scripts.q35_p0_supervisor.STATE',root):
                command=[sys.executable,'-u',str(script),str(out)]
                self.assertFalse(execute(command,out,'complete',lock))
                self.assertFalse(execute(command,out,'complete',lock))
                manual(out,'second failure',preserve=True)
            record=json.loads((out/'needs_manual.json').read_text())
            self.assertEqual((Path(record['preserved_path'])/'checkpoint.pt').read_text(),'checkpoint')
            self.assertEqual(len(json.loads((root/'run_attempts.json').read_text())),2)


class ReceiptTest(unittest.TestCase):
    def fixture(self,root):
        import numpy as np
        from safetensors.numpy import save_file
        code=Path(__file__).resolve().parents[1]
        tasks=json.loads((code/'eval_sets/supplemental500/tasks.json').read_text())
        records=[dict(task=t,reward=j%2,lineage=str(j)) for t in tasks for j in range(4)]
        def write(name,value):
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
        def lines(name,values):
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(v)+'\n' for v in values))
        write('status.json',dict(status='complete'))
        write('config.json',dict(max_turns=10,is_slippery=False,eval_sets_dir=str(code/'eval_sets')))
        lines('train_metrics.jsonl',[dict(step=i,grad_norm=1.,kl=.01,loss=.1) for i in range(1,401)])
        lines('final_test.jsonl',records[:400]);lines('map8.jsonl',records[:400])
        for i in range(5):lines(f'supplemental500/batch{i+1}.jsonl',records[i*400:(i+1)*400])
        write('supplemental500/summary.json',summarize(records,500))
        write('filtered409.json',dict(success_rate=.5,pass_at_4=1.))
        for step in range(0,401,25):write(f'validation_{step:04d}.json',dict(success_rate_pct=50.,pass_at_4_pct=100.))
        write('final_model/config.json',{});write('final_model/tokenizer_config.json',{})
        save_file(dict(weight=np.zeros((2,2),dtype=np.float32)),str(root/'final_model/model.safetensors'))
        (root/'checkpoint.pt').write_text('retained until verification')

    def test_verified_checkpoint_pruned_but_records_and_final_model_remain(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root);r=verify(root,prune_checkpoint=True)
            self.assertTrue(r['passed']);self.assertFalse((root/'checkpoint.pt').exists())
            self.assertTrue((root/'final_model/model.safetensors').exists())
            self.assertTrue((root/'train_metrics.jsonl').exists())

    def test_incomplete_evaluation_never_prunes_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            (root/'supplemental500/batch5.jsonl').unlink()
            with self.assertRaises(FileNotFoundError):verify(root,prune_checkpoint=True)
            self.assertTrue((root/'checkpoint.pt').exists())
