import gzip
import json
from pathlib import Path
import tempfile
import unittest
from collaborator_support import validate_model_dir, evaluation_runtime
from scripts.prepare_collaborator import configs, write_configs
from scripts.signal_stats import summarize


class CollaboratorTests(unittest.TestCase):
    def test_shards_must_really_exist(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); (p/'config.json').write_text('{}')
            with self.assertRaises(ValueError): validate_model_dir(p)
            (p/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': 'a.safetensors', 'b': 'b.safetensors'}}))
            (p/'a.safetensors').write_bytes(b'placeholder')
            with self.assertRaises(ValueError): validate_model_dir(p)
            (p/'b.safetensors').write_bytes(b'placeholder'); validate_model_dir(p)
            (p/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': '../outside'}}))
            with self.assertRaises(ValueError): validate_model_dir(p)

    def test_runtime_never_changes_evaluation_protocol(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'run.json'
            p.write_text(json.dumps({'temperature': 99, 'eval_samples': 1, 'vllm_memory_fraction': .18, 'seed': 99}))
            before = {'temperature': 1, 'eval_samples': 4, 'seed': 42}
            after = evaluation_runtime(before, p, d, {'vllm_memory_fraction'})
            self.assertEqual({k: after[k] for k in before}, before)
            self.assertEqual(after['vllm_memory_fraction'], .18)

    def test_config_identity_and_overwrite_protection(self):
        values = configs('4b', 2)
        self.assertEqual(values['nosft_g8.json']['optimizer'], 'adamw8bit')
        self.assertEqual(values['sft_g4.json']['flat_group_size'], 4)
        self.assertEqual(values['sft_g8.json']['model'], values['sft_g4.json']['model'])
        self.assertEqual(values['sft_g8.json']['branch_interval'], 5)
        with tempfile.TemporaryDirectory() as d:
            write_configs(d, values); write_configs(d, values)
            with self.assertRaises(ValueError): write_configs(d, configs('4b', 1))
            self.assertEqual(json.loads((Path(d)/'nosft_g8.json').read_text()), values['nosft_g8.json'])

    def test_signal_uses_final_reward_not_nested_step_reward(self):
        with tempfile.TemporaryDirectory() as d:
            with gzip.open(Path(d)/'rollouts_0201.jsonl.gz', 'wt') as f:
                for uid, reward in [('a', 1), ('a', 0), ('b', 1), ('b', 1), ('c', 0)]:
                    f.write(json.dumps({'trace': [{'reward': 999}], 'task': {'uid': uid}, 'reward': reward})+'\n')
            r = summarize(d)[0]
            self.assertEqual(r['tasks'], 3); self.assertEqual(r['step'], 201)
            self.assertAlmostEqual(r['mixed'], 1/3)
            self.assertAlmostEqual(r['all_fail'], 1/3)
