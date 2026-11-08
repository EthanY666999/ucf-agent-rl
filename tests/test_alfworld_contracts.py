import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from alfworld_adapter import parse_action, render_prompt
from alfworld_adapter.actions import (validate_action_tokens, action_distribution,
                                     force_action, require_ucf_acceptance, score_actions, first_fence_prefix)
from alfworld_adapter.sft_data import token_ids

class ALFWorldContractTests(unittest.TestCase):
    def test_mapping_is_not_iterated_as_token_ids(self):
        self.assertEqual(token_ids({'input_ids':[1,2],'attention_mask':[1,1]}),[1,2])
        with self.assertRaises(ValueError):token_ids(['input_ids','attention_mask'])
    def test_first_fence_not_future_last_action(self):
        self.assertEqual(parse_action('reason ```look``` then ```take apple 1```'), 'look')
        self.assertIsNone(parse_action('reason ```look'))
        one=first_fence_prefix([100],[7,8,99,10,99],99)
        two=first_fence_prefix([100],[7,8,99,66,33,99,99],99)
        self.assertEqual(one,two)
        self.assertEqual(one,([100,7,8,99],3))
        self.assertIsNone(first_fence_prefix([100],[7,8,10],99))

    def test_two_turn_window_and_actions(self):
        history=[dict(observation='obs'+str(i),action='action'+str(i)) for i in range(4)]
        prompt=render_prompt('current',['look','go to fridge 1'],history,'task')
        self.assertNotIn('obs1',prompt)
        self.assertIn('obs2',prompt)
        self.assertIn('obs3',prompt)
        self.assertIn('go to fridge 1',prompt)

    def test_reject_overlapping_string_events(self):
        with self.assertRaises(ValueError):validate_action_tokens([[1,2],[1,2,3]])
        with self.assertRaises(ValueError):validate_action_tokens([[1,2],[1,2]])
        validate_action_tokens([[1,2,9],[1,2,3,9]])

    def test_unrestricted_probabilities_and_residual(self):
        actions=['look','go to fridge 1']
        logps=[[math.log(.5),math.log(.5)],[math.log(.2),math.log(.5)]]
        p,other=action_distribution(actions,logps)
        self.assertAlmostEqual(p['look'],.25)
        self.assertAlmostEqual(p['go to fridge 1'],.1)
        self.assertAlmostEqual(other,.65)
        chosen,ids,old=force_action(actions,[[1,9],[2,9]],logps,p,{'go to fridge 1'},42)
        self.assertEqual(chosen,'go to fridge 1')
        self.assertEqual(old,logps[1])
        self.assertEqual(ids,[2,9])

    def test_prompt_scoring_uses_full_actual_suffix(self):
        class Engine:
            def generate(self,jobs,params,use_tqdm):
                assert params.prompt_logprobs==1 and params.max_tokens==1
                result=[]
                for job in jobs:
                    ids=job['prompt_token_ids']
                    assert ids[:2]==[100,200]
                    result.append(SimpleNamespace(prompt_logprobs=[None]+[
                        {token:SimpleNamespace(logprob=-i)} for i,token in enumerate(ids[1:],1)]))
                return result
        with patch.dict(sys.modules,{'vllm':SimpleNamespace(SamplingParams=lambda **kw:SimpleNamespace(**kw))}):
            rows=score_actions(Engine(),[100,200],[[1,2,9],[1,3,4,9]])
        self.assertEqual(rows,[[-2,-3,-4],[-2,-3,-4,-5]])

    def test_reject_invalid_logprob(self):
        for row in [[],[float('nan')],[1.0]]:
            with self.assertRaises(ValueError):action_distribution(['look'],[row])

    def test_gate_without_gpu_acceptance(self):
        with self.assertRaises(RuntimeError):require_ucf_acceptance({})

if __name__=='__main__':unittest.main()
