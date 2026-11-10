import random
import tempfile
import unittest
from pathlib import Path
import torch
from scripts.sft_checkpoint import save,restore


class SFTRecoveryTest(unittest.TestCase):
    def test_resume_matches_uninterrupted_adamw_and_rng(self):
        torch.manual_seed(42);random.seed(42)
        model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.2),torch.nn.Linear(4,1))
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=0.)
        def step(m,o):
            o.zero_grad();x=torch.rand(2,3);m(x).square().sum().backward();o.step()
        step(model,optimizer)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'checkpoint.pt';cfg=dict(seed=42,epochs=1)
            save(path,model,optimizer,1,cfg,12.)
            step(model,optimizer);expected={k:v.detach().clone() for k,v in model.state_dict().items()}
            expected_rng=random.random()
            restored=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.2),torch.nn.Linear(4,1))
            opt=torch.optim.AdamW(restored.parameters(),lr=2e-5,weight_decay=0.)
            self.assertEqual(restore(path,restored,opt,cfg),(1,12.))
            step(restored,opt)
            self.assertEqual(random.random(),expected_rng)
            for name,v in restored.state_dict().items():self.assertTrue(torch.equal(v,expected[name]),name)
            with self.assertRaises(AssertionError):restore(path,restored,opt,dict(seed=43,epochs=1))
