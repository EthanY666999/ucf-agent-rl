import copy
import unittest
from scripts.crosscard_receipt import compare


class CrosscardReceiptTest(unittest.TestCase):
    def sample(self):
        pairs=[dict(step=i,**{k:dict(current=v,reference=v,ratio=1.) for k,v in [('kl',.01),('grad_norm',1.),('success_rate',.1)]}) for i in range(1,11)]
        records=[dict(step=i,kl=.01,grad_norm=1.,success_rate=.1,max_parameter_change=1e-6) for i in range(1,11)]
        return pairs,records
    def test_isolated_small_kl_retained_as_diagnostic(self):
        pairs,records=self.sample();pairs[0]['kl'].update(current=.0004,ratio=.04);records[0]['kl']=.0004
        result=compare(pairs,records);self.assertTrue(result['passed']);self.assertEqual(len(result['per_step_exceptions']),1)
    def test_persistent_drift_fails(self):
        pairs,records=self.sample()
        for p in pairs:p['kl'].update(current=.2,ratio=20.)
        self.assertFalse(compare(pairs,records)['passed'])
    def test_incomplete_or_nonfinite_or_no_update_never_passes(self):
        pairs,records=self.sample()
        with self.assertRaises(AssertionError):compare(pairs[:-1],records)
        for key,value in [('grad_norm',float('nan')),('max_parameter_change',0.)]:
            changed=copy.deepcopy(records);changed[0][key]=value
            with self.assertRaises(AssertionError):compare(pairs,changed)
