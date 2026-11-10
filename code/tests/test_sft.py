import os
import unittest
from sft import encode_trace

@unittest.skipUnless(os.environ.get('SFT_TEST_MODEL'), 'Set SFT_TEST_MODEL for the required model-specific tokenizer audit')
class MaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.t = AutoTokenizer.from_pretrained(os.environ['SFT_TEST_MODEL'], local_files_only=True)

    def test_mask_and_causal_prefix(self):
        m=[dict(role='system',content='Follow map.'),dict(role='user',content='State one'),
           dict(role='assistant',content='```Down```'),dict(role='user',content='State two'),
           dict(role='assistant',content='```Right```')]
        ids,mask=encode_trace(self.t,m)
        target=[i for i,k in zip(ids,mask) if k]
        self.assertEqual(self.t.decode(target),'```Down```'+self.t.eos_token+'```Right```'+self.t.eos_token)
        first_ids,first_mask=encode_trace(self.t,m[:3])
        self.assertEqual(ids[:len(first_ids)],first_ids)
        self.assertEqual(mask[:len(first_mask)],first_mask)
        m[3]['content']='A different FUTURE observation'
        changed,_=encode_trace(self.t,m)
        self.assertEqual(changed[:len(first_ids)],first_ids)

    def test_reject_non_action(self):
        with self.assertRaises(ValueError):
            encode_trace(self.t,[dict(role='user',content='map'),dict(role='assistant',content='maybe down')])
