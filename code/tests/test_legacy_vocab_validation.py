import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from scripts.legacy_vocab_validation import install


class VocabValidationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.events = Path(self.tmp.name)/'events.jsonl'
        self.backend = SimpleNamespace(model_config=SimpleNamespace(is_multimodal_model=False),tokenizer=SimpleNamespace(get_lora_tokenizer=lambda _: SimpleNamespace(max_token_id=151664)))
        self.original_tokenizer = self.backend.tokenizer
        def validate(inputs, lora, *, prompt_type):
            ids=inputs['prompt_token_ids']
            if not ids: raise ValueError('empty')
            if max(ids)>self.backend.tokenizer.get_lora_tokenizer(lora).max_token_id:
                raise ValueError(f'Token id {max(ids)} is out of vocabulary')
            if len(ids)>3: raise ValueError('too long')
            return tuple(ids)
        self.backend._validate_model_input=validate
        install(SimpleNamespace(llm_engine=self.backend),151936,self.events)

    def call(self,ids):
        return self.backend._validate_model_input({'prompt_token_ids':ids},None,prompt_type='decoder')

    def test_model_ids_preserved_and_original_tokenizer_restored(self):
        self.assertEqual(self.call([42,151881]),(42,151881))
        self.assertIs(self.backend.tokenizer,self.original_tokenizer)
        self.assertIn('151881',self.events.read_text())

    def test_true_oov_and_other_checks_still_rejected(self):
        for ids in [[151936],[151881]*4,[]]:
            with self.assertRaises(ValueError):self.call(ids)
            self.assertIs(self.backend.tokenizer,self.original_tokenizer)
        self.assertFalse(self.events.exists())

    def test_named_ids_keep_original_path(self):
        self.assertEqual(self.call([42,151645]),(42,151645))
        self.assertFalse(self.events.exists())
