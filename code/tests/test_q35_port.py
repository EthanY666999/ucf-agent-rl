"""Required tokenizer compatibility and unchanged base configuration checks."""
import json,unittest
from pathlib import Path
from transformers import AutoTokenizer
from patr_core import chat_ids,turn_suffix
from sft import encode_trace

class TokenizerPort(unittest.TestCase):
 def test_q25_exact_compatibility(self):
  t=AutoTokenizer.from_pretrained('/data/models/Qwen2.5-0.5B-Instruct',local_files_only=True)
  for text in ['State two','中文地图\nUp Down']:
   self.assertEqual(turn_suffix(t,text),'\n<|im_start|>user\n'+text+'<|im_end|>\n<|im_start|>assistant\n')
  m=[dict(role='user',content='Map'),dict(role='assistant',content='```Down```'),dict(role='user',content='Next')]
  old=t.apply_chat_template(m,tokenize=True,add_generation_prompt=True)
  old=list(old['input_ids']) if hasattr(old,'keys') else list(old)
  self.assertEqual(chat_ids(t,m),old)
  m.append(dict(role='assistant',content='```Right```'))
  ids,mask=encode_trace(t,m)
  old=t.apply_chat_template(m,tokenize=True,add_generation_prompt=False)
  old=list(old['input_ids']) if hasattr(old,'keys') else list(old)
  self.assertEqual(old,ids+[198])
  old_mask=[0]*len(old)
  for i,msg in enumerate(m):
   if msg['role']=='assistant':
    prefix=t.apply_chat_template(m[:i],tokenize=True,add_generation_prompt=True)
    prefix=list(prefix['input_ids']) if hasattr(prefix,'keys') else list(prefix)
    target=t.encode(msg['content'],add_special_tokens=False)+[t.eos_token_id]
    old_mask[len(prefix):len(prefix)+len(target)]=[1]*len(target)
  self.assertEqual(old_mask,mask+[0])
  self.assertEqual(t.decode([x for x,k in zip(ids,mask) if k]),'```Down```'+t.eos_token+'```Right```'+t.eos_token)

 def test_q35_suffix_and_mask(self):
  t=AutoTokenizer.from_pretrained('/data/models/Qwen3.5-2B',local_files_only=True)
  self.assertTrue(turn_suffix(t,'Next').endswith('<think>\n\n</think>\n\n'))
  m=[dict(role='user',content='Map'),dict(role='assistant',content='```Down```'),dict(role='user',content='Next'),dict(role='assistant',content='```Right```')]
  ids,mask=encode_trace(t,m)
  self.assertEqual(t.decode([x for x,k in zip(ids,mask) if k]),'```Down```'+t.eos_token+'```Right```'+t.eos_token)
  text=t.decode(ids)
  self.assertTrue(text.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n```Right```<|im_end|>'))
  self.assertEqual(text.count('<think>\n\n</think>\n\n'),2)
  first,first_mask=encode_trace(t,m[:2])
  self.assertEqual(ids[:len(first)],first)
  self.assertEqual(mask[:len(first)],first_mask)
  m[2]['content']='Changed future observation'
  changed,_=encode_trace(t,m)
  self.assertEqual(changed[:len(first)],first)

 def test_only_model_and_label_change(self):
  root=Path(__file__).resolve().parents[1]
  a=json.loads((root/'configs/frozenlake_05b.json').read_text());b=json.loads((root/'configs/frozenlake_q35_2b.json').read_text())
  self.assertEqual({k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)},{'model','run_label'})
