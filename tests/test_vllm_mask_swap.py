import ast,copy,json,types
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent/'fixtures'
def load(path):
 tree=ast.parse(path.read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='swap_states');m=ast.Module(body=[f],type_ignores=[]);ns={'MoveDirectionality':types.SimpleNamespace(SWAP='swap'),'swap_dict_values':lambda d,a,b:None};exec(compile(ast.fix_missing_locations(m),str(path),'exec'),ns);return ns['swap_states']
def obj(mask):
 x=types.SimpleNamespace(_req_ids=['a','b','c'],req_output_token_ids=[[],[],[]],spec_token_ids=[[],[],[]],req_id_to_index={'a':0,'b':1,'c':2},_get_active_token_count=lambda i:2,token_ids_cpu=np.arange(12).reshape(3,4),is_token_ids=np.ones((3,4),bool),req_prompt_embeds={},block_table=types.SimpleNamespace(swap_row=lambda a,b:None),is_pooling_model=False,batch_update_builder=types.SimpleNamespace(moved=[]),generators={},bad_words_token_ids={},allowed_token_ids_mask_cpu_tensor=mask)
 for name in ['num_tokens_no_spec','num_prompt_tokens','num_computed_tokens_cpu','request_lora_mapping','temperature_cpu','top_p_cpu','top_k_cpu','frequency_penalties_cpu','presence_penalties_cpu','repetition_penalties_cpu','num_accepted_tokens_cpu']:setattr(x,name,np.arange(3).copy())
 return x
mask=torch.tensor([[False,True,True,True],[True,False,True,True],[True,True,False,True]])
x=obj(mask.clone());load(P/'vllm_swap_before.py')(x,0,1);assert not torch.equal(x.allowed_token_ids_mask_cpu_tensor,mask[[1,0,2]])
f=load(P/'vllm_swap_after.py');cases=0
for schedule in [[(0,1)],[(0,2),(1,2),(0,1)],[(0,1),(0,1)],[(1,1)],[(2,0)]*50]:
 x=obj(mask.clone())
 for a,b in schedule:
  f(x,a,b);expected=torch.stack([mask['abc'.index(r)] for r in x._req_ids]);assert torch.equal(x.allowed_token_ids_mask_cpu_tensor,expected);cases+=1
x=obj(None);f(x,0,1);assert x._req_ids==['b','a','c'] and x.allowed_token_ids_mask_cpu_tensor is None
print(json.dumps({'original_bug_reproduced':True,'patched_actual_method_swap_checks':cases,'no_mask_case_passed':True,'GPU_used':False}))

def test_actual_extracted_method_regression():
 assert cases == 57
