"""Step-independent supervised examples, matching history window 2.

These demonstrations are expert ACTION traces, not fabricated teacher reasoning.
Training-token budget is reported in turns/tokens, not 1000 single-step samples.
"""
from collections.abc import Mapping

def token_ids(encoded):
    if isinstance(encoded,Mapping):
        encoded=encoded['input_ids']
    result=list(encoded)
    if not result or any(not isinstance(x,int) for x in result):
        raise ValueError('Expected a nonempty flat list of integer token ids')
    return result

def encode_example(tokenizer, messages, max_tokens=16384):
    if len(messages)!=2 or [m['role'] for m in messages]!=['user','assistant']:
        raise ValueError('Expected one step-independent user/assistant pair')
    options=dict(tokenize=True,enable_thinking=False)
    prefix=tokenizer.apply_chat_template(messages[:1],add_generation_prompt=True,**options)
    full=tokenizer.apply_chat_template(messages,add_generation_prompt=False,**options)
    prefix=token_ids(prefix);full=token_ids(full)
    if full[:len(prefix)]!=prefix:
        raise ValueError('Chat template changed the teacher prefix; do not guess the mask')
    if len(full)>max_tokens:
        raise ValueError('Example exceeds maximum tokens; no silent truncation')
    if len(full)<=len(prefix):
        raise ValueError('Example has no supervised response tokens')
    return {'input_ids':full,'labels':[-100]*len(prefix)+full[len(prefix):],
            'attention_mask':[1]*len(full),'response_start':len(prefix),
            'loss_tokens':len(full)-len(prefix)}
