"""Accept generated model-vocabulary IDs in legacy vLLM text-only inputs.

Qwen2.5 has 151936 embedding/logit rows but a smaller named tokenizer
vocabulary. vLLM 0.9 can sample padded rows, then reject its own output on
the next turn. Keep IDs, decoding, logits and sampling probabilities intact.
Only the legacy input range check uses the model's actual embedding range.
"""
import json
import time
from types import SimpleNamespace


def install(engine, model_vocab_size, event_path):
    backend = engine.llm_engine
    assert not backend.model_config.is_multimodal_model
    original = backend._validate_model_input

    def validate(inputs, lora_request, *, prompt_type):
        try:
            return original(inputs, lora_request, prompt_type=prompt_type)
        except ValueError as exc:
            ids = inputs.get('prompt_token_ids', [])
            if (not str(exc).startswith('Token id ') or
                    not str(exc).endswith(' is out of vocabulary') or
                    not ids or min(ids) < 0 or max(ids) >= model_vocab_size):
                raise
            group = backend.tokenizer
            named_max = group.get_lora_tokenizer(lora_request).max_token_id
            # Re-run ALL original checks; restore the original tokenizer even
            # when another check (e.g. context length) fails. Enqueue is serial.
            backend.tokenizer = SimpleNamespace(get_lora_tokenizer=lambda _: SimpleNamespace(max_token_id=model_vocab_size-1))
            try:
                result = original(inputs, lora_request, prompt_type=prompt_type)
            finally:
                backend.tokenizer = group
            with event_path.open('a') as f:
                f.write(json.dumps(dict(unix=time.time(),token_ids=sorted({i for i in ids if i>named_max}),model_vocab_size=model_vocab_size,distribution_changed=False))+'\n')
            return result

    backend._validate_model_input = validate
