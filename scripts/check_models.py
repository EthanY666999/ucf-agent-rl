"""CPU/tokenizer and shape-only checks: no checkpoint files or GPU allocation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PARSER_DIAGNOSTIC_ACTIONS = ("Left", "Down", "Right", "Up", "left", "down", "right", "up")


def audit(root: Path):
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/ucf"))
    from sft import encode_trace
    from cert_entropy import token_map
    from tokens import fence_id

    assert transformers.__version__ == '5.22.2', 'CPU acceptance requires the pinned transformers 5.22.2'
    torch.set_num_threads(1)
    manifest = json.loads((root / "manifest.json").read_text())
    results = {"schema": 1, "device": "meta_shapes_and_cpu_tokenizer", "weights_downloaded": False,
               "torch": torch.__version__, "transformers": transformers.__version__, "models": {}}
    token_contracts = []
    for model, entry in manifest["models"].items():
        folder = root / model.split("/")[-1]
        for name, item in entry["files"].items():
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == item["sha256"], (model, name)
        tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        fence = tokenizer.encode("```", add_special_tokens=False)
        assert len(fence) == 1, (model, fence)
        assert fence_id(tokenizer) == fence[0]
        actual_mapping, actual_examples = token_map(tokenizer)
        assert len(actual_mapping) == 8
        actual_ids = {}
        for text, example in actual_examples.items():
            assert example['position'] == 1
            assert example['ids'] == [fence[0], example['first'], fence[0]], (text, example)
            actual_ids[text[3:-3]] = example['first']
        tokens = {a: tokenizer.encode(a, add_special_tokens=False) for a in PARSER_DIAGNOSTIC_ACTIONS}
        assert all(len(ids) == 1 for ids in tokens.values())
        for a in PARSER_DIAGNOSTIC_ACTIONS:
            assert tokenizer.encode("```" + a + "```", add_special_tokens=False) == fence + tokens[a] + fence
        conversation = [{"role": "user", "content": "Choose an action."},
                        {"role": "assistant", "content": "<think>\n\n</think>\n\n```Left```"},
                        {"role": "user", "content": "Now choose the next action."}]
        rendered = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert "```Left```" in rendered and "<think>" in rendered and "</think>" in rendered
        # Official re-rendering strips historical reasoning blocks; production
        # therefore preserves the sampled token prefix and appends turn_suffix.
        messages = [{"role": "user", "content": "Map"}, {"role": "assistant", "content": "```Down```"},
                    {"role": "user", "content": "Next"}, {"role": "assistant", "content": "```Right```"}]
        ids, mask = encode_trace(tokenizer, messages)
        first_ids, first_mask = encode_trace(tokenizer, messages[:2])
        assert ids[:len(first_ids)] == first_ids and mask[:len(first_mask)] == first_mask
        assert tokenizer.decode(ids).count("<think>\n\n</think>\n\n") == 2
        assert tokenizer.decode([t for t, active in zip(ids, mask) if active]) == "```Down```" + tokenizer.eos_token + "```Right```" + tokenizer.eos_token
        thinking = tokenizer.apply_chat_template(conversation[:1], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        default_prompt = tokenizer.apply_chat_template(conversation[:1], tokenize=False, add_generation_prompt=True)
        # Config shape construction allocates no model storage.
        config = AutoConfig.from_pretrained(folder, local_files_only=True)
        text = config.get_text_config()
        with torch.device("meta"):
            actor = AutoModelForCausalLM.from_config(text, torch_dtype=torch.float32)
        assert all(p.device.type == "meta" for p in actor.parameters())
        count = sum(p.numel() for p in actor.parameters())
        index = json.loads((folder / "model.safetensors.index.json").read_text())
        total_storage = index["metadata"]["total_size"]
        report = {"revision": entry["revision"], "fence_id": fence[0],
                  "actual_ucf_action_ids": actual_ids,
                  "actual_ucf_token_to_environment_action": actual_mapping,
                  "parser_diagnostic_action_ids": {k: v[0] for k, v in tokens.items()},
                  "vocab_size": len(tokenizer), "eos_token_id": tokenizer.eos_token_id,
                  "tie_word_embeddings": text.tie_word_embeddings, "text_parameter_count": count,
                  "original_checkpoint_total_bytes": total_storage,
                  "historical_empty_think_preserved": True,
                  "official_rerender_historical_think_blocks": rendered.count("<think>") - 1,
                  "default_thinking_enabled": default_prompt.endswith("<think>\n"),
                  "nonthinking_template_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                  "thinking_template_sha256": hashlib.sha256(thinking.encode()).hexdigest(),
                  "memory_gib": {"actor_fp32": count * 4 / 2**30, "grad_fp32": count * 4 / 2**30,
                                 "adamw_m_v_fp32": count * 8 / 2**30, "reference_bf16": count * 2 / 2**30,
                                 "vllm_text_bf16": count * 2 / 2**30, "static_text_only_total": count * 20 / 2**30,
                                 "checkpoint_fp32_actor_plus_adam": count * 12 / 2**30}}
        results["models"][model] = report
        token_contracts.append((report["fence_id"], report["actual_ucf_action_ids"], report["parser_diagnostic_action_ids"], report["vocab_size"], report["eos_token_id"],
                                report["nonthinking_template_sha256"], report["thinking_template_sha256"]))
        del actor
    assert all(contract == token_contracts[0] for contract in token_contracts), "Family tokenizer/template mismatch"
    results["family_contract_identical"] = True
    results["contract_scope"] = "tokenizer core plus explicit enable_thinking=False/True fixtures; NOT implicit template defaults"
    results["tokenizer_core_files_identical"] = len({e['files']['tokenizer.json']['sha256'] for e in manifest['models'].values()}) == 1
    assert results["tokenizer_core_files_identical"]
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path(__file__).resolve().parents[1] / "data/model_metadata")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.metadata)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)
