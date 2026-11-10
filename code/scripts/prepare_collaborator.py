"""Write isolated 3B/4B configs without editing tracked files or existing runs."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {'3b': 'Qwen2.5-3B-Instruct', '4b': 'Qwen3-4B-Instruct-2507'}


def configs(size, tier, model=None, seed=42):
    base = json.loads((ROOT / 'configs/frozenlake_05b.json').read_text())
    base.update(seed=seed, micro_batch_size=8, micro_batch_token_limit=12000,
                vllm_memory_fraction=.22 if size == '4b' and tier == 2 else .18,
                optimizer='adamw8bit' if tier == 2 else 'adamw',
                vllm_enforce_eager=True, vllm_attention_backend='XFORMERS')
    models = {'nosft': model or f'artifacts/{MODELS[size]}', 'sft': f'runs/sft_{size}/final_model'}
    return {f'{start}_g{group}.json': dict(base, model=models[start], flat_group_size=group,
            run_label=f'{size}_{start}_g{group}_tier{tier}_seed{seed}')
            for start, group in [('nosft', 8), ('sft', 8), ('sft', 4)]}


def write_configs(destination, values):
    destination = Path(destination)
    # Validate every collision before writing any file. Identical reruns are harmless.
    for name, cfg in values.items():
        path = destination / name
        if path.exists() and json.loads(path.read_text()) != cfg:
            raise ValueError(f'Refusing to replace different config {path}; use a new --output directory')
    destination.mkdir(parents=True, exist_ok=True)
    for name, cfg in values.items():
        path = destination / name
        if not path.exists():
            with path.open('x') as f:
                f.write(json.dumps(cfg, indent=2) + '\n')
        print(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--size', choices=MODELS, required=True)
    p.add_argument('--tier', type=int, choices=[1, 2], required=True)
    p.add_argument('--model', help='Existing local model directory; absolute paths supported')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    model = str(Path(a.model).resolve()) if a.model else None
    write_configs(a.output, configs(a.size, a.tier, model, a.seed))
    print('No training started. Choose ONE experiment from the task guide.')


if __name__ == '__main__':
    main()
