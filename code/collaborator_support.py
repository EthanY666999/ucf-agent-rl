"""CPU-only release helpers; no changes to the rollout or learning objective."""
import json
from pathlib import Path


def validate_model_dir(path):
    """Accept a complete local safetensors model, including indexed shards."""
    path = Path(path)
    if not (path / 'config.json').is_file():
        raise ValueError(f'Missing model config: {path}')
    index = path / 'model.safetensors.index.json'
    if index.is_file():
        weights = json.loads(index.read_text()).get('weight_map', {})
        if not weights:
            raise ValueError(f'Empty weight index: {index}')
        for name in set(weights.values()):
            shard = path / name
            if shard.resolve().parent != path.resolve() or not shard.is_file() or shard.stat().st_size == 0:
                raise ValueError(f'Missing or invalid weight shard: {name}')
    elif not (path / 'model.safetensors').is_file() or (path / 'model.safetensors').stat().st_size == 0:
        raise ValueError(f'Missing safetensors weights: {path}')


def evaluation_runtime(cfg, config_path, root, runtime_keys):
    """Keep the archived evaluation protocol; copy only engineering settings."""
    cfg = dict(cfg)
    if config_path:
        supplied = json.loads(Path(config_path).read_text())
        cfg.update({k: supplied[k] for k in runtime_keys if k in supplied})
    override = Path(root) / 'artifacts/runtime_override.json'
    if override.exists():
        values = json.loads(override.read_text())
        if not set(values) <= runtime_keys:
            raise ValueError('Only engineering runtime overrides are allowed')
        cfg.update(values)
    return cfg
