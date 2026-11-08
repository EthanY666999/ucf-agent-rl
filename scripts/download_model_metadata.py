"""Download an explicit metadata allowlist; this command never downloads weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

MODELS = ["Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"]
FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "model.safetensors.index.json", "LICENSE")


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "ucf-agent-rl-metadata/1"}), timeout=120) as response:
        return response.read()


def download(destination: Path, models=MODELS):
    destination.mkdir(parents=True, exist_ok=True)
    old = destination / "manifest.json"
    manifest = json.loads(old.read_text()) if old.exists() else {"schema": 1, "weights_downloaded": False, "models": {}}
    for model in models:
        # Once resolved, retain the exact revision on reruns.
        entry = manifest["models"].get(model, {})
        revision = entry.get("revision") or json.loads(fetch(f"https://huggingface.co/api/models/{model}"))["sha"]
        folder = destination / model.split("/")[-1]
        folder.mkdir(exist_ok=True)
        files = {}
        for name in FILES:
            target = folder / name
            prior = entry.get("files", {}).get(name, {})
            if not (target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == prior.get("sha256")):
                payload = fetch(f"https://huggingface.co/{model}/resolve/{revision}/{name}")
                target.write_bytes(payload)
            data = target.read_bytes()
            files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        manifest["models"][model] = {"revision": revision, "files": files, "source": f"https://huggingface.co/{model}/tree/{revision}"}
        old.write_text(json.dumps(manifest, indent=2) + "\n")
        print(model, revision, sum(f["bytes"] for f in files.values()), "metadata bytes")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data/model_metadata")
    args = parser.parse_args()
    download(args.output)
