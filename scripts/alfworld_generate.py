#!/usr/bin/env python3
"""Generate train-only hand-coded expert episodes, then replay in fresh envs.

No LLM, model weights, GPU, or THOR visual assets are involved.
"""
from __future__ import annotations
import argparse
import concurrent.futures
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from alfworld_adapter import ALFWorldEpisode

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def generate_one(args):
    gamefile, root, seed = args
    env = None
    try:
        env = ALFWorldEpisode(gamefile, seed)
        initial = env.observation
        steps = []
        while not env.done:
            observation = env.observation
            admissible = env.admissible
            prompt = env.prompt()
            action = env.expert_action()
            after, reward, done = env.step(action)
            steps.append(dict(observation=observation, admissible_actions=admissible,
                              prompt=prompt, action=action, response="```" + action + "```",
                              next_observation=after, reward=reward, done=done))
        if not env.won:
            return {"failure": "expert_not_solved_within_50", "game": str(Path(gamefile).relative_to(root))}
        env.close(); env = ALFWorldEpisode(gamefile, seed)
        assert env.observation == initial
        for step in steps:
            assert env.observation == step["observation"]
            assert env.admissible == step["admissible_actions"]
            after, reward, done = env.step(step["action"])
            assert (after, reward, done) == (step["next_observation"], step["reward"], step["done"])
        assert env.won
        key = str(Path(gamefile).relative_to(root))
        return dict(id=hashlib.sha256(key.encode()).hexdigest()[:24], game=key,
                    game_sha256=hashlib.sha256(Path(gamefile).read_bytes()).hexdigest(),
                    split="train", seed=seed, teacher="ALFWorld HandCodedTWAgent",
                    synthetic_reasoning=False, initial_observation=initial, steps=steps,
                    independently_replayed=True, replay_terminal_success=True)
    except Exception as exc:
        return {"failure": type(exc).__name__ + ": " + str(exc), "game": str(Path(gamefile).relative_to(root))}
    finally:
        if env is not None:
            env.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if not 1 <= args.workers <= 4:
        p.error("CPU release generation allows 1 to 4 workers")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    manifest = json.loads(args.manifest.read_text())
    games = list(manifest["splits"]["train"])
    random.Random(args.seed).shuffle(games)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / 'golden_traces.jsonl.gz').exists():
        raise RuntimeError('Refusing to overwrite an existing Golden Trace file; use a new output directory')
    rows, failures = [], []
    started = time.time()
    # Bounded task submissions: avoid generating the entire train set merely
    # because we requested 1000 successful demonstrations.
    with concurrent.futures.ProcessPoolExecutor(args.workers) as executor:
        for offset in range(0, len(games), 16):
            jobs = [(str(args.games_root / x), str(args.games_root), args.seed) for x in games[offset:offset+16]]
            for result in executor.map(generate_one, jobs):
                if "failure" in result:
                    failures.append(result)
                elif len(rows) < args.count:
                    rows.append(result)
            progress = dict(accepted=len(rows), target=args.count, attempted=min(offset+16, len(games)),
                            failures=len(failures), elapsed_seconds=time.time()-started,
                            verification="fresh environment, exact observation/action/reward replay")
            (args.output_dir / "generation_progress.json").write_text(json.dumps(progress, indent=2) + "\n")
            print(json.dumps(progress), flush=True)
            if len(rows) >= args.count:
                break
    if len(rows) != args.count:
        raise RuntimeError(f"Only {len(rows)} verified episodes; target {args.count}")
    for filename, values in [("golden_traces.jsonl.gz", rows)]:
        with (args.output_dir / filename).open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as output:
                for row in values:
                    output.write((canonical(row) + "\n").encode())
    sft = []
    for row in rows:
        for i, step in enumerate(row["steps"]):
            sft.append(dict(episode_id=row["id"], turn=i, split="train", messages=[
                {"role":"user", "content":step["prompt"]},
                {"role":"assistant", "content":step["response"]}]))
    with (args.output_dir / "sft_messages.jsonl.gz").open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as output:
            for row in sft:output.write((canonical(row)+"\n").encode())
    summary = dict(progress, completed=True, sft_examples=len(sft),
                   successful_not_retained_after_target=progress['attempted']-len(failures)-len(rows),
                   failure_counts=dict(Counter(x['failure'] for x in failures)),
                   task_types={t:sum('/'+t+'-' in x['game'] for x in rows) for t in manifest['task_types']},
                   episode_steps={"min":min(len(x['steps']) for x in rows),
                                  "max":max(len(x['steps']) for x in rows),
                                  "mean":sum(len(x['steps']) for x in rows)/len(rows)},
                   files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                          for p in args.output_dir.glob('*.gz')})
    (args.output_dir / "verification.json").write_text(json.dumps(summary, indent=2)+"\n")
    (args.output_dir / "generation_progress.json").unlink()

if __name__ == "__main__":
    main()
