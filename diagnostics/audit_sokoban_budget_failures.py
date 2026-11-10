"""Read-only BFS diagnosis of original trajectories in completed pilot updates.

Uses recorded environment actions and checks every stored state against the
frozen production transition. This does not change reward, scoring or training.
"""
import argparse
import collections
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(code, run, through):
    sys.path.insert(0, str(code))
    from task_envs import SokobanEnv
    from sokoban_exact import distance_for_task
    cfg = json.loads((run / 'config.json').read_text())
    origin = json.loads((run / 'initialization.json').read_text())
    assert cfg['environment'] == 'sokoban'
    results = []
    for step in range(origin['initial_step'] + 1, through + 1):
        path = run / f'rollouts_{step:04d}.jsonl.gz'
        reasons, fatal_turns, difficulty = (collections.Counter() for _ in range(3))
        buckets = collections.defaultdict(lambda: collections.Counter())
        tasks = set()
        total = successes = action_count = ineffective = invalid = checked_states = 0
        for line in gzip.open(path, 'rt'):
            row = json.loads(line)
            if str(row['lineage']) not in {'0', '1', '2', '3'}:
                continue
            task = row['task']
            env = SokobanEnv(task, max_steps=cfg['max_turns'])
            distance = distance_for_task(task)
            assert distance == task['shortest_path_length']
            assert distance is not None and distance <= cfg['max_turns']
            if task['uid'] not in tasks:
                tasks.add(task['uid'])
                difficulty[distance] += 1
            initial_distance = distance
            first_failure = None
            for turn in row['trace']:
                action = turn['action']
                if action is None:
                    continue  # A token-limit ending has not executed an action.
                assert list(env.anchor()) == turn['anchor_before']
                before = env.anchor()
                observation, reward, done, info = env.step(action)
                assert list(env.anchor()) == turn['anchor_after']
                assert observation == turn['observation'] and reward == turn['reward']
                assert env.turns == turn['turn']
                checked_states += 1
                action_count += 1
                ineffective += before == env.anchor()
                invalid += action == 0
                distance = distance_for_task(task, env.player, env.box)
                remaining = cfg['max_turns'] - env.turns
                if first_failure is None and (distance is None or distance > remaining):
                    first_failure = ('structural_deadlock' if distance is None else 'insufficient_environment_turns', env.turns)
            succeeded = row['reward'] > 0
            assert succeeded == env.success()
            if succeeded:
                assert first_failure is None and row['terminal'] == 'success'
            else:
                if first_failure is None:
                    assert row['terminal'] in ('token_limit', 'context_limit')
                    reasons['length_ended_while_still_solvable_within_turns'] += 1
                else:
                    reasons[first_failure[0]] += 1
                    fatal_turns[first_failure[1]] += 1
            total += 1
            successes += succeeded
            buckets[initial_distance]['trajectories'] += 1
            buckets[initial_distance]['successes'] += succeeded
        assert total == cfg['train_batch_size'] * cfg['initial_branches']
        assert len(tasks) == cfg['train_batch_size']
        assert sum(reasons.values()) == total - successes
        results.append(dict(step=step, original_count=total, successes=successes,
                            failed=total-successes, first_irrecoverable_cause=dict(reasons),
                            first_irrecoverable_turn=dict(sorted(fatal_turns.items())),
                            irrecoverable_by_turn3=sum(v for k, v in fatal_turns.items() if k <= 3),
                            initial_question_difficulty=dict(sorted(difficulty.items())),
                            original_successes_by_difficulty={str(k): dict(v) for k, v in sorted(buckets.items())},
                            executed_actions=action_count, ineffective_actions=ineffective,
                            invalid_actions=invalid, checked_after_action_states=checked_states,
                            rollout_sha256=sha(path)))
    return dict(unix=time.time(), run=str(run), through_step=through,
                source_sha256={name: sha(code/name) for name in ('task_envs.py','sokoban_exact.py')},
                protocol='Original four only. First transition to structural deadlock or shortest remaining path longer than remaining environment turns. Length is assigned only if the terminal state is otherwise still solvable within the horizon.',
                caveat='Training-only mechanism audit on varying sampled questions. No rewards, prompts, masks, sampling rules or queue entries changed. These categories do not establish UCF benefit or causal attribution of training behavior.',
                rows=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--through', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.code, args.run, args.through)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix('.tmp')
    temp.write_text(json.dumps(report, indent=2)+'\n')
    temp.replace(args.output)
    print(json.dumps([dict(step=r['step'], failed=r['failed'], reasons=r['first_irrecoverable_cause'], early=r['irrecoverable_by_turn3']) for r in report['rows']]))
