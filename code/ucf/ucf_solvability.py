"""CPU-only, explicitly enabled P1/P2 selection policies.

This changes only candidate admission/probability, not rewards, consequence
classes, process scores or the Part B loss. In particular it makes no claim
that removing dead-state rows preserves the full PPO+KL update.
"""
import math
from collections import Counter, deque
from ucf_select import RHO_MIN, feature


def pilot_enabled(cfg):
    for key in ('solvability_gate', 'early_exploration_floor'):
        if not isinstance(cfg.get(key, False), bool):
            raise TypeError(key + ' must be boolean')
    if cfg.get('early_exploration_floor', False) and not cfg.get('solvability_gate', False):
        raise ValueError('P2 early exploration requires the P1 gate')
    return cfg.get('solvability_gate', False)


def controller_bootstrap_c(saved_c):
    """Restore the legacy bookkeeping object after a zero-support batch.

    Pilot probabilities never use this bootstrap: question-LOO recalculates
    every c below. The actual zero c is retained in controller.json and logs.
    """
    if not math.isfinite(saved_c) or saved_c < 0:
        raise ValueError('Invalid saved pilot calibration constant')
    return saved_c if saved_c > 0 else 1.


def exact_distance(env):
    """Primitive-action distance; None means structural death, not timeout."""
    if getattr(env, 'environment', None) == 'sokoban':
        from sokoban_exact import shortest_distance
        return shortest_distance(env.walls, env.goal, env.player, env.box, env.nrow, env.ncol)
    if getattr(env, 'is_slippery', False):
        raise ValueError('Exact gate requires the deterministic FrozenLake protocol')
    start = int(env.s)
    def tile(pos):
        x = env.desc[pos // env.ncol, pos % env.ncol]
        return x.decode() if isinstance(x, bytes) else str(x)
    if tile(start) == 'H':
        return None  # Holes are terminal; one cannot walk back out.
    q = deque([(start, 0)])
    seen = {start}
    while q:
        pos, d = q.popleft()
        if tile(pos) == 'G':
            return d
        r, c = divmod(pos, env.ncol)
        for dr, dc in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < env.nrow and 0 <= cc < env.ncol:
                nxt = rr * env.ncol + cc
                if nxt not in seen and tile(nxt) != 'H':
                    seen.add(nxt)
                    q.append((nxt, d + 1))
    return None


def annotate_candidate(node, first, cfg):
    """Use the copied pre-action state, never the sampled action or suffix."""
    branch = node['branch']
    elapsed = int(branch.turns)
    assert int(node['turn']) == elapsed + 1, 'Gate expects a pre-action node'
    remaining = int(cfg['max_turns']) - elapsed
    assert remaining >= 0
    d = exact_distance(branch.env)
    within = d is not None and d <= remaining
    w, ph, _ = feature(node, first, cfg['ucf_variant'])
    early = 1 <= int(node['turn']) <= 3
    floor = RHO_MIN
    if cfg.get('early_exploration_floor', False) and within and early:
        floor += .05 * (1. - ph)
    anchor = branch.env.anchor() if hasattr(branch.env, 'anchor') else (int(branch.env.s),)
    node['solvability'] = dict(
        bfs_distance=d, remaining_turns=remaining, structural_deadlock=d is None,
        within_budget=within, solvability_gated=not within, early_node=early,
        rho_floor=0. if not within else floor, state=int(branch.env.s),
        state_anchor=list(anchor), prefix_turns=elapsed,
    )
    return w, floor


def calibrate_floors(task_specs, target=2.):
    """Calibrate on all OTHER tasks, including tasks with no eligible nodes.

    Each eligible node has (w, floor). Inaccessible targets use the pre-existing
    reachable-upper-limit rule. Zero-weight nodes retain their explicit floor.
    """
    if not task_specs:
        raise ValueError('Calibration needs at least one other task')
    if not math.isfinite(target) or target < 0:
        raise ValueError('Invalid target')
    for specs in task_specs:
        for w, floor in specs:
            if not (math.isfinite(w) and 0 <= w <= 1 and math.isfinite(floor) and 0 <= floor <= 1):
                raise ValueError('Invalid candidate weight or floor')
    n = len(task_specs)
    lower = sum(sum(f for _, f in ns) for ns in task_specs) / n
    upper = sum(sum(1. if w > 0 else f for w, f in ns) for ns in task_specs) / n
    if lower > target + 1e-12:
        raise ValueError('Declared probability floors exceed target budget')
    def avg(c):
        return sum(sum(min(1., max(f, c*w)) for w, f in ns) for ns in task_specs) / n
    if upper <= target:
        positive = [w for ns in task_specs for w, _ in ns if w > 0]
        c = math.nextafter(max(1./w for w in positive), math.inf) if positive else 0.
    else:
        lo, hi = 0., 1.
        while avg(hi) < target:
            hi *= 2
        for _ in range(80):
            mid = (lo + hi) / 2
            if avg(mid) < target:
                lo = mid
            else:
                hi = mid
        c = (lo + hi) / 2
    return c, dict(target=target, minimum_expected_forks=lower,
                   maximum_expected_forks=upper, calibrated_expected_forks=avg(c),
                   target_unreachable=upper < target, calibration_task_count=n)


def pilot_probability(node, c, w):
    meta = node['solvability']
    return 0. if meta['solvability_gated'] else min(1., max(meta['rho_floor'], c*w))


def prepare_pilot(candidates, first, cfg, target=2.):
    assert pilot_enabled(cfg)
    if cfg.get('ucf_controller', 'v2') != 'v2' or cfg.get('ucf_variant', 'ucf') != 'ucf':
        raise ValueError('P1/P2 require unmodified v2.2 UCF with question-LOO calibration')
    if cfg.get('is_slippery', False):
        raise ValueError('Exact gate requires deterministic environment transitions')
    if len(candidates) < 2:
        raise ValueError('Question-LOO requires >=2 tasks')
    specs = {}
    for ti, nodes in candidates.items():
        specs[ti] = []
        for node in nodes:
            w, floor = annotate_candidate(node, first, cfg)
            if not node['solvability']['solvability_gated']:
                specs[ti].append((w, floor))
    constants, diagnostics = {}, {}
    for ti in specs:
        constants[ti], diagnostics[ti] = calibrate_floors([ns for tj, ns in specs.items() if tj != ti], target)
    expected = {ti:sum(pilot_probability(node, constants[ti], feature(node, first, cfg['ucf_variant'])[0])
                      for node in nodes) for ti, nodes in candidates.items()}
    attainable = {ti:sum(1. if w > 0 else floor for w, floor in ns) for ti, ns in specs.items()}
    budget = dict(target=target, calibration='leave_one_question_out_with_node_floors',
                  c_by_task=constants, loo_diagnostics=diagnostics,
                  expected_forks=sum(expected.values())/len(specs), expected_forks_by_task=expected,
                  attainable_expected_forks=sum(attainable.values())/len(specs),
                  attainable_expected_forks_by_task=attainable,
                  target_unreachable=sum(attainable.values())/len(specs) < target,
                  loo_unreachable_tasks=sum(d['target_unreachable'] for d in diagnostics.values()),
                  exact_batch_budget_guaranteed=False,
                  note='All tasks retain their denominator; held-out support does not tune its own c')
    return constants, budget


def selection_log_fields(node):
    return dict(node['solvability'], S=node['S'], H_cons=node['H_cons'])


def pilot_statistics(candidates, first, children, budget):
    task_ids = list(candidates)
    success = {ti:sum(b.reward > 0 for b in first if b.task_id == ti) for ti in task_ids}
    types = {ti:'all_failure' if x == 0 else 'all_success' if x == 4 else 'mixed' for ti, x in success.items()}
    rescued = {b.task_id for b in children.values() if b.reward > 0}
    groups = {}
    for kind in ('all_failure', 'mixed', 'all_success'):
        tasks = [ti for ti in task_ids if types[ti] == kind]
        nodes = [node for ti in tasks for node in candidates[ti]]
        dead = [n for n in nodes if not n['solvability']['within_budget']]
        early = [n for n in nodes if n['solvability']['early_node']]
        expected = sum(n['rho'] for n in nodes)
        groups[kind] = dict(task_count=len(tasks), rescued_tasks=sum(ti in rescued for ti in tasks),
            candidate_count=len(nodes), eligible_candidate_count=len(nodes)-len(dead),
            dead_candidate_count=len(dead), sum_rho=expected,
            selected_count=sum(n['selected'] for n in nodes),
            dead_sum_rho=sum(n['rho'] for n in dead), dead_selected_count=sum(n['selected'] for n in dead),
            early_count=len(early), early_sum_rho=sum(n['rho'] for n in early),
            early_selected_count=sum(n['selected'] for n in early), early_rhos=[n['rho'] for n in early],
            dead_budget_share=sum(n['rho'] for n in dead)/expected if expected else None,
            early_budget_share=sum(n['rho'] for n in early)/expected if expected else None,
            early_mean_rho=sum(n['rho'] for n in early)/len(early) if early else None,
            rescue_rate=sum(ti in rescued for ti in tasks)/len(tasks) if tasks else None)
    return dict(group_counts=dict(Counter(types.values())), per_group=groups, **groups,
                expected_forks=budget['expected_forks'],
                attainable_expected_forks=budget['attainable_expected_forks'],
                target_unreachable=budget['target_unreachable'],
                loo_unreachable_tasks=budget['loo_unreachable_tasks'],
                interpretation='Gate removes dead-state Part B rows, including their KL; not a full-update unbiasedness claim')
