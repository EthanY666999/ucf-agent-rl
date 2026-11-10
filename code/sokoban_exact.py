"""CPU-only exact reachability for finite, deterministic, one-box Sokoban.

Distances count primitive player actions, including pushes.  They are NOT capped
by the episode horizon: ``None`` means no solution even with unlimited steps.
The horizon is applied separately by ``classify_state``.  No policy, tokenizer,
GPU library, environment mutation, or rollout RNG is involved.
"""
from collections import deque
from functools import lru_cache

_DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))


@lru_cache(maxsize=256)
def _distances(walls, goal, nrow, ncol):
    """Reverse BFS from all solved states over the complete legal state graph."""
    walls = frozenset(walls)
    floor = frozenset((r, c) for r in range(nrow) for c in range(ncol)
                      if (r, c) not in walls)
    if goal not in floor:
        raise ValueError('Goal must be a traversable cell')
    predecessors = {}
    solved = []
    for player in floor:
        for box in floor:
            if player == box:
                continue
            state = (player, box)
            if box == goal:
                solved.append(state)
                continue  # Success is absorbing, as in the environment.
            for dr, dc in _DIRECTIONS:
                moved_player = (player[0] + dr, player[1] + dc)
                if moved_player not in floor:
                    continue  # A blocked move is a self loop; cannot shorten a path.
                moved_box = box
                if moved_player == box:
                    moved_box = (box[0] + dr, box[1] + dc)
                    if moved_box not in floor:
                        continue
                successor = (moved_player, moved_box)
                predecessors.setdefault(successor, []).append(state)
    distance = {state: 0 for state in solved}
    queue = deque(solved)
    while queue:
        state = queue.popleft()
        for previous in predecessors.get(state, ()):
            if previous not in distance:
                distance[previous] = distance[state] + 1
                queue.append(previous)
    return distance


def shortest_distance(walls, goal, player, box, nrow=6, ncol=6):
    """Return exact minimum remaining actions, zero if solved, or None if dead."""
    walls = tuple(sorted(set(tuple(p) for p in walls)))
    goal, player, box = tuple(goal), tuple(player), tuple(box)
    if nrow <= 0 or ncol <= 0:
        raise ValueError('Board dimensions must be positive')
    for p in (goal, player, box):
        if len(p) != 2 or not (0 <= p[0] < nrow and 0 <= p[1] < ncol) or p in walls:
            raise ValueError('Goal, player and box must be valid floor positions')
    if player == box:
        raise ValueError('Player and box cannot share a cell')
    return _distances(walls, goal, nrow, ncol).get((player, box))


def distance_for_task(task, player=None, box=None):
    """Convenient read-only adapter for task dictionaries used in saved logs."""
    return shortest_distance(task['walls'], task['goal'],
                             task['player'] if player is None else player,
                             task['box'] if box is None else box,
                             task.get('size', 6), task.get('size', 6))


def classify_state(walls, goal, player, box, remaining_steps=None, nrow=6, ncol=6):
    """Separate structural deadlock from an otherwise solvable state over budget."""
    if remaining_steps is not None and remaining_steps < 0:
        raise ValueError('Remaining steps cannot be negative')
    distance = shortest_distance(walls, goal, player, box, nrow, ncol)
    within = None if remaining_steps is None else distance is not None and distance <= remaining_steps
    return dict(distance=distance, solvable=distance is not None,
                solved=distance == 0, structural_deadlock=distance is None,
                remaining_steps=remaining_steps, within_budget=within,
                budget_exhausted=remaining_steps is not None and distance is not None and distance > remaining_steps)


def cache_info():
    return _distances.cache_info()


def clear_cache():
    _distances.cache_clear()
