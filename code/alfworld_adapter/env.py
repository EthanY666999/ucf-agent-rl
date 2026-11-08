"""ALFWorld 0.4.2 / TextWorld 1.7.0, text-only CPU execution.

Environment protocol: 50 actions, two observation/action pairs in history,
official train/valid_seen/valid_unseen games, no domain randomization.
The fenced action format is a deliberate difference from verl-agent's XML tags.
"""
from __future__ import annotations
import hashlib
import json
import random
import re
from pathlib import Path

MAX_STEPS = 50
HISTORY_LENGTH = 2

def parse_action(response: str) -> str | None:
    """Use the FIRST complete fence, consistent with the proposed v2.1 node.

    Do not scan backwards for a later valid action; a later block cannot move
    the branch boundary. Legality is checked separately against environment.
    """
    begin = response.find("```")
    if begin < 0:
        return None
    end = response.find("```", begin + 3)
    return response[begin + 3:end].strip() if end >= 0 else None

def render_prompt(observation: str, admissible: list[str], history=(), task="") -> str:
    recent = list(history)[-HISTORY_LENGTH:]
    past = "\n".join(f"Observation: {x['observation']}\nAction: {x['action']}" for x in recent)
    return (f"You are an agent in the ALFRED text environment. Task: {task}\n"
            f"Steps already taken: {len(history)}. Recent {len(recent)} observation/action pairs:\n{past}\n"
            f"Current observation: {observation}\nAdmissible actions: {json.dumps(admissible)}\n"
            "Reason about this step, then output one admissible action inside triple backticks. "
            "The first fenced block is executed. Do not put examples or code fences in your reasoning.")

class ALFWorldEpisode:
    def __init__(self, gamefile: str | Path, seed: int = 42, max_steps: int = MAX_STEPS):
        if max_steps != MAX_STEPS:
            raise ValueError("Release protocol fixes ALFWorld to 50 environment steps")
        import numpy as np
        import textworld
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos, AlfredExpert
        self.gamefile = Path(gamefile).resolve()
        self.seed = seed
        ambient = (random.getstate(), np.random.get_state())
        try:
            random.seed(seed)
            np.random.seed(seed)
            self.engine = textworld.start(str(self.gamefile),
                textworld.EnvInfos(won=True, admissible_commands=True, extras=["expert_plan"]),
                wrappers=[AlfredDemangler(shuffle=False), AlfredInfos, AlfredExpert])
            self.state = self.engine.reset()
            self._random_state = (random.getstate(), np.random.get_state())
        finally:
            random.setstate(ambient[0]); np.random.set_state(ambient[1])
        self.actions: list[str] = []
        self.history: list[dict] = []
        self.done = False
        self.task = self.state.feedback.split("Your task is to:")[-1].strip()

    @property
    def observation(self):
        return self.state.feedback

    @property
    def admissible(self):
        return list(self.state.admissible_commands)

    @property
    def won(self):
        return bool(self.state.won)

    def prompt(self):
        return render_prompt(self.observation, self.admissible, self.history, self.task)

    def step(self, action: str):
        if self.done:
            raise RuntimeError("Cannot step a terminal episode")
        previous = self.observation
        import numpy as np
        ambient = (random.getstate(), np.random.get_state())
        try:
            random.setstate(self._random_state[0]); np.random.set_state(self._random_state[1])
            self.state, _, native_done = self.engine.step(action)
            self._random_state = (random.getstate(), np.random.get_state())
        finally:
            random.setstate(ambient[0]); np.random.set_state(ambient[1])
        self.actions.append(action)
        self.history.append({"observation": previous, "action": action})
        self.done = bool(native_done or len(self.actions) >= MAX_STEPS)
        return self.observation, float(self.won), self.done

    def fork(self):
        """Fresh environment plus exact action replay; never mutate the parent."""
        clone = type(self)(self.gamefile, self.seed)
        for action in self.actions:
            clone.step(action)
        if clone.observation != self.observation or clone.won != self.won:
            clone.close()
            raise AssertionError("ALFWorld fork replay diverged")
        return clone

    def consequences(self):
        result = {}
        for action in self.admissible:
            copy = self.fork()
            try:
                obs, _, _ = copy.step(action)
                result[action] = ("GOAL" if copy.won else "STAY" if "Nothing happens" in obs
                                  else "OBS:" + hashlib.sha256(obs.encode()).hexdigest())
            finally:
                copy.close()
        return result

    def expert_action(self):
        action = self.state["extra.expert_plan"][0]
        if action not in self.admissible:
            raise RuntimeError("Hand-coded expert returned a non-admissible action")
        return action

    def expert_remaining(self) -> int | None:
        """Number of actions in a fresh replayed expert continuation; None on timeout.

        This is an expert rollout length, NOT an optimal distance. No model
        continuation or eventual actor reward is used. None must fail closed in
        UCF preflight until a missing-expert policy is explicitly specified.
        """
        if self.won:
            return 0
        copy = self.fork()
        try:
            distance = 0
            while not copy.done:
                copy.step(copy.expert_action())
                distance += 1
            return distance if copy.won else None
        finally:
            copy.close()

    def process_score(self, initial_expert_steps: int):
        if initial_expert_steps <= 0:
            raise ValueError("initial expert steps must be positive")
        remaining = self.expert_remaining()
        if remaining is None:
            raise RuntimeError("Expert cannot solve current state within the remaining 50-step horizon")
        # Preserve the requested formula; do not silently clamp it or change beta/F.
        return 1.0 - remaining / initial_expert_steps

    def close(self):
        self.engine.close()
