"""The expert: a team's agent from the course repository, cloned next to this project."""

import importlib
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from env import CONTROLLED_ACTIONS

COURSE_REPO = Path(__file__).resolve().parent.parent / "pystk_project"


class Expert:
    """Gives the agent's [acceleration, steer] for the current state, without acting."""

    def __init__(self, env: gym.Env, team: int = 3, course_repo: Path = COURSE_REPO):
        # Appended, so the pinned pystk2_gymnasium wins over the course repo's copy
        sys.path.append(str(course_repo / "src"))
        module = importlib.import_module(f"agents.team{team}.agent{team}")

        self.agent_class = getattr(module, f"Agent{team}")
        self.env = env.unwrapped
        self.reset()

    def reset(self) -> None:
        """Clear the agent's timers for a new lap. A new agent is needed because
        its own reset() would also restart the environment."""
        self.agent = self.agent_class(self.env, path_lookahead=3)

    def __call__(self, _obs: Any = None) -> np.ndarray:
        """Label the current state; the agent reads the raw observation, not `_obs`."""
        raw_obs = self.env.get_observation(self.env.kart_ix, self.env.agent.use_ai)
        action = self.agent.choose_action(raw_obs)
        return np.array(
            [np.ravel(action[key])[0] for key in CONTROLLED_ACTIONS], dtype=np.float32
        )
