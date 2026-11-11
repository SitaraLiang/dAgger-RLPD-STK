"""
Same wrapper used for both RLPD and Dagger.
"""
from typing import Any

import gymnasium as gym
import numpy as np

from dagger_rlpd_stk.observations import MixedObservationEncoder


class STKRLPDWrapper(gym.Wrapper):
    """Adapt SuperTuxKart's observations and actions for RLPD.

    Mixed observations are flattened into a single vector, and native actions
    are exposed through a normalized ``[-1, 1]`` action space.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_encoder = MixedObservationEncoder(env.observation_space)
        self.observation_space = self.observation_encoder.observation_space
        self.continuous_observation_dim = self.observation_encoder.continuous_dim

        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError("action_space must be a gymnasium.spaces.Box")
        self._action_low = env.action_space.low.copy()
        self._action_high = env.action_space.high.copy()
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=env.action_space.shape,
            dtype=np.float32,
        )

    def _state(self, observation: dict[str, Any]) -> np.ndarray:
        return self.observation_encoder.encode(observation)

    def _action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return self._action_low + 0.5 * (action + 1.0) * (
            self._action_high - self._action_low
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self._state(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(
            self._action(action)
        )
        return self._state(observation), reward, terminated, truncated, info
