from typing import Any

import gymnasium as gym
import numpy as np


class MixedObservationEncoder:
    """Encode continuous and categorical observations into one flat vector.

    Continuous features are kept as-is. Gymnasium's flattening convention is
    used for the discrete space, which one-hot encodes Discrete and
    MultiDiscrete values.
    """

    def __init__(self, observation_space: gym.spaces.Dict):
        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError("observation_space must be a gymnasium.spaces.Dict")
        if "continuous" not in observation_space.spaces:
            raise ValueError("observation_space must contain 'continuous'")
        if "discrete" not in observation_space.spaces:
            raise ValueError("observation_space must contain 'discrete'")

        self.continuous_space = observation_space["continuous"]
        self.discrete_space = observation_space["discrete"]
        self.continuous_dim = gym.spaces.flatdim(self.continuous_space)
        self.discrete_dim = gym.spaces.flatdim(self.discrete_space)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.continuous_dim + self.discrete_dim,),
            dtype=np.float32,
        )

    def encode(self, observation: dict[str, Any]) -> np.ndarray:
        continuous = gym.spaces.flatten(
            self.continuous_space, observation["continuous"]
        ).astype(np.float32, copy=False)
        discrete = gym.spaces.flatten(
            self.discrete_space, observation["discrete"]
        ).astype(np.float32, copy=False)
        return np.concatenate((continuous, discrete), dtype=np.float32)


class MixedObservationWrapper(gym.ObservationWrapper):
    """Expose a Dict observation as continuous values plus one-hot categories."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_encoder = MixedObservationEncoder(env.observation_space)
        self.observation_space = self.observation_encoder.observation_space
        # RLPD uses this boundary to normalize only genuinely continuous inputs.
        self.continuous_observation_dim = self.observation_encoder.continuous_dim

    def observation(self, observation):
        return self.observation_encoder.encode(observation)


class ContinuousObservationNormalizer(gym.ObservationWrapper):
    """Standardize a flat observation's continuous prefix with fixed statistics."""

    def __init__(self, env: gym.Env, mean: np.ndarray, std: np.ndarray):
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("observation_space must be a gymnasium.spaces.Box")
        if len(env.observation_space.shape) != 1:
            raise ValueError("observation_space must be one-dimensional")

        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        self.std = np.asarray(std, dtype=np.float32).reshape(-1)
        if self.mean.shape != self.std.shape:
            raise ValueError("mean and std must have the same shape")
        if len(self.mean) > env.observation_space.shape[0]:
            raise ValueError("normalization statistics exceed the observation size")
        if np.any(self.std <= 0):
            raise ValueError("std must be strictly positive")

        self.continuous_observation_dim = len(self.mean)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, observation):
        observation = np.asarray(observation, dtype=np.float32).copy()
        continuous = observation[: self.continuous_observation_dim]
        observation[: self.continuous_observation_dim] = (
            continuous - self.mean
        ) / self.std
        return observation
