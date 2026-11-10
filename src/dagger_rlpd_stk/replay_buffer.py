from typing import Protocol

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor
from tqdm.auto import trange


class Policy(Protocol):
    def select_action(self, state: Tensor | np.ndarray) -> np.ndarray: ...


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size: int = int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = torch.zeros((max_size, state_dim))
        self.action = torch.zeros((max_size, action_dim))
        self.next_state = torch.zeros((max_size, state_dim))
        self.reward = torch.zeros((max_size, 1))
        self.terminated = torch.zeros((max_size, 1))
        self.truncated = torch.zeros((max_size, 1))

    def add(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
        reward: float,
        terminated: float,
        truncated: float,
    ):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.terminated[self.ptr] = terminated
        self.truncated[self.ptr] = truncated

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        indices = torch.randint(0, self.size, size=(batch_size,))

        return (
            self.state[indices],
            self.action[indices],
            self.next_state[indices],
            self.reward[indices],
            self.terminated[indices],
            self.truncated[indices],
        )


def compute_observation_statistics(
    replay_buffer: ReplayBuffer,
    continuous_dim: int,
) -> tuple[Tensor, Tensor]:
    """Compute fixed continuous-feature statistics from a replay buffer."""
    if replay_buffer.size == 0:
        raise ValueError("cannot compute statistics from an empty replay buffer")
    if not 0 < continuous_dim <= replay_buffer.state.shape[1]:
        raise ValueError("continuous_dim must fit within the state dimension")

    continuous = replay_buffer.state[: replay_buffer.size, :continuous_dim]
    mean = continuous.mean(dim=0)
    std = continuous.std(dim=0).clamp_min(1e-3)
    return mean, std


def normalize_replay_buffer(
    replay_buffer: ReplayBuffer,
    mean: Tensor,
    std: Tensor,
) -> ReplayBuffer:
    """Normalize a replay buffer in place using fixed continuous statistics."""
    mean = torch.as_tensor(mean, dtype=replay_buffer.state.dtype).reshape(-1)
    std = torch.as_tensor(std, dtype=replay_buffer.state.dtype).reshape(-1)
    if mean.shape != std.shape:
        raise ValueError("mean and std must have the same shape")
    if len(mean) > replay_buffer.state.shape[1]:
        raise ValueError("normalization statistics exceed the state size")
    if torch.any(std <= 0):
        raise ValueError("std must be strictly positive")

    rows = slice(0, replay_buffer.size)
    columns = slice(0, len(mean))
    replay_buffer.state[rows, columns] = (
        replay_buffer.state[rows, columns] - mean
    ) / std
    replay_buffer.next_state[rows, columns] = (
        replay_buffer.next_state[rows, columns] - mean
    ) / std
    return replay_buffer


def create_offline_replay_buffer(
    policy: Policy,
    env: gym.Env,
    size: int,
    seed: int | None = None,
) -> ReplayBuffer:
    """Collect a fixed offline dataset by rolling out ``policy`` in ``env``."""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(state_dim, action_dim, size)

    state, _ = env.reset(seed=seed)
    for _ in trange(size):
        action = policy.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        replay_buffer.add(
            torch.as_tensor(state, dtype=torch.float32),
            torch.as_tensor(action, dtype=torch.float32),
            torch.as_tensor(next_state, dtype=torch.float32),
            reward,
            terminated,
            truncated,
        )

        if terminated or truncated:
            state, _ = env.reset()
        else:
            state = next_state

    return replay_buffer
