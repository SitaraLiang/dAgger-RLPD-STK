"""Utilities shared by algorithm-specific training entry points."""

from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


def timestamped_log_dir(log_dir: str | Path | None):
    if log_dir is None:
        return None
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{log_dir}_{timestamp}"


def evaluate(
    policy, env: gym.Env, eval_episodes: int
) -> dict[str, float | None]:
    """Return aggregate reward and successful-completion length metrics."""
    if eval_episodes < 1:
        raise ValueError("eval_episodes must be at least 1")

    episode_returns: list[float] = []
    completion_lengths: list[int] = []
    getattr(policy, "actor", policy).eval()
    with torch.no_grad():
        for _ in range(eval_episodes):
            observation, _ = env.reset()
            episode_return = 0.0
            episode_length = 0
            terminated = truncated = False
            while not (terminated or truncated):
                action = policy.select_action(observation)
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                episode_length += 1
            episode_returns.append(episode_return)
            if terminated:
                completion_lengths.append(episode_length)

    return {
        "mean_return": float(np.mean(episode_returns)),
        "mean_completion_length": (
            float(np.mean(completion_lengths)) if completion_lengths else None
        ),
    }
