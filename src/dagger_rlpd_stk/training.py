"""Utilities shared by algorithm-specific training entry points."""

from datetime import datetime
from pathlib import Path

import gymnasium as gym
import torch


def timestamped_log_dir(log_dir: str | Path | None):
    if log_dir is None:
        return None
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{log_dir}_{timestamp}"


def evaluate(policy, env: gym.Env, eval_episodes: int):
    episode_returns = []
    getattr(policy, "actor", policy).eval()
    with torch.no_grad():
        for _ in range(eval_episodes):
            observation, _ = env.reset()
            episode_return = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                action = policy.select_action(observation)
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
            episode_returns.append(episode_return)
    return sum(episode_returns) / eval_episodes
