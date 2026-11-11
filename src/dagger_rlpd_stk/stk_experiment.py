"""Shared SuperTuxKart experiment utilities."""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import trange

from dagger_rlpd_stk.env import make_env
from dagger_rlpd_stk.observations import ContinuousObservationNormalizer
from dagger_rlpd_stk.replay_buffer import (
    ReplayBuffer,
    compute_observation_statistics,
    normalize_replay_buffer,
)
from dagger_rlpd_stk.stk_rlpd_wrapper import STKRLPDWrapper

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYSTK_SOURCE = PROJECT_ROOT / "pystk_project" / "src"


def agent3_class():
    """Load the project-local Agent 3 implementation."""
    source = str(PYSTK_SOURCE)
    if source not in sys.path:
        sys.path.append(source)
    from agents.team3.agent3 import Agent3

    return Agent3


def make_stk_env(track: str, player_name: str, render_mode=None):
    return STKRLPDWrapper(make_env(track=track, render_mode=render_mode))


def agent3_action(action) -> np.ndarray:
    """
    convert the dicrionary chosen by expert (agent 3) into learner's format 
    [acceleration, steer] and range [-1, 1]
    """
    acceleration = float(np.asarray(action["acceleration"]).item())
    steering = float(np.asarray(action["steer"]).item())
    return np.array([2.0 * acceleration - 1.0, steering], dtype=np.float32)


def collect_agent3_offline_buffer(
    num_transitions: int,
    seed: int,
    track: str,
    player_name: str,
) -> tuple[ReplayBuffer, int]:
    env = make_stk_env(track, player_name)
    replay_buffer = ReplayBuffer(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        num_transitions,
    )
    continuous_dim = env.continuous_observation_dim
    policy_class = agent3_class()

    try:
        state, _ = env.reset(seed=seed)
        native_env = env.unwrapped
        policy = policy_class(native_env)

        for _ in trange(num_transitions, desc="Collecting Agent 3 data"):
            native_observation = native_env.get_observation(native_env.kart_ix, False)
            action = agent3_action(policy.choose_action(native_observation))
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
                policy = policy_class(native_env)
            else:
                state = next_state
    finally:
        env.close()

    return replay_buffer, continuous_dim


class Agent3DAggerExpert:
    def __init__(self, wrapped_env, policy_class):
        self.native_env = wrapped_env.unwrapped
        self.policy_class = policy_class
        self.policy = policy_class(self.native_env)

    def __call__(self, _encoded_observation): # ignore learner's vector
        """
        we ignore the learner's vector and expert asks the raw game env for current
        state as a dictionary (same state the vector was made from)
        """
        native_observation = self.native_env.get_observation(
            self.native_env.kart_ix,
            False,
        )
        return agent3_action(self.policy.choose_action(native_observation)) # [acceleration, steer] : expert label for current state.

    def reset(self):
        self.policy = self.policy_class(self.native_env)


def evaluate_policy(policy, env, episodes: int, seed: int) -> dict[str, Any]:
    episode_results = []
    getattr(policy, "actor", policy).eval()

    with torch.no_grad():
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            episode_return = 0.0
            terminated = truncated = False
            info = {}
            length = 0

            while not (terminated or truncated):
                action = policy.select_action(state)
                state, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                length += 1

            episode_results.append(
                {
                    "seed": seed + episode,
                    "return": episode_return,
                    "length": length,
                    "finished": bool(terminated),
                    "distance": float(info.get("distance", 0.0)),
                    "position": int(info.get("position", 0)),
                }
            )

    completed_lengths = [
        result["length"] for result in episode_results if result["finished"]
    ]
    return {
        "summary": {
            "mean_return": float(
                np.mean([result["return"] for result in episode_results])
            ),
            "return_std": float(
                np.std([result["return"] for result in episode_results])
            ),
            "completion_rate": float(
                np.mean([result["finished"] for result in episode_results])
            ),
            "mean_completion_length": (
                float(np.mean(completed_lengths)) if completed_lengths else None
            ),
            "mean_distance": float(
                np.mean([result["distance"] for result in episode_results])
            ),
        },
        "episodes": episode_results,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def prepare_offline_data(
    num_transitions: int,
    seed: int,
    track: str,
    player_name: str,
    run_dir: Path,
):
    """Collect Agent 3 data, normalize it, and persist the statistics."""
    offline_buffer, continuous_dim = collect_agent3_offline_buffer(
        num_transitions, seed, track, player_name
    )
    observation_mean, observation_std = compute_observation_statistics(
        offline_buffer, continuous_dim
    )
    normalize_replay_buffer(offline_buffer, observation_mean, observation_std)
    torch.save(
        {
            "mean": observation_mean,
            "std": observation_std,
            "continuous_dim": continuous_dim,
        },
        run_dir / "observation_normalization.pt",
    )
    return offline_buffer, observation_mean, observation_std


def make_normalized_env(
    track: str,
    player_name: str,
    observation_mean: torch.Tensor,
    observation_std: torch.Tensor,
):
    return ContinuousObservationNormalizer(
        make_stk_env(track, player_name),
        observation_mean.numpy(),
        observation_std.numpy(),
    )
