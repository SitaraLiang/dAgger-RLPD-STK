"""Run a seeded DAgger and RLPD comparison on SuperTuxKart."""

import argparse
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from tqdm.auto import trange

from dagger_rlpd_stk.env import make_env
from dagger_rlpd_stk.dagger import DAggerConfig, run_dagger
from dagger_rlpd_stk.observations import ContinuousObservationNormalizer
from dagger_rlpd_stk.replay_buffer import (
    ReplayBuffer,
    compute_observation_statistics,
    normalize_replay_buffer,
)
from dagger_rlpd_stk.rlpd import RLPDConfig, run_rlpd
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

    def __call__(self, _encoded_observation):
        native_observation = self.native_env.get_observation(
            self.native_env.kart_ix,
            False,
        )
        return agent3_action(self.policy.choose_action(native_observation))

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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run seeded DAgger and RLPD training on the same STK task."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/stk"))
    parser.add_argument("--track", default="abyss")
    parser.add_argument("--player-name", default="Jean")
    parser.add_argument("--offline-transitions", type=int, default=100_000)
    parser.add_argument("--dagger-iterations", type=int, default=50)
    parser.add_argument("--dagger-beta", type=float, default=0.0)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--rlpd-timesteps", type=int, default=500_000)
    parser.add_argument("--rlpd-eval-frequency", type=int, default=2_000)
    parser.add_argument("--rlpd-eval-episodes", type=int, default=2)
    parser.add_argument("--final-eval-episodes", type=int, default=100)
    parser.add_argument("--validation-episodes", type=int, default=5)
    parser.add_argument("--validation-seed", type=int, default=5_000)
    parser.add_argument("--evaluation-seed", type=int, default=10_000)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_dir / f"seed_{args.seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    logs_dir = run_dir / "tensorboard"
    logs_dir.mkdir()

    dagger_config = DAggerConfig(
        seed=args.seed,
        n_iterations=args.dagger_iterations,
        beta_p=args.dagger_beta,
        ridge_lambda=args.ridge_lambda,
        results_dir=run_dir / "dagger",
    )
    rlpd_config = RLPDConfig(
        replay_buffer_size=args.offline_transitions,
        hidden_dim=256,
        layer_normalization=True,
        batch_size=256,
        start_timesteps=0,
        max_timesteps=args.rlpd_timesteps,
        eval_episodes=args.rlpd_eval_episodes,
        eval_freq=args.rlpd_eval_frequency,
        n_ensemble_critics=3,
        n_subset_critics=2,
        utd_ratio=2,
        entropy=False,
        normalize_observations=False,
        seed=args.seed,
    )
    metadata = {
        "seed": args.seed,
        "track": args.track,
        "player_name": args.player_name,
        "offline_transitions": args.offline_transitions,
        "validation_episodes": args.validation_episodes,
        "validation_seed": args.validation_seed,
        "final_eval_episodes": args.final_eval_episodes,
        "evaluation_seed": args.evaluation_seed,
        "dagger_config": asdict(dagger_config),
        "rlpd_config": asdict(rlpd_config),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    write_json(run_dir / "metadata.json", metadata)

    offline_buffer, continuous_dim = collect_agent3_offline_buffer(
        args.offline_transitions,
        args.seed,
        args.track,
        args.player_name,
    )
    observation_mean, observation_std = compute_observation_statistics(
        offline_buffer,
        continuous_dim,
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

    def make_learning_env():
        return ContinuousObservationNormalizer(
            make_stk_env(args.track, args.player_name),
            observation_mean.numpy(),
            observation_std.numpy(),
        )

    dagger_env = make_learning_env()
    dagger_validation_env = make_learning_env()
    try:
        expert = Agent3DAggerExpert(dagger_env, agent3_class())
        dagger_policy, dagger_history = run_dagger(
            env=dagger_env,
            expert=expert,
            validate=lambda policy: evaluate_policy(
                policy,
                dagger_validation_env,
                args.validation_episodes,
                args.validation_seed,
            )["summary"]["mean_return"],
            config=dagger_config,
            log_dir=logs_dir / "dagger",
        )
    finally:
        dagger_env.close()
        dagger_validation_env.close()

    np.savez(run_dir / "dagger_history.npz", **dagger_history)
    torch.save(dagger_policy.state_dict(), run_dir / "dagger_policy.pt")
    dagger_final_env = make_learning_env()
    try:
        dagger_final = evaluate_policy(
            dagger_policy,
            dagger_final_env,
            args.final_eval_episodes,
            args.evaluation_seed,
        )
    finally:
        dagger_final_env.close()
    write_json(run_dir / "dagger_final_evaluation.json", dagger_final)

    rlpd_env = make_learning_env()
    rlpd_evaluation_env = make_learning_env()
    try:
        rlpd_policy, rlpd_evaluations = run_rlpd(
            env=rlpd_env,
            evaluation_env=rlpd_evaluation_env,
            offline_buffer=offline_buffer,
            config=rlpd_config,
            save_results=run_dir / "rlpd_learning_curve.pt",
            log_dir=logs_dir / "rlpd",
        )
    finally:
        rlpd_env.close()
        rlpd_evaluation_env.close()

    torch.save(
        {
            "actor": rlpd_policy.actor.state_dict(),
            "critics": rlpd_policy.critics.state_dict(),
            "evaluations": rlpd_evaluations,
        },
        run_dir / "rlpd_policy.pt",
    )
    rlpd_final_env = make_learning_env()
    try:
        rlpd_final = evaluate_policy(
            rlpd_policy,
            rlpd_final_env,
            args.final_eval_episodes,
            args.evaluation_seed,
        )
    finally:
        rlpd_final_env.close()
    write_json(run_dir / "rlpd_final_evaluation.json", rlpd_final)
    write_json(
        run_dir / "summary.json",
        {
            "seed": args.seed,
            "dagger": dagger_final["summary"],
            "rlpd": rlpd_final["summary"],
        },
    )
    print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()
