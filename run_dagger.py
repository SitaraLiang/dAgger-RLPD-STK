"""Train DAgger on one track, for one seed."""

import argparse

import gymnasium as gym
from torch import nn

from dagger import DAggerConfig, select_action, to_vector, train_dagger
from env import TRACK, make_env
from expert import Expert

VALIDATION_SEED = 1000  # Kept apart from the training seeds
UNFINISHED_PENALTY = 1e6  # An unfinished lap always scores below a finished one


def validate(policy: nn.Module, env: gym.Env) -> float:
    """Score one lap: -steps if finished, otherwise distance - UNFINISHED_PENALTY."""
    obs, _ = env.reset(seed=VALIDATION_SEED)
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = select_action(policy, to_vector(obs), env)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
    return -steps if terminated else info["distance"] - UNFINISHED_PENALTY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default=TRACK)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--team", type=int, default=3, help="Expert's team number")
    parser.add_argument("--iterations", type=int, default=DAggerConfig.n_iterations)
    parser.add_argument("--results-dir", help="Defaults to results/<track>")
    args = parser.parse_args()

    config = DAggerConfig(
        seed=args.seed,
        n_iterations=args.iterations,
        results_dir=args.results_dir or f"results/{args.track}",
    )
    env = make_env(track=args.track)
    validation_env = make_env(track=args.track)
    try:
        train_dagger(
            env,
            Expert(env, team=args.team),
            lambda policy: validate(policy, validation_env),
            config,
        )
    finally:
        env.close()
        validation_env.close()


if __name__ == "__main__":
    main()
