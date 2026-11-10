"""Run a seeded RLPD experiment on SuperTuxKart."""

import argparse
import platform
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from dagger_rlpd_stk.rlpd import RLPDConfig, run_rlpd
from dagger_rlpd_stk.stk_experiment import (
    evaluate_policy,
    make_normalized_env,
    prepare_offline_data,
    write_json,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run seeded RLPD training on STK.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/stk/rlpd"))
    parser.add_argument("--track", default="abyss")
    parser.add_argument("--player-name", default="Jean")
    parser.add_argument("--offline-transitions", type=int, default=100_000)
    parser.add_argument("--rlpd-timesteps", type=int, default=500_000)
    parser.add_argument("--rlpd-eval-frequency", type=int, default=2_000)
    parser.add_argument("--rlpd-eval-episodes", type=int, default=2)
    parser.add_argument("--final-eval-episodes", type=int, default=100)
    parser.add_argument("--evaluation-seed", type=int, default=10_000)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_dir / f"seed_{args.seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    logs_dir = run_dir / "tensorboard"
    logs_dir.mkdir()

    config = RLPDConfig(
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
    write_json(
        run_dir / "metadata.json",
        {
            "algorithm": "rlpd",
            "seed": args.seed,
            "track": args.track,
            "player_name": args.player_name,
            "offline_transitions": args.offline_transitions,
            "final_eval_episodes": args.final_eval_episodes,
            "evaluation_seed": args.evaluation_seed,
            "rlpd_config": asdict(config),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    )

    offline_buffer, observation_mean, observation_std = prepare_offline_data(
        args.offline_transitions,
        args.seed,
        args.track,
        args.player_name,
        run_dir,
    )

    def make_learning_env():
        return make_normalized_env(
            args.track, args.player_name, observation_mean, observation_std
        )

    env = make_learning_env()
    evaluation_env = make_learning_env()
    try:
        policy, evaluations = run_rlpd(
            env=env,
            evaluation_env=evaluation_env,
            offline_buffer=offline_buffer,
            config=config,
            save_results=run_dir / "rlpd_learning_curve.pt",
            log_dir=logs_dir / "rlpd",
        )
    finally:
        env.close()
        evaluation_env.close()

    torch.save(
        {
            "actor": policy.actor.state_dict(),
            "critics": policy.critics.state_dict(),
            "evaluations": evaluations,
        },
        run_dir / "rlpd_policy.pt",
    )
    final_env = make_learning_env()
    try:
        final_evaluation = evaluate_policy(
            policy, final_env, args.final_eval_episodes, args.evaluation_seed
        )
    finally:
        final_env.close()
    write_json(run_dir / "rlpd_final_evaluation.json", final_evaluation)
    write_json(
        run_dir / "summary.json",
        {"seed": args.seed, "rlpd": final_evaluation["summary"]},
    )
    print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()
