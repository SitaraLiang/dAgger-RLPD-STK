"""Run a seeded DAgger experiment on SuperTuxKart."""

import argparse
import platform
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from dagger_rlpd_stk.dagger import DAggerConfig, run_dagger
from dagger_rlpd_stk.stk_experiment import (
    Agent3DAggerExpert,
    agent3_class,
    evaluate_policy,
    make_normalized_env,
    prepare_offline_data,
    write_json,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run seeded DAgger training on STK.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/stk/dagger"))
    parser.add_argument("--track", default="abyss")
    parser.add_argument("--player-name", default="Jean")
    parser.add_argument("--offline-transitions", type=int, default=100_000)
    parser.add_argument("--dagger-iterations", type=int, default=20)
    parser.add_argument("--dagger-beta", type=float, default=0.0)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
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

    config = DAggerConfig(
        seed=args.seed,
        n_iterations=args.dagger_iterations,
        beta_p=args.dagger_beta,
        ridge_lambda=args.ridge_lambda,
        results_dir=run_dir / "dagger",
    )
    write_json(
        run_dir / "metadata.json",
        {
            "algorithm": "dagger",
            "seed": args.seed,
            "track": args.track,
            "player_name": args.player_name,
            "offline_transitions": args.offline_transitions,
            "validation_episodes": args.validation_episodes,
            "validation_seed": args.validation_seed,
            "final_eval_episodes": args.final_eval_episodes,
            "evaluation_seed": args.evaluation_seed,
            "dagger_config": asdict(config),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    )

    _, observation_mean, observation_std = prepare_offline_data(
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
    validation_env = make_learning_env()
    try:
        expert = Agent3DAggerExpert(env, agent3_class())
        policy, history = run_dagger(
            env=env,
            expert=expert,
            validate=lambda candidate: evaluate_policy(
                candidate,
                validation_env,
                args.validation_episodes,
                args.validation_seed,
            )["summary"]["mean_return"],
            config=config,
            log_dir=logs_dir / "dagger",
        )
    finally:
        env.close()
        validation_env.close()

    np.savez(run_dir / "dagger_history.npz", **history)
    torch.save(policy.state_dict(), run_dir / "dagger_policy.pt")
    final_env = make_learning_env()
    try:
        final_evaluation = evaluate_policy(
            policy, final_env, args.final_eval_episodes, args.evaluation_seed
        )
    finally:
        final_env.close()
    write_json(run_dir / "dagger_final_evaluation.json", final_evaluation)
    write_json(
        run_dir / "summary.json",
        {"seed": args.seed, "dagger": final_evaluation["summary"]},
    )
    print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()
