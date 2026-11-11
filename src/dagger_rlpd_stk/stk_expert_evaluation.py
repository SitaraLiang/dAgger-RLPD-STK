"""Evaluate the expert (Agent 3, restricted to acceleration and steering) on the
same final-evaluation laps as DAgger and RLPD."""

import argparse
from pathlib import Path

import numpy as np

from dagger_rlpd_stk.stk_experiment import (
    Agent3DAggerExpert,
    agent3_class,
    evaluate_policy,
    make_stk_env,
    write_json,
)


class ExpertPolicy:
    """Gives `evaluate_policy` the expert through the learners' interface."""

    def __init__(self, expert: Agent3DAggerExpert):
        self.expert = expert

    def select_action(self, _state) -> np.ndarray:
        return self.expert(_state)

    def eval(self) -> None:
        """Nothing to switch off: the expert is not a network."""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/stk/expert"))
    parser.add_argument("--track", default="abyss")
    parser.add_argument("--player-name", default="Jean")
    parser.add_argument("--final-eval-episodes", type=int, default=100)
    parser.add_argument("--evaluation-seed", type=int, default=10_000)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = make_stk_env(args.track, args.player_name)
    try:
        expert = Agent3DAggerExpert(env, agent3_class())
        policy = ExpertPolicy(expert)
        episodes = []
        # One lap at a time, so that every lap starts with a fresh Agent 3
        for lap in range(args.final_eval_episodes):
            expert.reset()
            result = evaluate_policy(policy, env, 1, args.evaluation_seed + lap)
            episodes += result["episodes"]
            print(episodes[-1])
    finally:
        env.close()

    lengths = [e["length"] for e in episodes if e["finished"]]
    summary = {
        "mean_return": float(np.mean([e["return"] for e in episodes])),
        "return_std": float(np.std([e["return"] for e in episodes])),
        "completion_rate": float(np.mean([e["finished"] for e in episodes])),
        "mean_completion_length": float(np.mean(lengths)) if lengths else None,
        "mean_distance": float(np.mean([e["distance"] for e in episodes])),
    }
    write_json(
        args.output_dir / "expert_final_evaluation.json",
        {"summary": summary, "episodes": episodes},
    )
    print(summary)
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
