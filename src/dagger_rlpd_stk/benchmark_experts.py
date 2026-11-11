"""Compare course agents alone on a track, with full and restricted controls.

The course repository is in pystk_project/ at the project root (see --course-repo).
"""

import argparse
import csv
import importlib
import os
import sys
from pathlib import Path

import numpy as np

from dagger_rlpd_stk.env import (
    MAX_EPISODE_STEPS,
    TRACK,
    make_full_env,
    restrict_action,
)
from dagger_rlpd_stk.stk_experiment import PROJECT_ROOT

MODES = ("full", "restricted")
FIELDS = [
    "agent",
    "mode",
    "seed",
    "status",
    "steps",
    "distance",
    "error",
]


def load_agent_class(team: int):
    module = importlib.import_module(f"agents.team{team}.agent{team}")
    return getattr(module, f"Agent{team}")


def run_lap(env, agent_class, mode: str, seed: int) -> dict:
    """Drive one lap. A new agent per lap clears its timers, because the agents'
    own reset() would also restart the environment."""
    agent = agent_class(env.unwrapped, path_lookahead=3)
    obs, _ = env.reset(seed=seed)
    steps = 0
    distance = 0.0
    terminated = truncated = False

    while not (terminated or truncated):
        action = agent.choose_action(obs)
        if mode == "restricted":
            action = restrict_action(action)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        distance = float(info["distance"])

    return {
        "status": "finished" if terminated else "time_limit", # finished or time limit: the finish rate
        "steps": steps, #the lap time, the main metric, lower is better
        "distance": round(distance, 1), # shows how far an unfinished lap got
        "error": "", # records crashed laps
    }


def print_summary(rows: list[dict]) -> None:
    print(
        f"\n{'agent':<8}{'mode':<12}{'finished':<10}{'mean steps':<12}{'std':<8}"
    )
    for agent in sorted({row["agent"] for row in rows}):
        for mode in MODES:
            laps = [r for r in rows if r["agent"] == agent and r["mode"] == mode]
            if not laps:
                continue
            steps = [r["steps"] for r in laps if r["status"] == "finished"]
            finished = f"{len(steps)}/{len(laps)}"
            if steps:
                print(
                    f"{agent:<8}{mode:<12}{finished:<10}{np.mean(steps):<12.1f}"
                    f"{np.std(steps):<8.1f}"
                )
            else:
                print(f"{agent:<8}{mode:<12}{finished:<10}{'-':<12}{'-':<8}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--course-repo", type=Path, default=PROJECT_ROOT / "pystk_project"
    )
    parser.add_argument("--teams", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--track", default=TRACK)
    parser.add_argument("--laps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument(
        "--output", type=Path, help="Defaults to results/expert_<track>.csv"
    )
    args = parser.parse_args()

    output = (args.output or PROJECT_ROOT / f"results/expert_{args.track}.csv").resolve()
    src = args.course_repo.resolve() / "src"
    if not (src / "agents").is_dir():
        parser.error(f"course agents not found in {src}")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Appended, so the pinned pystk2_gymnasium wins over the course repo's copy
    sys.path.append(str(src))
    # Team 2 loads its configuration from a path relative to src/main
    os.chdir(src / "main")

    env = make_full_env(track=args.track, max_episode_steps=args.max_steps)
    rows = []
    try:
        with output.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            for team in args.teams:
                for mode in MODES:
                    for seed in range(args.laps):
                        row = {"agent": f"team{team}", "mode": mode, "seed": seed}
                        try:
                            row |= run_lap(env, load_agent_class(team), mode, seed)
                        except Exception as error:  # A failed lap must stay visible
                            row |= {"status": "error", "error": repr(error)}
                        rows.append(row)
                        writer.writerow(row)
                        file.flush()
                        print(row)
    finally:
        env.close()

    print_summary(rows)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
