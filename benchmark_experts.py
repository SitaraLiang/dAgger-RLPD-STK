"""Compare course agents alone on a track, with full and restricted controls.

The course repository is cloned next to this project (see --course-repo).
"""

import argparse
import csv
import importlib
import os
import sys
import time
from pathlib import Path

import numpy as np

from env import MAX_EPISODE_STEPS, STEP_SECONDS, TRACK, make_full_env, restrict_action

MODES = ("full", "restricted")
FIELDS = [
    "agent",
    "mode",
    "seed",
    "status",
    "steps",
    "game_seconds",
    "distance",
    "wall_seconds",
    "steps_per_second",
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
    start = time.perf_counter()

    while not (terminated or truncated):
        action = agent.choose_action(obs)
        if mode == "restricted":
            action = restrict_action(action)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        distance = float(info["distance"])

    wall_seconds = time.perf_counter() - start
    return {
        "status": "finished" if terminated else "time_limit",
        "steps": steps,
        "game_seconds": round(steps * STEP_SECONDS, 1),
        "distance": round(distance, 1),
        "wall_seconds": round(wall_seconds, 2),
        "steps_per_second": round(steps / wall_seconds, 1),
        "error": "",
    }


def print_summary(rows: list[dict]) -> None:
    print(f"\n{'agent':<8}{'mode':<12}{'finished':<10}{'mean steps':<12}{'std':<8}{'mean s':<8}")
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
                    f"{np.std(steps):<8.1f}{np.mean(steps) * STEP_SECONDS:<8.1f}"
                )
            else:
                print(f"{agent:<8}{mode:<12}{finished:<10}{'-':<12}{'-':<8}{'-':<8}")


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course-repo", type=Path, default=root.parent / "pystk_project")
    parser.add_argument("--teams", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--track", default=TRACK)
    parser.add_argument("--laps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument("--output", type=Path, help="Defaults to results/expert_<track>.csv")
    args = parser.parse_args()

    output = (args.output or root / f"results/expert_{args.track}.csv").resolve()
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
