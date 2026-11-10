"""The race shared by every method: one kart alone, a fixed track, one lap."""

from typing import Any

import gymnasium as gym
import numpy as np
from pystk2_gymnasium import AgentSpec
from pystk2_gymnasium.stk_wrappers import (
    ConstantSizedObservations,
    OnlyContinuousActionsWrapper,
    PolarObservations,
)
from pystk2_gymnasium.wrappers import FlattenerWrapper

TRACK = "abyss"
MAX_EPISODE_STEPS = 1500  # The default time limit of the pystk2-gymnasium environments
STEP_SECONDS = 0.1  # Documented default simulation time per step (pystk2 step_size)

#: Order of the learner's actions: the flattener sorts the action keys
CONTROLLED_ACTIONS = ("acceleration", "steer")
DISABLED_ACTIONS = ("brake", "drift", "fire", "nitro", "rescue")


def _make(
    env_id: str, track: str, max_episode_steps: int, render_mode: str | None
) -> gym.Env:
    return gym.make(
        env_id,
        render_mode=render_mode,
        track=track,
        num_kart=1,
        laps=1,
        agent=AgentSpec(use_ai=False),
        max_episode_steps=max_episode_steps,
        disable_env_checker=True,
    )


class FillEmptyObservations(gym.ObservationWrapper):
    """Add one zero entry to empty lists, which ConstantSizedObservations cannot stack.

    Alone on the track, there is never another kart in `karts_position`.
    """

    def observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        obs = {**obs}
        if len(obs["karts_position"]) == 0:
            obs["karts_position"] = (np.zeros(3, dtype=np.float32),)
        if len(obs["items_position"]) == 0:
            obs["items_position"] = (np.zeros(3, dtype=np.float32),)
            obs["items_type"] = (0,)
        return obs


def make_env(
    track: str = TRACK,
    max_episode_steps: int = MAX_EPISODE_STEPS,
    render_mode: str | None = None,
) -> gym.Env:
    """The learner's environment: flattened observations, actions [acceleration, steer].

    Same wrappers as `supertuxkart/flattened_continuous_actions-v0`, which crashes
    with a single kart, and no slots for other karts beyond one zero entry.
    """
    env = make_full_env(track, max_episode_steps, render_mode)
    env = FillEmptyObservations(env)
    env = ConstantSizedObservations(env, state_karts=1)
    env = PolarObservations(env)
    env = OnlyContinuousActionsWrapper(env)
    return FlattenerWrapper(env)


def make_full_env(
    track: str = TRACK,
    max_episode_steps: int = MAX_EPISODE_STEPS,
    render_mode: str | None = None,
) -> gym.Env:
    """The expert's environment: raw observations and dict actions."""
    return _make("supertuxkart/full-v0", track, max_episode_steps, render_mode)


def restrict_action(action: dict[str, Any]) -> dict[str, Any]:
    """Keep only the learner's controls, as the continuous-actions wrapper does."""
    return {**action, **{key: 0 for key in DISABLED_ACTIONS}}


if __name__ == "__main__":
    # Check the learner's environment with one episode of random actions
    env = make_env()
    try:
        print("action space:", env.action_space)
        print("observation space:", env.observation_space)
        env.action_space.seed(0)
        env.reset(seed=0)
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            steps += 1
        print(
            f"steps={steps} terminated={terminated} truncated={truncated} "
            f"distance={info['distance']:.1f}"
        )
    finally:
        env.close()
