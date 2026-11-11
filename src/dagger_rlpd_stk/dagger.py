"""Dataset Aggregation (DAgger) for continuous-control environments."""

import copy
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import trange

from dagger_rlpd_stk.training import timestamped_log_dir


class Expert(Protocol):
    def __call__(self, observation: np.ndarray) -> np.ndarray: ...
    def reset(self) -> None: ...


@dataclass(frozen=True)
class DAggerConfig:
    seed: int = 0
    n_iterations: int = 20
    beta_p: float = 0.0
    ridge_lambda: float = 1e-3
    results_dir: str | Path | None = None


class DAggerPolicy(nn.Module):
    """Linear policy with the same inference interface as SAC and RLPD."""

    def __init__(self, state_dim: int, action_low: np.ndarray, action_high: np.ndarray):
        super().__init__()

        # action = W · observation + b
        self.linear = nn.Linear(state_dim, len(action_low))
        self.register_buffer(
            "action_low", torch.as_tensor(action_low, dtype=torch.float32)
        )
        self.register_buffer(
            "action_high", torch.as_tensor(action_high, dtype=torch.float32)
        )

    def forward(self, observation: Tensor) -> Tensor:
        return torch.clamp(self.linear(observation), self.action_low, self.action_high)

    def select_action(self, state: Tensor | np.ndarray) -> np.ndarray:
        state = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            return self(state).cpu().numpy()


def fit_ridge(
    observations: Tensor,
    actions: Tensor,
    ridge_lambda: float,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> DAggerPolicy:
    """
    Fit a linear policy by closed-form ridge regression.
    
    x = [ normalised velocity (3), front (3), …, paths_start (15), …, items (15), kart (3),   ← 80 continuous numbers
      one-hot phase, one-hot powerup, …, one-hot item types ]                              ← 0/1 numbers
    """
    x, y = observations.double(), actions.double()
    x_mean, y_mean = x.mean(dim=0), y.mean(dim=0)
    x_centered, y_centered = x - x_mean, y - y_mean
    n = len(x)
    covariance = x_centered.T @ x_centered / n
    regularizer = (ridge_lambda / 2) * torch.eye(x.shape[1], dtype=x.dtype)
    weights = torch.linalg.solve(
        covariance + regularizer, x_centered.T @ y_centered / n
    )
    bias = y_mean - x_mean @ weights

    policy = DAggerPolicy(x.shape[1], action_low, action_high)
    with torch.no_grad():
        policy.linear.weight.copy_(weights.T.float())
        policy.linear.bias.copy_(bias.float())
    return policy


def run_dagger(
    env: gym.Env,
    expert: Expert,
    validate: Callable[[DAggerPolicy], float],
    config: DAggerConfig | None = None,
    log_dir: str | Path | None = None,
) -> tuple[DAggerPolicy, dict[str, np.ndarray]]:
    """Aggregate expert labels on visited states and refit the policy."""
    config = config or DAggerConfig()
    if config.n_iterations < 1:
        raise ValueError("n_iterations must be at least 1")
    if not 0.0 <= config.beta_p <= 1.0:
        raise ValueError("beta_p must be between 0 and 1")
    if (
        not isinstance(env.observation_space, gym.spaces.Box)
        or len(env.observation_space.shape) != 1
    ):
        raise TypeError(
            "DAgger requires a flat Box observation space; use MixedObservationWrapper"
        )
    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError("DAgger requires a continuous Box action space")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    env.action_space.seed(config.seed)
    dataset_obs: list[Tensor] = []
    dataset_actions: list[Tensor] = []
    history: dict[str, list[float]] = {
        "beta": [],
        "n_labels": [],
        "train_loss": [],
        "validation_score": [],
    }
    policy: DAggerPolicy | None = None
    best_policy: DAggerPolicy | None = None
    best_score = -np.inf
    observation, _ = env.reset(seed=config.seed)

    with SummaryWriter(timestamped_log_dir(log_dir)) as writer:
        progress = trange(config.n_iterations, desc="DAgger")
        for iteration in progress:
            beta = config.beta_p**iteration
            terminated = truncated = False
            while not (terminated or truncated):
                observation_vector = torch.as_tensor(
                    np.array(observation, dtype=np.float32)
                )
                expert_action = np.asarray(
                    expert(observation), dtype=np.float32
                ).ravel()
                expert_action = np.clip(
                    expert_action, env.action_space.low, env.action_space.high
                )
                dataset_obs.append(observation_vector)
                dataset_actions.append(torch.as_tensor(expert_action.copy()))
                if random.random() < beta:
                    action = expert_action
                else:
                    if policy is None:
                        raise RuntimeError(
                            "policy is unavailable before the expert rollout"
                        )
                    action = policy.select_action(observation_vector)
                observation, _, terminated, truncated, _ = env.step(action)

            observation, _ = env.reset()
            expert.reset()
            observations = torch.stack(dataset_obs)
            actions = torch.stack(dataset_actions)
            policy = fit_ridge(
                observations,
                actions,
                config.ridge_lambda,
                env.action_space.low,
                env.action_space.high,
            )
            with torch.no_grad():
                train_loss = nn.functional.mse_loss(
                    policy(observations), actions
                ).item()
            score = validate(policy)
            if score > best_score:
                best_policy, best_score = copy.deepcopy(policy), score

            iteration_number = iteration + 1
            n_labels = len(dataset_obs)
            history["beta"].append(beta)
            history["n_labels"].append(n_labels)
            history["train_loss"].append(train_loss)
            history["validation_score"].append(score)
            writer.add_scalar("dagger/beta", beta, iteration_number)
            writer.add_scalar("dagger/n_labels", n_labels, iteration_number)
            writer.add_scalar("loss/train", train_loss, iteration_number)
            writer.add_scalar("evaluation/score", score, iteration_number)
            progress.set_postfix(
                beta=f"{beta:.3f}",
                labels=n_labels,
                train_loss=f"{train_loss:.5f}",
                validation_score=f"{score:.2f}",
            )

    if best_policy is None:
        raise RuntimeError("DAgger did not produce a policy")
    results = {key: np.asarray(values) for key, values in history.items()}
    if config.results_dir is not None:
        save_results(best_policy, results, config)
    return best_policy, results


def save_results(
    policy: DAggerPolicy, results: dict[str, np.ndarray], config: DAggerConfig
) -> None:
    if config.results_dir is None:
        return
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = results_dir / f"DAgger_seed_{config.seed}"
    np.savez(f"{stem}.npz", **results, config=np.array(str(asdict(config))))
    torch.save(policy, f"{stem}.pth")
