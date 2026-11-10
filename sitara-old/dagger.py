"""DAgger, Algorithm 3.1 of Ross, Gordon & Bagnell (AISTATS 2011).

Hyper-parameters are those of the paper's Super Tux Kart experiment (Section 5.1).
"""

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor, nn


@dataclass
class DAggerConfig:
    seed: int = 0
    n_iterations: int = 20  # N
    beta_p: float = 0.0  # beta_i = p^(i-1); p = 0 gives the paper's beta_i = I(i = 1)
    ridge_lambda: float = 1e-3
    results_dir: str = "results"


def to_vector(obs: Any) -> Tensor:
    """Convert a flattened observation to a vector, ignoring non-numeric `kart_type`."""
    parts = [
        np.asarray(obs[key], dtype=np.float32).ravel()
        for key in ("continuous", "discrete")
    ]
    return torch.as_tensor(np.concatenate(parts))


def select_action(policy: nn.Module, obs: Tensor, env: gym.Env) -> np.ndarray:
    with torch.no_grad():
        action = policy(obs).numpy()
    return np.clip(action, env.action_space.low, env.action_space.high)


def fit_ridge(observations: Tensor, actions: Tensor, ridge_lambda: float) -> nn.Linear:
    """Train the linear policy y = w^T x + b on the whole dataset.

    Closed-form minimizer of (1/n) sum_i (w^T x_i + b - y_i)^2 + (lambda/2) w^T w;
    the bias b is not regularized.
    """
    x, y = observations.double(), actions.double()
    x_mean, y_mean = x.mean(dim=0), y.mean(dim=0)
    x_centered, y_centered = x - x_mean, y - y_mean
    n = len(x)

    covariance = x_centered.T @ x_centered / n
    regularizer = (ridge_lambda / 2) * torch.eye(x.shape[1], dtype=x.dtype)
    weights = torch.linalg.solve(covariance + regularizer, x_centered.T @ y_centered / n)
    bias = y_mean - x_mean @ weights

    policy = nn.Linear(x.shape[1], y.shape[1])
    with torch.no_grad():
        policy.weight.copy_(weights.T.float())
        policy.bias.copy_(bias.float())
    return policy


def train_dagger(
    env: gym.Env,
    expert: Callable[[Any], np.ndarray],
    validate: Callable[[nn.Module], float],
    config: DAggerConfig,
) -> tuple[nn.Module, dict[str, np.ndarray]]:
    """Run DAgger.

    :param env: A single-kart environment with a continuous (`Box`) action space
    :param expert: Maps the current observation to the expert's action; has `reset()`
    :param validate: Scores a policy on validation laps; higher is better
    :param config: The hyper-parameters
    :return: The best policy on validation and the per-iteration history
    """
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
    policy = None  # With beta_1 = 1, the initial policy is never executed
    best_policy, best_score = None, -np.inf
    obs, _ = env.reset(seed=config.seed)

    for iteration in range(config.n_iterations):
        beta = config.beta_p**iteration

        # Sample one trajectory with pi_i = beta_i expert + (1 - beta_i) policy,
        # until the lap is finished or the environment's time limit is reached
        terminated = truncated = False
        while not (terminated or truncated):
            # Every visited state gets an expert label, whoever drives
            obs_vector = to_vector(obs)
            expert_action = np.asarray(expert(obs), dtype=np.float32)
            dataset_obs.append(obs_vector)
            dataset_actions.append(torch.as_tensor(expert_action.ravel()))

            if random.random() < beta:
                action = expert_action
            else:
                action = select_action(policy, obs_vector, env)

            obs, _, terminated, truncated, _ = env.step(action)
        obs, _ = env.reset()
        expert.reset()

        # Train the next policy on the aggregated dataset
        observations = torch.stack(dataset_obs)
        actions = torch.stack(dataset_actions)
        policy = fit_ridge(observations, actions, config.ridge_lambda)
        with torch.no_grad():
            train_loss = nn.functional.mse_loss(policy(observations), actions).item()

        score = validate(policy)
        if score > best_score:
            best_policy, best_score = policy, score

        history["beta"].append(beta)
        history["n_labels"].append(len(dataset_obs))
        history["train_loss"].append(train_loss)
        history["validation_score"].append(score)
        print(
            f"iter {iteration + 1}: beta={beta:.3f} labels={len(dataset_obs)} "
            f"train_loss={train_loss:.5f} validation_score={score:.2f}"
        )

    results = {key: np.asarray(values) for key, values in history.items()}
    save_results(best_policy, results, config)
    return best_policy, results


def save_results(
    policy: nn.Module, results: dict[str, np.ndarray], config: DAggerConfig
) -> None:
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = results_dir / f"DAgger_seed_{config.seed}"
    np.savez(f"{stem}.npz", **results, config=np.array(str(asdict(config))))
    torch.save(policy.state_dict(), f"{stem}.pth")
