import copy
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import trange

from dagger_rlpd_stk.models import Critic, SquashedGaussianActor
from dagger_rlpd_stk.replay_buffer import ReplayBuffer
from dagger_rlpd_stk.training import evaluate, timestamped_log_dir


@dataclass(frozen=True)
class RLPDConfig:
    discount: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    initial_alpha: float = 0.2
    hidden_dim: int = 64
    min_std: float = 1e-4
    layer_normalization: bool = True
    batch_size: int = 256
    start_timesteps: int = 0
    max_timesteps: int = 500_000
    eval_episodes: int = 10
    eval_freq: int = 5_000
    replay_buffer_size: int = 1_000_000
    # new things different from SAC
    n_ensemble_critics: int = 10
    n_subset_critics: int = 2
    utd_ratio: int = 10
    entropy: bool = False
    normalize_observations: bool = True
    seed: int = 0


class RLPD:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        config: RLPDConfig | None = None,
        observation_mean: Tensor | None = None,
        observation_std: Tensor | None = None,
        continuous_state_dim: int | None = None,
    ):
        self.config = config or RLPDConfig()
        self.continuous_state_dim = continuous_state_dim or state_dim
        self.observation_mean = (
            observation_mean
            if observation_mean is not None
            else torch.zeros(self.continuous_state_dim)
        )
        self.observation_std = (
            observation_std
            if observation_std is not None
            else torch.ones(self.continuous_state_dim)
        )

        self.actor = SquashedGaussianActor(
            state_dim,
            action_dim,
            max_action,
            self.config.hidden_dim,
            self.config.min_std,
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_lr
        )

        self.critics = nn.ModuleList(
            [
                Critic(
                    state_dim,
                    action_dim,
                    self.config.layer_normalization,
                    self.config.hidden_dim,
                )
                for _ in range(self.config.n_ensemble_critics)
            ]
        )
        self.critic_targets = copy.deepcopy(self.critics)
        self.critic_optimizer = torch.optim.Adam(
            self.critics.parameters(), lr=self.config.critic_lr
        )

        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.tensor(
            np.log(self.config.initial_alpha),
            dtype=torch.float32,
            requires_grad=True,
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.config.alpha_lr
        )

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state: Tensor | np.ndarray, deterministic: bool = True):
        state = torch.as_tensor(state, dtype=torch.float32).reshape(1, -1)
        state = self._normalize_observation(state)

        with torch.no_grad():
            if deterministic:
                action = self.actor.act(state)
            else:
                action, _ = self.actor(state)

        return action.numpy().flatten()

    def train(self, online_buffer: ReplayBuffer, offline_buffer: ReplayBuffer):
        self.actor.train()
        self.critics.train()

        critic_losses = []
        for _ in range(self.config.utd_ratio):
            batch = self._sample_mixed_batch(online_buffer, offline_buffer)
            critic_loss = self._update_critics(batch)
            critic_losses.append(critic_loss.item())
            self._update_targets()

        state = batch[0]
        actor_loss, log_prob = self._update_actor(state)

        if self.config.entropy:
            self._update_alpha(log_prob)

        return actor_loss.item(), sum(critic_losses) / len(critic_losses)

    def _sample_mixed_batch(
        self, online_buffer: ReplayBuffer, offline_buffer: ReplayBuffer
    ):
        # Sample 50% online and 50% offline
        online_batch_size = self.config.batch_size // 2
        offline_batch_size = self.config.batch_size - online_batch_size
        online_batch = online_buffer.sample(online_batch_size)
        offline_batch = offline_buffer.sample(offline_batch_size)

        state, action, next_state, reward, terminated, truncated = tuple(
            torch.cat((online, offline), dim=0)
            for online, offline in zip(online_batch, offline_batch)
        )
        return (
            self._normalize_observation(state),
            action,
            self._normalize_observation(next_state),
            reward,
            terminated,
            truncated,
        )

    def _normalize_observation(self, observation: Tensor):
        continuous = observation[..., : self.continuous_state_dim]
        continuous = (continuous - self.observation_mean) / self.observation_std
        if self.continuous_state_dim == observation.shape[-1]:
            return continuous
        discrete = observation[..., self.continuous_state_dim :]
        return torch.cat((continuous, discrete), dim=-1)

    def _update_critics(self, batch):
        state, action, next_state, reward, terminated, _truncated = batch

        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_state)
            # Randomly choose n_subset_critics critics to use
            target_indices = torch.randperm(self.config.n_ensemble_critics)[
                : self.config.n_subset_critics
            ]
            target_q_values = torch.stack(
                [
                    self.critic_targets[index](next_state, next_action)
                    for index in target_indices
                ]
            )
            target_q = target_q_values.min(dim=0).values
            if self.config.entropy:
                target_q = target_q - self.alpha.detach() * next_log_prob.unsqueeze(1)
            target_q = reward + ((1.0 - terminated) * self.config.discount * target_q)

        critic_loss = sum(
            F.mse_loss(critic(state, action), target_q) for critic in self.critics
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss

    def _update_actor(self, state: Tensor):
        new_action, log_prob = self.actor(state)
        q_values = torch.stack([critic(state, new_action) for critic in self.critics])
        actor_loss = -q_values.mean(dim=0)
        if self.config.entropy:
            actor_loss = actor_loss + self.alpha.detach() * log_prob.unsqueeze(1)
        actor_loss = actor_loss.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss, log_prob

    def _update_alpha(self, log_prob: Tensor):
        alpha_loss = -(
            self.log_alpha * (log_prob + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

    def _update_targets(self):
        for critic, critic_target in zip(self.critics, self.critic_targets):
            for parameter, target_parameter in zip(
                critic.parameters(), critic_target.parameters()
            ):
                target_parameter.data.copy_(
                    self.config.tau * parameter.data
                    + (1 - self.config.tau) * target_parameter.data
                )


def run_rlpd(
    env: gym.Env,
    evaluation_env: gym.Env,
    offline_buffer: ReplayBuffer,
    config: RLPDConfig | None = None,
    save_results: str | Path | None = None,
    log_dir: str | Path | None = None,
):
    config = config or RLPDConfig()
    env.action_space.seed(config.seed)
    evaluation_env.action_space.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    observation_mean = None
    observation_std = None
    continuous_state_dim = getattr(env, "continuous_observation_dim", state_dim)
    if config.normalize_observations:
        offline_states = offline_buffer.state[
            : offline_buffer.size, :continuous_state_dim
        ]
        observation_mean = offline_states.mean(dim=0)
        observation_std = offline_states.std(dim=0).clamp_min(1e-3)
    policy = RLPD(
        state_dim,
        action_dim,
        max_action,
        config,
        observation_mean,
        observation_std,
        continuous_state_dim,
    )
    online_buffer = ReplayBuffer(state_dim, action_dim, config.replay_buffer_size)
    evaluations = [evaluate(policy, evaluation_env, config.eval_episodes)]

    with SummaryWriter(timestamped_log_dir(log_dir)) as writer:
        writer.add_scalar("evaluation/return", evaluations[0], 0)
        state, _ = env.reset(seed=config.seed)
        episode_return = 0.0
        episode_timesteps = 0
        episode = 0
        progress = trange(1, config.max_timesteps + 1)
        for timestep in progress:
            if timestep <= config.start_timesteps:
                action = env.action_space.sample()
            else:
                action = policy.select_action(state, deterministic=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            online_buffer.add(
                torch.as_tensor(state, dtype=torch.float32),
                torch.as_tensor(action, dtype=torch.float32),
                torch.as_tensor(next_state, dtype=torch.float32),
                reward,
                terminated,
                truncated,
            )
            state = next_state
            episode_return += float(reward)
            episode_timesteps += 1
            if timestep > config.start_timesteps:
                actor_loss, critic_loss = policy.train(online_buffer, offline_buffer)
                writer.add_scalar("loss/actor", actor_loss, timestep)
                writer.add_scalar("loss/critic", critic_loss, timestep)
                if config.entropy:
                    writer.add_scalar("policy/alpha", policy.alpha.item(), timestep)
            if terminated or truncated:
                episode += 1
                writer.add_scalar("training/episode_return", episode_return, timestep)
                writer.add_scalar(
                    "training/episode_length", episode_timesteps, timestep
                )
                progress.set_postfix(
                    episode=episode, episode_return=f"{episode_return:.2f}"
                )
                state, _ = env.reset()
                episode_return = 0.0
                episode_timesteps = 0
            if timestep % config.eval_freq == 0:
                evaluation = evaluate(policy, evaluation_env, config.eval_episodes)
                evaluations.append(evaluation)
                writer.add_scalar("evaluation/return", evaluation, timestep)
                progress.set_postfix(evaluation_return=f"{evaluation:.2f}")

    if save_results is not None:
        torch.save(evaluations, save_results)
    return policy, evaluations
