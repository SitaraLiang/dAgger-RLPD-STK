import copy
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import trange

from dagger_rlpd_stk.models import Critic, SquashedGaussianActor
from dagger_rlpd_stk.replay_buffer import ReplayBuffer
from dagger_rlpd_stk.training import evaluate, timestamped_log_dir


@dataclass(frozen=True)
class SACConfig:
    discount: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    initial_alpha: float = 0.2
    hidden_dim: int = 64
    min_std: float = 1e-4
    layer_normalization: bool = False
    batch_size: int = 256
    start_timesteps: int = 25_000
    max_timesteps: int = 500_000
    eval_episodes: int = 10
    eval_freq: int = 5_000
    replay_buffer_size: int = 1_000_000
    seed: int = 0


class SAC:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        config: SACConfig | None = None,
    ):
        self.config = config or SACConfig()

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

        self.critic_1 = Critic(
            state_dim,
            action_dim,
            self.config.layer_normalization,
            self.config.hidden_dim,
        )
        self.critic_2 = Critic(
            state_dim,
            action_dim,
            self.config.layer_normalization,
            self.config.hidden_dim,
        )
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_2_target = copy.deepcopy(self.critic_2)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=self.config.critic_lr,
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

        with torch.no_grad():
            if deterministic:
                action = self.actor.act(state)
            else:
                action, _ = self.actor(state)

        return action.numpy().flatten()

    def train(self, replay_buffer: ReplayBuffer):
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()

        state, action, next_state, reward, terminated, _truncated = (
            replay_buffer.sample(self.config.batch_size)
        )

        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_state)
            target_q_1 = self.critic_1_target(next_state, next_action)
            target_q_2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q_1, target_q_2)
            target_q = target_q - self.alpha.detach() * next_log_prob.unsqueeze(1)
            target_q = reward + ((1.0 - terminated) * self.config.discount * target_q)

        current_q_1 = self.critic_1(state, action)
        current_q_2 = self.critic_2(state, action)
        critic_loss = F.mse_loss(current_q_1, target_q) + F.mse_loss(
            current_q_2, target_q
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_action, log_prob = self.actor(state)
        q_1 = self.critic_1(state, new_action)
        q_2 = self.critic_2(state, new_action)
        actor_loss = (
            self.alpha.detach() * log_prob.unsqueeze(1) - torch.min(q_1, q_2)
        ).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(
            self.log_alpha * (log_prob + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for parameter, target_parameter in zip(
            self.critic_1.parameters(), self.critic_1_target.parameters()
        ):
            target_parameter.data.copy_(
                self.config.tau * parameter.data
                + (1 - self.config.tau) * target_parameter.data
            )

        for parameter, target_parameter in zip(
            self.critic_2.parameters(), self.critic_2_target.parameters()
        ):
            target_parameter.data.copy_(
                self.config.tau * parameter.data
                + (1 - self.config.tau) * target_parameter.data
            )

        return actor_loss.item(), critic_loss.item()


def run_sac(
    env: gym.Env,
    evaluation_env: gym.Env,
    config: SACConfig | None = None,
    save_results: str | Path | None = None,
    log_dir: str | Path | None = None,
):
    config = config or SACConfig()
    env.action_space.seed(config.seed)
    evaluation_env.action_space.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    policy = SAC(state_dim, action_dim, max_action, config)
    replay_buffer = ReplayBuffer(state_dim, action_dim, config.replay_buffer_size)
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
            replay_buffer.add(
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
                actor_loss, critic_loss = policy.train(replay_buffer)
                writer.add_scalar("loss/actor", actor_loss, timestep)
                writer.add_scalar("loss/critic", critic_loss, timestep)
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
