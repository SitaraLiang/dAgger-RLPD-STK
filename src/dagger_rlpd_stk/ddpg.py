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

from dagger_rlpd_stk.models import Actor, Critic
from dagger_rlpd_stk.replay_buffer import ReplayBuffer
from dagger_rlpd_stk.training import evaluate, timestamped_log_dir


@dataclass(frozen=True)
class DDPGConfig:
    discount: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    critic_weight_decay: float = 1e-2
    hidden_dim: int = 64
    layer_normalization: bool = False
    batch_size: int = 256
    start_timesteps: int = 25_000
    max_timesteps: int = 500_000
    eval_episodes: int = 10
    eval_freq: int = 5_000
    exploration_noise: float = 0.1
    replay_buffer_size: int = 1_000_000
    seed: int = 0


class DDPG:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        config: DDPGConfig | None = None,
    ):
        self.config = config or DDPGConfig()

        self.actor = Actor(state_dim, action_dim, max_action, self.config.hidden_dim)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_lr
        )

        self.critic = Critic(
            state_dim,
            action_dim,
            self.config.layer_normalization,
            self.config.hidden_dim,
        )
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.config.critic_lr,
            weight_decay=self.config.critic_weight_decay,
        )

    def select_action(self, state: Tensor | np.ndarray):
        state = torch.as_tensor(state, dtype=torch.float32).reshape(1, -1)

        with torch.no_grad():
            action = self.actor(state)

        return action.numpy().flatten()

    def train(self, replay_buffer: ReplayBuffer):
        self.actor.train()
        self.critic.train()

        state, action, next_state, reward, terminated, _truncated = (
            replay_buffer.sample(self.config.batch_size)
        )

        with torch.no_grad():
            target_q = self.critic_target(next_state, self.actor_target(next_state))
            target_q = reward + ((1.0 - terminated) * self.config.discount * target_q)

        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for parameter, target_parameter in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_parameter.data.copy_(
                self.config.tau * parameter.data
                + (1 - self.config.tau) * target_parameter.data
            )

        for parameter, target_parameter in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            target_parameter.data.copy_(
                self.config.tau * parameter.data
                + (1 - self.config.tau) * target_parameter.data
            )

        return actor_loss.item(), critic_loss.item()

    def save(self, filename: str):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename: str):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer")
        )
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


def run_ddpg(
    env: gym.Env,
    evaluation_env: gym.Env,
    config: DDPGConfig | None = None,
    save_results: str | Path | None = None,
    log_dir: str | Path | None = None,
):
    config = config or DDPGConfig()
    env.action_space.seed(config.seed)
    evaluation_env.action_space.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    policy = DDPG(state_dim, action_dim, max_action, config)
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
                action = policy.select_action(state)
                noise = np.random.normal(
                    0, max_action * config.exploration_noise, size=action_dim
                )
                action = np.clip(
                    action + noise, env.action_space.low, env.action_space.high
                )
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
