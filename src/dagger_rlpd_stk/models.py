import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Independent, Normal


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state: Tensor):
        action = torch.relu(self.l1(state))
        action = torch.relu(self.l2(action))
        return self.max_action * torch.tanh(self.l3(action))


class SquashedGaussianActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        hidden_dim: int = 64,
        min_std: float = 1e-4,
    ):
        super().__init__()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.std = nn.Linear(hidden_dim, action_dim)

        self.softplus = nn.Softplus()
        self.min_std = min_std
        self.max_action = max_action
        self.log_action_scale = action_dim * math.log(max_action)

    def features(self, state: Tensor):
        features = torch.relu(self.l1(state))
        return torch.relu(self.l2(features))

    def dist(self, state: Tensor):
        features = self.features(state)
        mean = self.mean(features)
        std = self.softplus(self.std(features)) + self.min_std
        return Independent(Normal(mean, std), 1)

    def forward(self, state: Tensor):
        distribution = self.dist(state)
        raw_action = distribution.rsample()
        normalized_action = torch.tanh(raw_action)
        log_prob_correction = 2 * (
            math.log(2) - raw_action - F.softplus(-2 * raw_action)
        )
        log_prob = distribution.log_prob(raw_action)
        log_prob = log_prob - log_prob_correction.sum(dim=1)
        log_prob = log_prob - self.log_action_scale
        return self.max_action * normalized_action, log_prob

    def act(self, state: Tensor):
        features = self.features(state)
        return self.max_action * torch.tanh(self.mean(features))


class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        layer_normalization: bool = False,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        self.layer_normalization = layer_normalization
        if layer_normalization:
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, state: Tensor, action: Tensor):
        features = self.l1(torch.cat([state, action], dim=1))

        if self.layer_normalization:
            features = self.ln1(features)

        features = torch.relu(features)
        features = self.l2(features)

        if self.layer_normalization:
            features = self.ln2(features)

        return self.l3(torch.relu(features))
