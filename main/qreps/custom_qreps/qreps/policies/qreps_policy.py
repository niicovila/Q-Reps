"""Python Script Template."""
import torch
import torch.nn as nn
from qreps.policies.abstract_q_policy import AbstractQFunctionPolicy
from qreps.valuefunctions import AbstractQFunction
import torch
from torch import nn
from torch import distributions
from torch.nn import functional as F
from functools import partial
from torch.distributions import Categorical
from .stochasticpolicy import StochasticPolicy


class QREPSPolicy(AbstractQFunctionPolicy):
    """Implementation of a softmax policy with some small off-set for stability."""

    def __init__(self, q_function: AbstractQFunction, temp):
        super().__init__(q_function)
        self.counter = 0
        self.temperature = temp

    def reset(self):
        """Reset parameters and update counter."""
        self.counter += 1

    def distribution(self, x) -> torch.distributions.Distribution:
        q_values = self.q_function.forward_state(x)
        return torch.distributions.Categorical(logits=q_values)

    def log_likelihood(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return self.distribution(features).log_prob(actions)

    def sample(self, observation: torch.Tensor):
        return self.distribution(observation).sample().item()




class QREPSPolicyV2(StochasticPolicy, nn.Module):
    def __init__(self, obs_shape, act_shape, *args, **kwargs):
        super(QREPSPolicyV2, self).__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Linear(obs_shape, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, act_shape),
        )

    def forward(self, x):
        return self.model(super(QREPSPolicyV2, self).forward(x))

    def distribution(self, observation) -> torch.distributions.Distribution:
        output = self.forward(observation)
        return torch.distributions.categorical.Categorical(logits=output)

    @torch.no_grad()
    def sample(self, observation):
        if self._stochastic:
            return self.distribution(observation).sample().item()
        else:
            return torch.argmax(self.forward(observation)).item()

    def log_likelihood(self, features, actions):
        return self.distribution(features).log_prob(actions)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=[256, 256], act=nn.ReLU, output_act=None):
        super().__init__()
        net = []
        last_dim = input_dim
        for dim in hidden_layers:
            net.append(nn.Linear(last_dim, dim))
            net.append(act())
            last_dim = dim
        net.append(nn.Linear(last_dim, output_dim))
        if not output_act is None:
            net.append(output_act())
        self.net = nn.Sequential(*net)
        self._has_output_act = False if output_act is None else True

    @property
    def last_layer(self):
        if self._has_output_act:
            return self.net[-2]
        else:
            return self.net[-1]
    def forward(self, x):
        return self.net(x)


class DiscreteMLPActorV2(nn.Module):
    def __init__(self, observation_space, action_space, hidden_layers=[256, 256], act=nn.ReLU, output_act=nn.Tanh, ortho_init=False, output_gain=None):
        super().__init__()
        self.mlp = MLP(observation_space.shape[0], action_space.n, hidden_layers=hidden_layers, act=act, output_act=output_act)
        # if ortho_init:
        #     self.apply(partial(weight_init, gain=float(ortho_init))) # use the fact that True converts to 1.0
        #     if output_gain is not None:
        #         self.mlp.last_layer.apply(partial(weight_init, gain=output_gain))
    
    def forward(self, obs):
        action_probs = F.softmax(self.mlp(obs), dim=1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1)
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)
        return actions, action_probs, log_action_probs
    
    def predict(self, obs, sample=False):
        h = self.mlp(obs)
        out = F.softmax(h, dim=1)
        actions = torch.argmax(
            out, dim=1)
        if sample:
            action_dist = Categorical(out)
            actions = action_dist.sample().view(-1)
        return actions

