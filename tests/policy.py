import torch
import torch.nn as nn

import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Union

from qfunction import AbstractQFunction


class Policy:
    def __init__(self, state_space, action_space, q_function):
        self.state_space = state_space
        self.action_space = action_space
        self.q_function = q_function
        self._stochastic = False
        self.counter = 0

        _policy = torch.ones((state_space, action_space))
        _policy /= torch.sum(_policy, 1, keepdim=True)
        self._policy = _policy

    def distribution(self, x) -> torch.distributions.Distribution:
        probs = self._policy[x]
        return torch.distributions.Categorical(logits=probs)
    
    def update(self, x):
        for i in range(len(x)):
            obs = int(x[i])
            qs = self.q_function.forward_state(torch.Tensor([obs])).detach()
            factor = torch.exp(qs)
            print(factor)
            self._policy[obs] *= factor
            self._policy[obs] = torch.clamp(self._policy[obs], 0, 1)
            
            
        self._policy /= torch.sum(self._policy, 1, keepdim=True)
        print(self._policy)

    def sample(self, observation: torch.Tensor):
        return self.distribution(observation).sample().item()
    
    def log_likelihood(self, features, actions):
        return self.distribution(features).log_prob(actions)
    
    # @torch.no_grad()
    # def sample(self, observation):
    #     if self._stochastic:
    #         return self.distribution(observation).sample().item()
    #     else:
    #         return torch.argmax(self._policy[self.forward(observation)]).item()


class StochasticPolicy(nn.Module, metaclass=ABCMeta):
    """Policy base class providing necessary interface for all derived policies.
    A policy provides two main mechanisms:
    * Sampling an action giving one observation necessary for running the policy
    * Updating the policy given a trajectory of transitions and weights for the transitions
    """

    def __init__(self, feature_fn=nn.Identity()):
        super(StochasticPolicy, self).__init__()
        self._stochastic = True
        self.feature_fn = feature_fn

    @abstractmethod
    def sample(self, observation: torch.Tensor) -> Union[int, np.array]:
        """Sample the policy to obtain an action to perform"""
        pass

    @abstractmethod
    def log_likelihood(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        pass

    def set_eval_mode(self, enabled: bool):
        self._stochastic = not enabled

    def set_stochastic(self, enabled: bool):
        """Sets the policy to eval mode to enable exploitation without doing any more exploration"""
        self._stochastic = enabled

    def forward(self, x):
        return self.feature_fn(x)

    @abstractmethod
    def distribution(self, x) -> torch.distributions.Distribution:
        """Return the distribution to a specific observation"""
        pass


class AbstractQFunctionPolicy(StochasticPolicy, metaclass=ABCMeta):
    """Interface for policies to control an environment.

    Parameters
    ----------
    q_function: q_function to derive policy from.
    param: policy parameter.

    """

    def __init__(self, q_function: AbstractQFunction):
        super().__init__()
        self.q_function = q_function


class StochasticTablePolicy(StochasticPolicy, nn.Module):
    def __init__(self, n_states: int, n_actions: int, *args, **kwargs):
        super(StochasticTablePolicy, self).__init__(
            feature_fn=lambda x: x.long(), *args, **kwargs
        )

        # Initialize with same prob for all actions in each state
        self._policy = nn.Parameter(torch.zeros((n_states, n_actions)))

    def forward(self, x):
        return super(StochasticTablePolicy, self).forward(x)

    def distribution(self, observation):
        logits = self._policy[self.forward(observation)]
        return torch.distributions.Categorical(logits=logits)

    def log_likelihood(self, feat, taken_actions) -> torch.FloatTensor:
        return self.distribution(feat).log_prob(taken_actions)

    @torch.no_grad()
    def sample(self, observation):
        if self._stochastic:
            return self.distribution(observation).sample().item()
        else:
            return torch.argmax(self._policy[self.forward(observation)]).item()


class CategoricalMLP(StochasticPolicy, nn.Module):
    def __init__(self, obs_shape, act_shape, *args, **kwargs):
        super(CategoricalMLP, self).__init__(*args, **kwargs)
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
        return self.model(super(CategoricalMLP, self).forward(x))

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
