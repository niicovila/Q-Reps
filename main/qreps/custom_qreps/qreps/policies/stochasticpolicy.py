from abc import ABCMeta, abstractmethod
from typing import Union

import numpy as np
import torch
import torch.nn as nn


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
