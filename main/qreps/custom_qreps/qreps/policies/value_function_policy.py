import random
from typing import Union

import numpy as np
import torch

from .stochasticpolicy import StochasticPolicy


class ValueFunctionPolicy(StochasticPolicy):
    def __init__(
        self, n_actions, value_function, transition_model, eps, *args, **kwargs
    ):
        super(ValueFunctionPolicy, self).__init__(*args, **kwargs)
        self.value_function = value_function
        self.transition_model = transition_model
        self.eps = eps
        self.n_actions = n_actions

    def optimal_action(self, observation):
        states = self.transition_model[observation.long()]
        values = self.value_function(states)
        action = torch.argmax(values)
        return action.item()

    @torch.no_grad()
    def sample(self, observation: torch.Tensor) -> Union[int, np.array]:
        if random.random() < self.eps and self._stochastic is True:
            action = torch.tensor(random.randint(0, self.n_actions - 1))
            return action.item()
        else:
            return self.optimal_action(observation)

    def log_likelihood(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        pass

    def distribution(self, x):
        pass
