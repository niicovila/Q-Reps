import torch.nn as nn

from qreps.utilities.nn import (
    accumulate_parameters,
    deep_copy_module,
    freeze_parameters,
)
from qreps.valuefunctions import AbstractQFunction


class AccumulativeModule(AbstractQFunction):
    """Function with accumulation."""

    @property
    def model(self):
        return self.func.model

    def __init__(self, func: AbstractQFunction):
        super().__init__(
            feature_fn=func.feature_fn, obs_dim=func.n_obs, act_dim=func.n_action
        )
        self.func = func
        running_func = deep_copy_module(func)
        freeze_parameters(running_func)
        self.running_func = running_func
        self.count = 0
        for key, value in func.__dict__.items():
            if key not in self.__dict__.keys():
                self.__dict__[key] = value

    def reset(self):
        """Update the value function."""
        accumulate_parameters(self.running_func, self.func, self.count)
        freeze_parameters(self.running_func)
        self.count += 1

    def forward(self, *args, **kwargs):
        """Combine mean with current function."""
        prior = self.count * self.running_func(*args, **kwargs)
        this = self.func(*args, **kwargs)
        return (prior + this) / (self.count + 1)

    def forward_state(self, *args, **kwargs):
        prior = self.count * self.running_func.forward_state(*args, **kwargs)
        this = self.func.forward_state(*args, **kwargs)
        return (prior + this) / (self.count + 1)
