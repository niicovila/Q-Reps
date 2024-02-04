import torch

from qreps.policies import StochasticPolicy
from qreps.utilities.util import integrate

from .q_function import AbstractQFunction
from .value_functions import AbstractValueFunction


class IntegratedQFunction(AbstractValueFunction):
    def __init__(self, q_func: AbstractQFunction, alpha=1.0, *args, **kwargs):
        super().__init__(obs_dim=q_func.n_obs, *args, **kwargs)
        self.alpha = alpha
        self.q_func = q_func

    def forward(self, obs):
        q_values = self.q_func.forward_state(obs)
        values = 1 / self.alpha * torch.logsumexp(self.alpha * q_values, -1)
        return values.squeeze(-1)
