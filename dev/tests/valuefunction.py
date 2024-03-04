import torch
import torch.nn as nn
from qfunction import QFunction, AbstractQFunction
from abc import ABCMeta, abstractmethod


class AbstractValueFunction(nn.Module, metaclass=ABCMeta):
    def __init__(self, obs_dim):
        super().__init__()
        self.obs_dim = obs_dim

    @abstractmethod
    def forward(self, obs):
        pass

class IntegratedQFunction(AbstractValueFunction):
    def __init__(self, q_func: AbstractQFunction, alpha=1.0, *args, **kwargs):
        super().__init__(obs_dim=q_func.n_obs, *args, **kwargs)
        self.alpha = alpha
        self.q_func = q_func

    def forward(self, obs):
        q_values = self.q_func.forward_state(obs)
        values = 1 / self.alpha * torch.logsumexp(self.alpha * q_values, -1)
        return values.squeeze(-1)
    