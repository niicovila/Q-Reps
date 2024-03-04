import numpy as np
import torch
import torch.nn as nn

from .stochasticpolicy import StochasticPolicy

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class CategoricalMLP(StochasticPolicy, nn.Module):
    def __init__(self, obs_shape, act_shape, *args, **kwargs):
        super(CategoricalMLP, self).__init__(*args, **kwargs)
        self.model = nn.Sequential(
            (nn.Linear(obs_shape, 256)),
            nn.ReLU(),
            (nn.Linear(256, 256)),
            nn.ReLU(),
            (nn.Linear(256, 256)),
            nn.ReLU(),
            nn.Linear(256, act_shape),
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
