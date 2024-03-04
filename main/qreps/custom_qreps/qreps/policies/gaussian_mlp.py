import torch
import torch.nn as nn

from .stochasticpolicy import StochasticPolicy


class GaussianMLPStochasticPolicy(StochasticPolicy, nn.Module):
    """Gaussian Multi Layer Perceptron as a Policy.

    Estimates mean of a gaussian distribution for every action and the corresponding deviation
    depending on the given observations.

    When set to eval mode returns the mean as action for every observation.
    """

    def __init__(self, obs_shape, act_shape, sigma=1.0, *args, **kwargs):
        super(GaussianMLPStochasticPolicy, self).__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Linear(obs_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_shape),
        )
        self.log_sigma = torch.log(torch.tensor(sigma))

    def forward(self, x):
        return self.model(super(GaussianMLPStochasticPolicy, self).forward(x))

    @property
    def sigma(self):
        return torch.exp(self.log_sigma)

    def distribution(self, observation) -> torch.distributions.Distribution:
        return torch.distributions.normal.Normal(self.forward(observation), self.sigma)

    @torch.no_grad()
    def sample(self, observation):
        if self._stochastic:
            return torch.tanh(self.distribution(observation).sample())
        else:
            return torch.tanh(self.forward(observation))

    def log_likelihood(self, feat, taken_actions) -> torch.FloatTensor:
        return self.distribution(feat).log_prob(taken_actions)
