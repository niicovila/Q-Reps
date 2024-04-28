import torch
import torch.nn as nn

from .stochasticpolicy import StochasticPolicy


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
