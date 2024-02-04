import torch

from .stochasticpolicy import StochasticPolicy


class DirectSetPolicy(StochasticPolicy):
    def __init__(self, n_states: int, n_actions: int, *args, **kwargs):
        super(DirectSetPolicy, self).__init__(
            feature_fn=lambda x: x.long(), *args, **kwargs
        )
        # Initialize with same prob for all actions in each state
        _policy = torch.ones((n_states, n_actions))
        _policy /= torch.sum(_policy, 1, keepdim=True)
        self._policy = _policy

    def forward(self, x):
        return super(DirectSetPolicy, self).forward(x)

    def distribution(self, observation):
        return torch.distributions.Categorical(self._policy[self.forward(observation)])

    def log_likelihood(self, feat, taken_actions) -> torch.FloatTensor:
        return self.distribution(feat).log_prob(taken_actions)

    def set_likelihoods(self, feat, actions, weights):
        for s, a, w in zip(feat.long(), actions.long(), weights):
            print(w)
            self._policy[s, a] = torch.exp(w)

    @torch.no_grad()
    def sample(self, observation):
        if self._stochastic:
            return self.distribution(observation).sample().item()
        else:
            return torch.argmax(self._policy[self.forward(observation)]).item()
