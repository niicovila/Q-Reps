import torch

from qreps.algorithms.sampler.abstract_sampler import AbstractSampler


class BestResponseSampler(AbstractSampler):
    """
    Best Response Sampler that directly gives a distribution according to the bellman errors.
    Useful for problems where the exploration space is huge.

    Used in Logistic Q-Learning for CartPole.
    """

    def __init__(self, *args, **kwargs):
        super(BestResponseSampler, self).__init__(*args, **kwargs)
        self.dist = torch.softmax(torch.ones((self.length,)), 0)

    def get_next_distribution(self, bellman):
        self.dist = torch.clip(torch.exp(self.eta * bellman), -20, 20)
        self.dist = self.dist / torch.sum(self.dist)
        return torch.distributions.Categorical(self.dist)

    def get_distribution(self):
        return torch.distributions.Categorical(self.dist)
