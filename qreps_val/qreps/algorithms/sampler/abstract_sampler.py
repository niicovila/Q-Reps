from abc import ABCMeta, abstractmethod

import torch


class AbstractSampler(metaclass=ABCMeta):
    def __init__(self, length, eta):
        self.length = length
        self.eta = eta

    @abstractmethod
    def get_next_distribution(
        self, bellman_error: torch.Tensor
    ) -> torch.distributions.Categorical:
        pass

    @abstractmethod
    def get_distribution(self) -> torch.distributions.Categorical:
        pass
