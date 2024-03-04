from abc import ABCMeta, abstractmethod

import torch


class AbstractStateActionFeatureFunction(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, state: torch.Tensor, action: torch.Tensor):
        """Expects both state and action to have [batch_size, feature_dim] shape.
        Batch_size should be identical for both"""
        pass


class AbstractFeatureFunction(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, value: torch.Tensor):
        """Expects features to be always given as [batch_size, feature_dim]"""
        pass
