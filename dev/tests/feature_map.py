import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod

class FeatureMap(nn.Module):
    def __init__(self, obs_dimension, feature_map_dimension, *args, **kwargs) -> None:
        self.dimension = feature_map_dimension
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
                    nn.Linear(obs_dimension, 200), nn.ReLU(), nn.Linear(200, feature_map_dimension), nn.ReLU(),
        )

        self.model.requires_grad_(False)

    def forward(self, x):
        return nn.Identity()(x)


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

class OneHotFeature(AbstractFeatureFunction):
    def __init__(self, num_classes: int):
        super(OneHotFeature, self).__init__()
        self.num_classes = num_classes

    def __call__(self, value):
        super(OneHotFeature, self).__call__(value)
        if value.ndim >= 1 and value.shape[-1] == 1:
            value = value.squeeze(-1)
        return F.one_hot(value.long(), num_classes=self.num_classes).float()


