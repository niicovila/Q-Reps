import torch
import torch.nn as nn
from feature_map import FeatureMap, AbstractFeatureFunction
from abc import ABCMeta, abstractmethod


class QFunction (nn.Module):
    def __init__(self, feature_map: FeatureMap, feature_map_dimension, action_space_dim ,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.feature_map = feature_map
        self.fwd = nn.Linear(feature_map_dimension, action_space_dim)

    def forward(self, x):
        feat = self.feature_map.forward(x)
        return self.fwd(feat)



class AbstractQFunction(nn.Module, metaclass=ABCMeta):
    def __init__(
        self, obs_dim, act_dim, feature_fn: AbstractFeatureFunction, *args, **kwargs
    ):
        super().__init__()
        self.feature_fn = feature_fn
        self.n_obs = obs_dim
        self.n_action = act_dim

    def forward(self, observation, action):
        model_output = self.forward_state(observation)
        return model_output.gather(-1, action.unsqueeze(-1).long()).squeeze(-1)

    def features(self, observation):
        return self.feature_fn(observation)

    def forward_state(self, observation):
        input = self.features(observation)
        return self.model(input)

    @property
    @abstractmethod
    def model(self):
        pass

class SimpleQFunction(AbstractQFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = nn.Linear(self.n_obs, self.n_action, bias=False)

    @property
    def model(self):
        return self._model

class NNQFunction(AbstractQFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = nn.Sequential(
            nn.Linear(self.n_obs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_action),
        )

    @property
    def model(self):
        return self._model

