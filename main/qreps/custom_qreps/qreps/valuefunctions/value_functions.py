from abc import ABCMeta, abstractmethod

import torch.nn as nn


class AbstractValueFunction(nn.Module, metaclass=ABCMeta):
    def __init__(self, obs_dim):
        super().__init__()
        self.obs_dim = obs_dim

    @abstractmethod
    def forward(self, obs):
        pass


class SimpleValueFunction(AbstractValueFunction):
    def __init__(self, feature_fn=nn.Identity(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_fn = feature_fn
        self.model = nn.Sequential(nn.Linear(self.obs_dim, 1, bias=False))

    def forward(self, obs):
        return self.model(self.feature_fn(obs)).squeeze(-1)


class NNValueFunction(AbstractValueFunction):
    def __init__(self, feature_fn=nn.Identity(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_fn = feature_fn
        self.model = nn.Sequential(
            nn.Linear(self.obs_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 1, bias=False),
        )

    def forward(self, x):
        return self.model(self.feature_fn(x)).squeeze(-1)
