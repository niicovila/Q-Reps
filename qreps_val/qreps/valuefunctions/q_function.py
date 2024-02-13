from abc import ABCMeta, abstractmethod
from functools import partial
import torch.nn as nn
import math, torch
import torch.nn as nn

from qreps.feature_functions import (
    AbstractFeatureFunction,
    AbstractStateActionFeatureFunction,
)
from qreps.feature_functions.identity import IdentityFeature

from torch.nn import AvgPool2d, Conv2d
import numpy as np


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

    def features(self, observation, action=None):
        if action is not None:
            return self.feature_fn(observation, action)
        return self.feature_fn(observation)

    def forward_state(self, observation):
        if self.n_obs == 1:
            observation = observation.reshape(observation.shape[0], 1)
        if len(observation.shape) == 3:
            observation = observation.reshape(1, observation.shape[2], observation.shape[0], observation.shape[1])
        elif len(observation.shape) == 4:
            observation = observation.reshape(observation.shape[0], observation.shape[3], observation.shape[1], observation.shape[2])
        input = self.features(observation)
        return self.model(input)

    @property
    @abstractmethod
    def model(self):
        pass


class Block(nn.Module):
  def __init__(self, in_channels, out, kern, stride):
        super(Block, self).__init__()
        self.block = nn.Sequential( Conv2d(in_channels=in_channels, out_channels=out, kernel_size=kern, stride=stride, padding=1), 
                                   nn.BatchNorm2d(out),
                                   nn.ELU())
  def forward(self, input):
    return self.block(input)
  
class ResnetModule(nn.Module):
    def __init__(self, n_action, k = 1):
        super().__init__()
        self.n_action = n_action
        ### Stage 1:
        self.k = k
        self.conv1=[Block(3, 4, 3, 1)]
        for i in range(1, int(k/3)):
            self.conv1.append(Block(4, 4, 3, 1))
        self.avgpool1 = AvgPool2d(2, 2)
        self.conv1 = nn.Sequential(*self.conv1)

        #### Stage 2:
        self.conv2=[Block(4, 8, 3, 1)]
        for i in range(1, int(k/3)):
            self.conv2.append(Block(8, 8, 3, 1))
        self.avgpool2 = AvgPool2d(2 ,2)
        self.conv2 = nn.Sequential(*self.conv2)

        #### Stage 3:
        self.conv3=[Block(8, 16, 3, 1)]
        for i in range(1, int(k/3)):
          self.conv3.append(Block(16, 16, 3, 1))
        self.avgpool3 = AvgPool2d(2, 2)
        self.conv3=nn.Sequential(*self.conv3)

        self.feedforward= nn.Linear(8320, self.n_action)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                sigma = 1/np.sqrt(3*m.weight.shape[1])
                m.weight.data.normal_(0, sigma)
                m.bias.data.zero_()


    def forward(self, input):
            u = self.conv1[0](input)
            res = u #set residual to 0
            for i in range(1, int(self.k/3)):
                u = self.conv1[i](u)+res
                res = u 
            u = self.avgpool1(u)

            #### Stage 2
            u = self.conv2[0](u)
            res = u
            for i in range(1, int(self.k/3)):
                u = self.conv2[i](u)+res
                res = u
            u = self.avgpool2(u)

            #Stage 3
            u = self.conv3[0](u)
            res = u
            for i in range(1, int(self.k/3)):
                u = self.conv3[i](u)+res
                res = u
            u = self.avgpool3(u)
            u = u.reshape(u.shape[0],-1)
            
            u = self.feedforward(u)
            return u



class ResNet(AbstractQFunction):
    def __init__(self, observation_space, action_space, feature_fn, k = 1):
        super().__init__(observation_space, action_space, feature_fn)
        self.resnet = ResnetModule(action_space, k = 1) 
    
    @property
    def model(self):
        return self.resnet
    
    def to(self, device):
        self.resnet.to(device)
        return self
        
class LinearEnsemble(nn.Module):

    def __init__(self, in_features, out_features, bias=True, ensemble_size=3, device=None, dtype=None):
        '''
        An Ensemble linear layer.
        For inputs of shape (B, H) will return (E, B, H) where E is the ensemble size
        See https://github.com/pytorch/pytorch/issues/54147
        '''
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = torch.empty((ensemble_size, in_features, out_features), **factory_kwargs)
        if bias:
            self.bias = torch.empty((ensemble_size, 1, out_features), **factory_kwargs)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Hack to make sure initialization is correct. This shouldn't be too bad though
        for w in self.weight:
            w.transpose_(0, 1)
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            w.transpose_(0, 1)
        self.weight = nn.Parameter(self.weight)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0].T)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            self.bias = nn.Parameter(self.bias)

    def forward(self, input):
        if len(input.shape) == 2:
            input = input.repeat(self.ensemble_size, 1, 1)
        elif len(input.shape) > 3:
            raise ValueError("LinearEnsemble layer does not support inputs with more than 3 dimensions.")
        return torch.baddbmm(self.bias, input, self.weight)

    def extra_repr(self) -> str:
        return 'ensemble_size={}, in_features={}, out_features={}, bias={}'.format(
            self.ensemble_size, self.in_features, self.out_features, self.bias is not None
        )



def weight_init(m, gain=1):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    if isinstance(m, LinearEnsemble):
        for i in range(m.ensemble_size):
            # Orthogonal initialization doesn't care about which axis is first
            # Thus, we can just use ortho init as normal on each matrix.
            nn.init.orthogonal_(m.weight.data[i], gain=gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)





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

class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_layers=[256, 256], act=nn.ReLU, output_act=None):
        super().__init__()
        net = []
        last_dim = input_dim
        for dim in hidden_layers:
            net.append(nn.Linear(last_dim, dim))
            net.append(act())
            last_dim = dim
        net.append(nn.Linear(last_dim, output_dim))
        if not output_act is None:
            net.append(output_act())
        self.net = nn.Sequential(*net)
        self._has_output_act = False if output_act is None else True

    @property
    def last_layer(self):
        if self._has_output_act:
            return self.net[-2]
        else:
            return self.net[-1]

    def forward(self, x):
        return self.net(x)

class DiscreteMLPCritic(AbstractQFunction):

    def __init__(self, observation_space, action_space, hidden_layers=[256, 256], act=nn.ReLU, ortho_init=True, output_gain=None, *args, **kwargs):
        super().__init__(observation_space, action_space, *args, **kwargs)
        self.q = MLP(observation_space, action_space, hidden_layers=hidden_layers, act=act)
        if ortho_init:
            self.apply(partial(weight_init, gain=float(ortho_init))) # use the fact that True converts to 1.0
            if output_gain is not None:
                self.mlp.last_layer.apply(partial(weight_init, gain=output_gain))
    
    @property
    def model(self):
        return self.q

    def predict(self, obs):
        q = self(obs)
        action = q.argmax(dim=-1)
        return action

