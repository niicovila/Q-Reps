# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import itertools
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical


class Policy:
    def __init__(self, q):
        self.q = q

    def set_q(self, q):
        self.q = q
    
    def get_action(self, x, action=None):
        logits = self.q(x)
        prob_dist = Categorical(logits=logits)
        
        if action is None:
            return prob_dist.sample(), prob_dist.probs
        else:
            return action, prob_dist.probs
    
class ExponentiatedGradientSampler:
    def __init__(self, args, N, device, eta, beta=0.01):
        self.n = N
        self.eta = eta
        self.beta = beta
        self.policy_opt = args.policy_opt
        self.h = torch.ones((self.n,)) / N
        self.z = torch.ones((self.n,)) / N

        self.prob_dist = Categorical(torch.ones((self.n,))/ N)
        self.device = device

    def reset(self):
        self.h = torch.ones((self.n,))
        self.z = torch.ones((self.n,))
        self.prob_dist = Categorical(torch.softmax(torch.ones((self.n,)), 0))
                                     
    def probs(self):
        return self.prob_dist.probs.to(self.device)
    
    def entropy(self):
        return self.prob_dist.entropy().to(self.device)
    
    def update(self, bellman):
        self.h = bellman
        if self.policy_opt:
             self.h -=  self.eta * torch.log(self.n * self.probs())
        t = self.beta*self.h
        t = torch.clamp(t, -30, 30) # Numerical stability
        self.z = self.probs() * torch.exp(t)
        self.z = torch.clamp(self.z / (torch.sum(self.z)), min=1e-8, max=1.0)
        self.prob_dist = Categorical(self.z)
