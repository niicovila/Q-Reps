import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs, args):
        super(Agent, self).__init__()
        self.alpha = args.alpha
        self.parametrized = args.parametrized

        self.critic = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, envs.single_action_space.n)
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        q = self.critic(x)
        z = q / self.alpha
        _, _, _, pi_k = self.get_action(x)
        # max_z = torch.max(z, dim=-1, keepdim=True)[0]
        # max_z = torch.where(max_z < -1.0, torch.tensor(-1.0, dtype=torch.float, device=max_z.device), max_z)
        v = self.alpha * (torch.log(torch.sum(pi_k * torch.exp(z), dim=1)))
        return q, v

    def get_action(self, x, action=None):
        if not self.parametrized:
            logits, v = self.get_value(x)
            policy_dist = Categorical(logits=logits)
            log_probs = F.log_softmax(logits, dim=1)
            action_probs = policy_dist.probs
            if action is None:
                action = policy_dist.sample()
            return action, policy_dist.log_prob(action), log_probs, action_probs
        
        else:
            logits = self.actor(x)
            policy_dist = Categorical(logits=logits)
            log_probs = F.log_softmax(logits, dim=1)
            action_probs = policy_dist.probs
            if action is None:
                action = policy_dist.sample()
            return action, policy_dist.log_prob(action), log_probs, action_probs
