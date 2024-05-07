import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.categorical import Categorical


def layer_init(layer, args, gain_ort=np.sqrt(2), bias_const=0.0, gain=1):

    if args.layer_init == "orthogonal_gain":
        torch.nn.init.orthogonal_(layer.weight, gain_ort)
    elif args.layer_init == "orthogonal":
        torch.nn.init.orthogonal_(layer.weight, gain)

    elif args.layer_init == "xavier_normal":
        torch.nn.init.xavier_normal_(layer.weight, gain)
    elif args.layer_init == "xavier_uniform":
        torch.nn.init.xavier_uniform_(layer.weight, gain)

    elif args.layer_init == "kaiming_normal":
        torch.nn.init.kaiming_normal_(layer.weight)
    elif args.layer_init == "kaiming_uniform":
        torch.nn.init.kaiming_uniform_(layer.weight)
    else:
        pass
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class QREPSAgent(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.env = env
        self.alpha = args.alpha
        self.use_policy = args.use_policy

        def init_layer(layer, gain_ort=np.sqrt(2), gain=1):
            if args.layer_init == "default":
                return layer
            else:
                return layer_init(layer, args, gain_ort=gain_ort, gain=gain)

        self.network = nn.Sequential(
            init_layer(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            init_layer(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_layer(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            init_layer(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.critic = nn.Sequential(
            self.network,
            init_layer(nn.Linear(512, args.q_hidden_size)),
            getattr(nn, args.q_activation)(),
            *[layer for _ in range(args.q_num_hidden_layers) for layer in (
                init_layer(nn.Linear(args.q_hidden_size, args.q_hidden_size)),
                getattr(nn, args.q_activation)()
            )],
            init_layer(nn.Linear(args.q_hidden_size, env.single_action_space.n), gain_ort=args.q_last_layer_std, gain=args.q_last_layer_std),
        )
        self.actor = nn.Sequential(
            self.network,
            init_layer(nn.Linear(512, args.hidden_size)),
            getattr(nn, args.policy_activation)(),
            *[layer for _ in range(args.num_hidden_layers) for layer in (
                init_layer(nn.Linear(args.hidden_size, args.hidden_size)),
                getattr(nn, args.policy_activation)()
            )],
            init_layer(nn.Linear(args.hidden_size, env.single_action_space.n), gain_ort=args.actor_last_layer_std, gain=args.actor_last_layer_std),
        )

    def forward(self, x):
        return self.critic(x / 255.0)
    
    def get_values(self, x, action=None):
        q = self(x)
        z = q / self.alpha
        if self.use_policy:
            _, _, _, pi_k = self.get_action(x); pi_k = pi_k.detach()
            v = self.alpha * (torch.log(torch.sum(pi_k * torch.exp(z), dim=1))).squeeze(-1)
        else:
            v = self.alpha * torch.log(torch.mean(torch.exp(z), dim=1)).squeeze(-1)

        if action is None:
            return q, v
        else:
            q = q.gather(-1, action.unsqueeze(-1).long()).squeeze(-1)
            return q, v
        
    def get_action(self, x, action=None):
        logits = self.actor(x)
        policy_dist = Categorical(logits=logits)
        if action is None: action = policy_dist.sample()
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        action_log_prob = policy_dist.log_prob(action)
        return action, action_log_prob, log_prob, action_probs

    