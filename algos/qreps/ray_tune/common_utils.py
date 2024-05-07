import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def nll_loss(alpha, observations, next_observations, rewards, actions, log_likes, q_net, policy):
    weights = torch.clamp(q_net.get_values(observations, actions)[0] / alpha, -50, 50)
    _, log_likes, _, _ = policy.get_action(observations, actions)
    nll = -torch.mean(torch.exp(weights.detach()) * log_likes)
    return nll

def kl_loss(alpha, observations, next_observations, rewards, actions, log_likes, q_net, policy):
    _, _, newlogprob, probs = policy.get_action(observations, actions)
    q_values, v = q_net.get_values(observations)
    advantage = q_values - v.unsqueeze(1)
    actor_loss = torch.mean(probs * (alpha * (newlogprob-log_likes.detach()) - advantage.detach()))
    return actor_loss

class Sampler(nn.Module):
    def __init__(self, args, N):
        super().__init__()
        self.n = N
            
        self.z = nn.Sequential(
            (nn.Linear(N, args.sampler_hidden_size)),
            getattr(nn, args.sampler_activation)(),
            *[layer for _ in range(args.sampler_num_hidden_layers) for layer in (
                (nn.Linear(args.sampler_hidden_size, args.sampler_hidden_size)),
                getattr(nn, args.sampler_activation)()
            )],
            (nn.Linear(args.sampler_hidden_size, N)),
        )

    def forward(self, x):
        return self.z(x)

    def get_probs(self, x):
        logits = self(x)
        sampler_dist = Categorical(logits=logits)
        return sampler_dist.probs

class ExponentiatedGradientSampler:
    def __init__(self, N, device, eta, beta=0.01):
        self.n = N
        self.eta = eta
        self.beta = beta

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
        self.h = bellman -  self.eta * torch.log(self.n * self.probs())
        t = self.beta*self.h
        t = torch.clamp(t, -30, 30) # Numerical stability
        self.z = self.probs() * torch.exp(t)
        self.z = torch.clamp(self.z / (torch.sum(self.z)), min=1e-8, max=1.0)
        self.prob_dist = Categorical(self.z)

