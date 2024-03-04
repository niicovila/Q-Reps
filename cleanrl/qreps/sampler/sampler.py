import torch
from torch.distributions import Categorical

class Sampler:
    def __init__(self, N, device, eta, beta=0.1):
        self.n = N
        self.eta = eta
        self.beta = beta

        self.h = torch.rand(self.n) * 0.1
        self.z = torch.rand(self.n) * 0.1

        self.prob_dist = Categorical(torch.softmax(torch.ones((self.n,)), 0))
        self.device = device

    def probs(self):
        return self.prob_dist.probs.to(self.device)
    
    def entropy(self):
        return self.prob_dist.entropy().to(self.device)
    
    def update(self, pred, label):
        # Calculate z with numerical stability
        
        self.z = self.probs() * torch.exp(self.beta * self.h ) # - torch.max(self.beta * self.h)
        self.z = self.z / (torch.sum(self.z) + 1e-8)
        
        # Update h
        bellman = torch.clamp(label - pred, -20, 20)
        self.h = bellman - torch.log(self.n * self.probs() + 1e-8) / self.eta
        # print(label)
        
        # Update the probability distribution
        self.prob_dist = Categorical(probs=self.z)
