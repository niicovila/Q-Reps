import torch
from torch.distributions import Categorical

class Sampler:
    def __init__(self, N, device, eta, beta=0.0001):
        self.n = N
        self.eta = eta
        self.beta = beta

        self.h = torch.ones((self.n,))
        self.z = torch.ones((self.n,))

        self.prob_dist = Categorical(torch.softmax(torch.ones((self.n,)), 0))
        self.device = device

    def probs(self):
        return self.prob_dist.probs.to(self.device)
    
    def entropy(self):
        return self.prob_dist.entropy().to(self.device)
    
    def update(self, pred, label):
        self.z = self.probs() * torch.clamp(torch.exp(self.beta*self.h), -50, 50)
        self.z = torch.clamp(self.z / (torch.sum(self.z)), min=1e-8, max=1.0)
        self.h = (label - pred) - torch.log(self.n * self.probs()) / self.eta
        self.prob_dist = Categorical(probs=self.z)
