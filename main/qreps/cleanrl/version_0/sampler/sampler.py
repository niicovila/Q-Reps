import torch
from torch.distributions import Categorical

class ExponentiatedGradientSampler:
    def __init__(self, N, device, eta, beta=0.01):
        self.n = N
        self.eta = eta
        self.beta = beta

        self.h = torch.ones((self.n,))
        self.z = torch.ones((self.n,))

        self.prob_dist = Categorical(torch.softmax(torch.ones((self.n,)), 0))
        self.device = device

    def reset(self):
        self.h = torch.ones((self.n,))
        self.z = torch.ones((self.n,))
        self.prob_dist = Categorical(torch.softmax(torch.ones((self.n,)), 0))
                                     
    def probs(self):
        return self.prob_dist.probs.to(self.device)
    
    def entropy(self):
        return self.prob_dist.entropy().to(self.device)
    
    def update(self, pred, label):
        self.h = (label - pred) -  self.eta * torch.log(self.n * self.probs())
        t = torch.clamp(self.beta*self.h, -10, 10)
        self.z = self.probs() * torch.exp(t)
        self.z = torch.clamp(self.z / (torch.sum(self.z)), min=1e-8, max=1.0)
        self.prob_dist = Categorical(self.z)

class BestResponseSampler:
    def __init__(self, N, device, eta):
        self.n = N
        self.eta = eta
        self.z = torch.ones((self.n,)) / N  # Initialize z randomly
        self.prob_dist = Categorical(torch.softmax(self.z, 0))
        self.device = device

    def reset(self):
        self.z = torch.randn((self.n,)) / self.n  # Initialize z randomly
        self.prob_dist = Categorical(torch.softmax(self.z, 0))    
                                     
    def probs(self):
        return self.prob_dist.probs.to(self.device)
    
    def entropy(self):
        return self.prob_dist.entropy().to(self.device)
    
    def update(self, pred, label):
        bellman = label - pred
        self.z = torch.exp(bellman / self.eta)
        self.z = torch.clamp(self.z / (torch.sum(self.z) + 1e-8), min=1e-8, max=1.0)
        self.prob_dist = Categorical(probs=self.z)
