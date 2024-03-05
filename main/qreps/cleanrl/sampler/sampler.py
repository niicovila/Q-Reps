import torch
from torch.distributions import Categorical

class ExponentiatedGradientSampler:
    def __init__(self, N, device, eta, beta=0.1):
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
        exponent = torch.clamp(self.beta*self.h, min=-50, max=50)
        self.z = self.probs() * torch.exp(exponent)
        self.z = torch.clamp(self.z / (torch.sum(self.z)), min=1e-8, max=1.0)
        self.h = (label - pred) -  self.eta * torch.log(self.n * self.probs())
        self.prob_dist = Categorical(probs=self.z)

class BestResponseSampler:
    def __init__(self, N, device, eta):
        self.n = N
        self.eta = eta

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
        bellman = label - pred
        self.z = torch.exp(bellman / self.eta)
        self.z = torch.clamp(self.z / (torch.sum(self.z)), min=1e-8, max=1.0)
        self.prob_dist = Categorical(probs=self.z)
