import torch

class Sampler:
    
    def __init__(self, length, eta):
        self.length = length
        self.dist = torch.softmax(torch.ones((self.length,)), 0)
        self.eta = eta
    
    def get_next_distribution(self, bellman_error):
        self.dist = torch.exp(self.eta * bellman_error)
        self.dist = self.dist / torch.sum(self.dist)
        return torch.distributions.Categorical(self.dist)
    
    def get_distribution(self):
        return torch.distributions.Categorical(self.dist)
    


class ExponentiatedGradientSampler:
    def __init__(self, length, eta, beta=0.1, *args, **kwargs):
        self.beta = beta
        self.eta = eta
        self.length = length
        self.h = torch.ones((self.length,))
        self.z = torch.ones((self.length,))
        self.z /= torch.sum(self.z)

    def get_next_distribution(self, bellman_error):
        
        self.z = self.z * torch.exp(self.beta * self.h)
        self.z = self.z / (torch.sum(self.z))
        self.h = bellman_error - torch.log(self.length * self.z) / self.eta
        

        return torch.distributions.Categorical(probs=self.z)

    def get_distribution(self):
        return torch.distributions.Categorical(probs=self.z)