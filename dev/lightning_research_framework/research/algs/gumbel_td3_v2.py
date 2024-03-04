import torch
import numpy as np
from functools import partial

from .td3 import TD3

def gumbel_loss(pred, label, beta, clip):
    assert pred.shape == label.shape, "Shapes were incorrect"
    z = (label - pred)/beta
    if clip is not None:
        z = torch.clamp(z, -clip, clip)
    loss = torch.exp(z) - z - 1
    return loss.mean()

def gumbel_log_loss_v1(pred, label, beta, clip):
    assert pred.shape == label.shape, "Shapes were incorrect"
    z = (label - pred)/beta
    if clip is not None:
        z = torch.clamp(z, -clip, clip)
    loss = torch.exp(z)
    loss_2 =  z + 1
    loss = beta * torch.log(loss.mean()) - loss_2.mean()
    return loss

def gumbel_log_loss_v2(pred, label, beta, clip):
    assert pred.shape == label.shape, "Shapes were incorrect"
    z = (label - pred)/beta
    if clip is not None:
        z = torch.clamp(z, -clip, clip)
    loss = torch.exp(z)
    loss_2 =  z + 1
    loss = torch.log(loss.mean()) - loss_2.mean()
    return loss

def gumbel_log_loss_v3(pred, label, beta, clip):
    assert pred.shape == label.shape, "Shapes were incorrect"
    z = (label - pred)/beta
    if clip is not None:
        z = torch.clamp(z, -clip, clip)
    loss = torch.exp(z)
    loss_2 =  z + 1
    loss = loss.mean() - loss_2.mean()
    return torch.log(loss)

def gumbel_rescale_loss(pred, label, beta, clip):
    assert pred.shape == label.shape, "Shapes were incorrect"
    z = (label - pred)/beta
    if clip is not None:
        z = torch.clamp(z, -clip, clip)
    max_z = torch.max(z)
    max_z = torch.where(max_z < -1.0, torch.tensor(-1.0, dtype=torch.float, device=max_z.device), max_z)
    max_z = max_z.detach() # Detach the gradients
    loss = torch.exp(z - max_z) - z*torch.exp(-max_z) - torch.exp(-max_z)    
    return loss.mean()

def gumbel_log_rescale_loss(pred, label, beta, clip):
    assert pred.shape == label.shape, "Shapes were incorrect"
    z = (label - pred)/beta
    if clip is not None:
        z = torch.clamp(z, -clip, clip)
    max_z = torch.max(z)
    max_z = torch.where(max_z < -1.0, torch.tensor(-1.0, dtype=torch.float, device=max_z.device), max_z)
    max_z = max_z.detach() # Detach the gradients
    loss_1 = torch.exp(z - max_z)
    loss_2 =  z*torch.exp(-max_z) + torch.exp(-max_z) 
    loss = torch.log(loss_1.mean())   - loss_2.mean()
    return loss

class GumbelTD3V2(TD3):

    def __init__(self, *args,
                       beta=4.0,
                       loss="mse",
                       exp_clip=10,
                       use_target_actor=True,
                       max_grad_value=None,
                       **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = loss
        self.beta = beta
        self.exp_clip = exp_clip
        self.use_target_actor = use_target_actor
        self.max_grad_value = max_grad_value

    def _update_critic(self, batch):
        with torch.no_grad():
            noise = (torch.randn_like(batch['action']) * self.target_noise).clamp(-self.noise_clip, self.noise_clip)
            net = self.target_network if self.use_target_actor else self.network
            noisy_next_action = (net.actor(batch['next_obs']) + noise).clamp(*self.action_range_tensor)
            target_q = self.target_network.critic(batch['next_obs'], noisy_next_action)
            target_q = torch.min(target_q, dim=0)[0]
            closed_solution_v = (self.beta * torch.log(torch.exp(target_q/self.beta)))

            gap = sum(target_q-closed_solution_v)
            
            target_q = batch['reward'] + batch['discount'] * closed_solution_v

        qs = self.network.critic(batch['obs'], batch['action'])
       
        if self.loss == "gumbel_rescale":
            loss_fn = partial(gumbel_rescale_loss, beta=self.beta, clip=self.exp_clip)
        elif self.loss == "gumbel":
            loss_fn = partial(gumbel_loss, beta=self.beta, clip=self.exp_clip)
        elif self.loss == "gumbel_log_v1":
            loss_fn = partial(gumbel_log_loss_v1, beta=self.beta, clip=self.exp_clip)
        elif self.loss == "gumbel_log_v2":
            loss_fn = partial(gumbel_log_loss_v2, beta=self.beta, clip=self.exp_clip)
        elif self.loss == "gumbel_log_v3":
            loss_fn = partial(gumbel_log_loss_v3, beta=self.beta, clip=self.exp_clip)
        elif self.loss == "mse":
            loss_fn = torch.nn.functional.mse_loss
        else:
            raise ValueError("Incorrect loss specified.")

        q_loss = sum([loss_fn(qs[i], target_q) for i in range(qs.shape[0])])

        self.optim['critic'].zero_grad(set_to_none=True)
        q_loss.backward()
        if self.max_grad_value is not None:
            torch.nn.utils.clip_grad_value_(self.network.critic.parameters(), self.max_grad_value)
        self.optim['critic'].step()

        return dict(q_loss=q_loss.item(), target_q=target_q.mean().item())
