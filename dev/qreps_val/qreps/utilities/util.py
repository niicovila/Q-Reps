import random
from typing import Callable

import numpy as np
import torch


def to_torch(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.tensor(x).float()


def integrate_discrete(func, distribution: torch.distributions.Distribution):
    values = torch.zeros(distribution.batch_shape)
    for action in distribution.enumerate_support():
        f_val = func(action)
        assert f_val.ndim <= values.ndim
        if f_val.ndim < values.ndim:
            f_val = f_val.unsqueeze(-1)
        log_probs = distribution.log_prob(action)
        values += f_val * torch.exp(log_probs.detach())
    return values


def integrate_continuous(
    func: Callable, distribution: torch.distributions.Distribution, samples=15
):
    values = torch.zeros(distribution.batch_shape)
    for _ in range(samples):
        if distribution.has_rsample:
            action = distribution.rsample().detach()
        else:
            action = distribution.sample().detach()
        f_val = func(action)
        if f_val.ndim > values.ndim:
            f_val = f_val.squeeze(-1)
        values += f_val
    return values / samples


def integrate(
    func: Callable,
    distribution: torch.distributions.Distribution,
    continuous_samples=15,
):
    if distribution.has_enumerate_support:
        return integrate_discrete(func, distribution)
    else:
        return integrate_continuous(func, distribution, continuous_samples)


def episode_normalize_rewards(rewards):
    return (rewards - rewards.mean(0)) / rewards.std()


def torch_batched(x: torch.Tensor):
    if len(x.shape) >= 2:
        return x

    if len(x.shape) == 0:
        return x.view(-1, 1)

    return x.view(-1, x.shape[0])


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
