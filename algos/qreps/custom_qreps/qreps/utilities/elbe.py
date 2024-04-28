import torch

from qreps.utilities.math import logmeanexp
from qreps.valuefunctions import AbstractQFunction, AbstractValueFunction


def empirical_bellman_error(
    features,
    features_next,
    actions,
    rewards,
    q_func: AbstractQFunction,
    v_func: AbstractValueFunction,
    discount,
):
    v_features = v_func(features_next)
    q_features = q_func(features, actions)
    output = rewards + discount * v_features - q_features
    return output


def empirical_logistic_bellman(eta, features, features_next, actions, rewards, q_func, v_func, discount):
    errors = 1 / eta * torch.logsumexp(
        eta
        * empirical_bellman_error(
            features, features_next, actions, rewards, q_func, v_func, discount)
            , 
        0
    ) + torch.mean((1 - discount) * v_func(features), 0)
    return errors
