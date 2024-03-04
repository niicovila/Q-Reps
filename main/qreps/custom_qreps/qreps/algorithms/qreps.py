import logging
import math

import torch
from torch import Tensor
from typing_extensions import Type

from qreps.algorithms.sampler import AbstractSampler, ExponentiatedGradientSampler
from qreps.utilities.elbe import empirical_logistic_bellman
from qreps.valuefunctions import IntegratedQFunction, SimpleQFunction

from .abstract_algorithm import AbstractAlgorithm

logger = logging.getLogger("qreps")
logger.addHandler(logging.NullHandler())


class QREPS(AbstractAlgorithm):

    def __init__(
        self,
        q_function: SimpleQFunction,
        saddle_point_steps: int = 300,
        beta: float = 0.1,
        eta: float = 0.5,
        alpha: float = None,
        learner: Type[torch.optim.Optimizer] = torch.optim.SGD,
        optimize_policy: bool = False,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.saddle_point_steps = saddle_point_steps
        self.eta = eta
        self.alpha = alpha if alpha is not None else eta
        self.q_function = q_function
        self.value_function = IntegratedQFunction(self.q_function, alpha=self.alpha)
        self.theta_opt = learner(self.q_function.parameters(), lr=beta)

        self.optimize_policy = optimize_policy


    def calc_weights(
        self, features: Tensor, features_next: Tensor, rewards: Tensor, actions: Tensor
    ) -> Tensor:
        # Clamping added for stability. Could potentially blow up the policy otherwise
        return torch.clamp(self.alpha * self.q_function(features, actions), -50, 50)

    def dual(self, observations, next_observations, rewards, actions):
        return empirical_logistic_bellman(
            eta=self.eta,
            features=observations,
            features_next=next_observations,
            actions=actions,
            rewards=rewards,
            q_func=self.q_function,
            v_func=self.value_function,
            discount=self.discount,
        )

    def update_policy(self, iteration):
        (
            observations,
            actions,
            rewards,
            next_observations,
        ) = self.buffer.get_all()

        observations = observations.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)   
        next_observations = next_observations.to(self.device)
        rewards = self.get_rewards(rewards)

        self.optimize_loss(self.dual, optimizer=self.theta_opt)
        qreps_loss = self.dual(observations, next_observations, rewards, actions)

        if self.optimize_policy is True:
            self.optimize_loss(
                self.nll_loss, self.pol_optimizer, optimizer_steps=self.policy_opt_steps
            )

        self.buffer.reset()

        self.report_tensorboard(
            observations,
            next_observations,
            rewards,
            actions,
            iteration,
        )
        if self.writer is not None:
            self.writer.add_scalar("train/qreps_loss", qreps_loss, iteration)
            self.writer.add_scalar(
                "train/q_values",
                self.q_function(observations, actions).mean(0),
                iteration,
            )
            self.writer.add_scalar(
                "train/next_v_function",
                self.value_function(next_observations).mean(0),
                iteration,
            )
