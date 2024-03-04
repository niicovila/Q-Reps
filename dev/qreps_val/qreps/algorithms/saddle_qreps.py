import logging
import math

import torch
from torch import Tensor
from typing_extensions import Type

from qreps.algorithms.sampler import AbstractSampler, ExponentiatedGradientSampler
from qreps.utilities.elbe import empirical_bellman_error
from qreps.valuefunctions import IntegratedQFunction, SimpleQFunction

from ..valuefunctions.AccumulativeModule import AccumulativeModule
from .abstract_algorithm import AbstractAlgorithm

logger = logging.getLogger("qreps")
logger.addHandler(logging.NullHandler())


class SaddleQREPS(AbstractAlgorithm):
    r"""Logistic Q-Learning Algorithm.

        Logistic Q-Learning optimizes a linear program of the optimal policy but uses p = d as trick to naturally
        introduce Q-functions.
        The following linear program is optimized:
        math:: \mathrm{maximize} \langle p, r \rangle - \frac{1}{\eta} D(p || p_0)  - \frac{1}{\alpha} H(d || d_0)
        math:: \text{s.t. }E^T d = \gamma P^T p + ( 1 - \gamma) \vu_0
        math:: \phi^T d = \phi^T p

        Through the dual we naturally introduce Q and V functions.
        The paper shows that the optimal solution for theses solves the linear program as well.

        We can get the optimal solution V and Q, by minimizing empirical logistic bellman error (ELBE)
        math:: \hat G (\theta) = \frac{1}{\eta} \log \left(\frac{1}{N}
            \sum_{n=1}^N e^{\eta \hat \nabla_\theta (\xi_{k,n} }\right) + (1 - \gamma) \langle \vu_0 , V_\theta \rangle

        This algorithm implement the version proposed in the paper as MinMaxQREPS.
        Therefore, this uses a mirror descent algorithm formulated as sampler and learner.

        References
        ----------
        [1] J. Bas-Serrano, S. Curi, A. Krause, and G. Neu, “Logistic $Q$-Learning,”
        arXiv:2010.11151 [cs, stat], Oct. 2020, Accessed: Nov. 11, 2020. [Online].
        Available: http://arxiv.org/abs/2010.11151.
        """

    def __init__(
        self,
        q_function: SimpleQFunction,
        saddle_point_steps: int = 300,
        beta: float = 0.1,
        eta: float = 0.01,
        alpha: float = None,
        learner: Type[torch.optim.Optimizer] = torch.optim.SGD,
        sampler: Type[AbstractSampler] = ExponentiatedGradientSampler,
        sampler_args: dict = None,
        optimize_policy: bool = True,
        *args,
        **kwargs,
    ):
        """
        This creates an QREPS algorithm instance.

        @param q_function: The q_function that should be used. Currently only the SimpleQFunction is supported.
        @param policy: The policy that is used for the problem and should be optimized. Only discrete policies currently
        @param saddle_point_steps: How many optimization steps should be done for the learner/sampler optimization.
        @param beta: Learning rate for the q_function parameters.
        @param eta: The entropy regularization parameter.
        @param learner: Which learner should be used. The default recommendations are SGD and Adam.
        @param sampler: Which sampler to use. This is highly problem dependent.
        @param sampler_args: Args that should be provided when creating the sampler.
        @param average_weights: Enable averaging the parameters after the optimization
        @param args: arguments for the abstract algorithm.
        @param kwargs: keyword arguments for the abstract algorithm.
        """
        super().__init__(*args, **kwargs)
        self.saddle_point_steps = saddle_point_steps
        self.eta = eta
        self.alpha = alpha if alpha is not None else eta
        self.q_function = AccumulativeModule(q_function)
        self.value_function = IntegratedQFunction(self.q_function, alpha=self.alpha)
        self.theta_opt = learner(self.q_function.parameters(), lr=beta)

        self.optimize_policy = optimize_policy
        self.sampler = sampler
        self.sampler_args = sampler_args if sampler_args is not None else {}

    def calc_weights(
        self, features: Tensor, features_next: Tensor, rewards: Tensor, actions: Tensor
    ) -> Tensor:
        """
        Calculate the weights from the advantage for updating the policy

        @param actions: the taken actions
        @param features: batched features for the states [N, feature_dim]
        @param features_next: batches features for the following states (e.g. features[1:]) [N, feature_dim]
        @param rewards: undiscounted rewards received in the states [N]
        @return: Tuple of the weights, calculated advantages
        """
        return torch.clamp(self.alpha * self.q_function(features, actions), -50, 50)

    def saddle_point_optimization(
        self, observations, next_observations, rewards, actions
    ):
        N = observations.shape[0]
        sampler = self.sampler(length=N, eta=self.eta, *self.sampler_args)
        z = sampler.get_distribution()
        for i in range(self.saddle_point_steps):
            self.theta_opt.zero_grad()
            bellman = empirical_bellman_error(
                observations,
                next_observations,
                actions,
                rewards,
                q_func=self.q_function,
                v_func=self.value_function,
                discount=self.discount,
            )
            S_k = torch.sum(
                z.probs.detach()
                * (bellman - 1 / self.eta * torch.log(N * z.probs.detach()))
                + (1 - self.discount) * self.value_function(observations)
            )
            S_k.backward()
            self.theta_opt.step()
            with torch.no_grad():
                z = sampler.get_next_distribution(bellman.detach())

        return S_k

    def update_policy(self, iteration):
        (
            observations,
            actions,
            rewards,
            next_observations,
        ) = self.buffer.get_all()

        rewards = self.get_rewards(rewards)
        dist_before = self.policy.distribution(observations)

        qreps_loss = self.saddle_point_optimization(
            observations, next_observations, rewards, actions
        )
        if hasattr(self.q_function, "reset"):
            self.q_function.reset()

        if self.optimize_policy is True:
            self.optimize_loss(
                self.nll_loss, self.pol_optimizer, optimizer_steps=self.policy_opt_steps
            )
        self.buffer.reset()
        dist_after = self.policy.distribution(observations)
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
            # self.writer.add_histogram(
            #     "train/q_weights", self.q_function.model.weight.data, iteration
            # )
