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
        eta: float = 0.5,
        alpha: float = None,
        learner: Type[torch.optim.Optimizer] = torch.optim.SGD,
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
        self.q_function = q_function
        self.value_function = IntegratedQFunction(self.q_function, alpha=self.alpha)
        self.theta_opt = learner(self.q_function.parameters(), lr=beta)

        self.optimize_policy = optimize_policy


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
        # Clamping added for stability. Could potentially blow up the policy otherwise
        return torch.clamp(self.alpha * self.q_function(features, actions), -20, 20)

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
        dist_before = self.policy.distribution(observations)

        self.optimize_loss(self.dual, optimizer=self.theta_opt)
        qreps_loss = self.dual(observations, next_observations, rewards, actions)

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
            dist_before,
            dist_after,
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
