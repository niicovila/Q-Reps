import logging
from abc import ABCMeta, abstractmethod
from typing import Callable, Type, Union

import dm_env
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from qreps.memory.replay_buffer import ReplayBuffer
from qreps.policies import StochasticPolicy

DEFAULT_REPLAY_BUFFER_SIZE = 100000

logger = logging.getLogger("algorithm")
logger.addHandler(logging.NullHandler())


class AbstractAlgorithm(nn.Module, metaclass=ABCMeta):
    """
    Abstract Algorithm provides general properties and configurations which are useful for all algorithms.
    """

    def __init__(
        self,
        device,
        policy: StochasticPolicy,
        writer: SummaryWriter = None,
        buffer=None,
        reward_transformer=None,
        discount=1.0,
        policy_optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        policy_opt_steps=150,
        policy_lr=1e-2,
        *args,
        **kwargs,
    ):
        super().__init__()
        # Organizational setup
        self.writer = writer
        if buffer is None:
            self.buffer = ReplayBuffer(DEFAULT_REPLAY_BUFFER_SIZE)
        else:
            self.buffer = buffer

        # Default MDP attributes
        self.reward_transformer = reward_transformer
        self.discount = discount

        # Policy setup
        self.device = device
        self.policy = policy
        self.policy_opt_steps = policy_opt_steps
        self.pol_optimizer = policy_optimizer(self.policy.parameters(), lr=policy_lr)

    def get_rewards(self, rewards):
        """
        Get the transformed rewards for the given rewards.

        Applies the specified transformers.
        @param rewards: rewards to transform
        @return: the transformed rewards
        """
        if self.reward_transformer is not None:
            return self.reward_transformer(rewards)
        else:
            return rewards

    def select_action(self, obs) -> Union[int, np.array]:
        """
        Selects actions using current policy in an on-policy setting.

        @param timestep: the current timestep containg the observation
        @return: An action confirming to the dm_control numpy or int convention
        """
        # if isinstance(timestep.observation, tuple) : obs_feat = torch.tensor([timestep.observation[0]]).float()
        # else: obs_feat = torch.tensor([timestep.observation]).float()
        action = self.policy.sample(torch.Tensor(obs).to(self.device))
        return action

    @abstractmethod
    def calc_weights(
        self, features: Tensor, features_next: Tensor, rewards: Tensor, actions: Tensor
    ) -> Tensor:
        pass

    def report_tensorboard(
        self,
        observations,
        next_observations,
        rewards,
        actions,
        iteration,
    ):
        pol_loss = self.nll_loss(observations, next_observations, rewards, actions)
        # kl_loss = torch.distributions.kl_divergence(dist_after, dist_before).mean(0)
        entropy = self.policy.distribution(observations).entropy().mean(0)
        weights = self.calc_weights(observations, next_observations, rewards, actions)
        weights = weights - torch.max(weights)

        logger.info(
            f"Iteration {iteration} done, " f"Reward {torch.sum(rewards).item():.2f}"
        )

        if self.writer is not None:
            self.writer.add_scalar("train/pol_loss", pol_loss, iteration)
            self.writer.add_scalar("train/reward", torch.sum(rewards), iteration)
            self.writer.add_histogram("train/actions", actions, iteration)
            # self.writer.add_scalar("train/kl_loss_mean", kl_loss, iteration)
            self.writer.add_scalar("train/entropy_mean", entropy, iteration)
            self.writer.add_histogram("train/weights", weights, iteration)
            self.writer.add_histogram("train/reward_hist", rewards, iteration)
            self.writer.add_scalar("train/num_obs", observations.shape[0], iteration)

    def update(
        self, obs, new_observation, action, reward, done):
        """Sampling: Obtain N samples (s_i, a_i, s_i', r_i)

        Currently done using the Agent interface of DM.
        """
        self.buffer.push(obs, new_observation, action, reward, done)

    def optimize_loss(
        self,
        loss_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ],
        optimizer: torch.optim.Optimizer,
        optimizer_steps=300,
    ):
        """
        Optimize the specified loss using batch gradient descent.

        Allows to specify an optimizer and is compatible with L-BFGS, Adam and SGD.

        @param loss_fn: the loss function which should be minimized.
        @param optimizer: the torch optimizer to use
        @param optimizer_steps: how many steps to do the optimization
        """
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

        # This is implemented using a closure mainly due to the potential usage of BFGS
        # BFGS needs to evaluate the function multiple times and therefore needs a defined closure
        # All other optimizers handle the closure just fine as well, but only execute it once
        def closure():
            optimizer.zero_grad()
            loss = loss_fn(observations, next_observations, rewards, actions)
            loss.backward()
            return loss

        for i in range(optimizer_steps):
            optimizer.step(closure)

    def nll_loss(self, observations, next_observations, rewards, actions):
        weights = self.calc_weights(observations, next_observations, rewards, actions)
        log_likes = self.policy.log_likelihood(observations, actions)
        nll = -torch.mean(torch.exp(weights.detach()) * log_likes)
        return nll
