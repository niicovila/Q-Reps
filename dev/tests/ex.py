import random, os
from collections import deque
import torch
from torch import nn
import numpy as np
from qreps import QREPS

from dm_env import truncation
import logging
logger = logging.getLogger("qreps")
logger.addHandler(logging.NullHandler())
import gym
import ray
from ray import tune, train
from ray.tune.search import Repeater
from ray.tune.search.hebo import HEBOSearch


from torch.utils.tensorboard import SummaryWriter
from gym.envs.toy_text.frozen_lake import generate_random_map

from policy import Policy, StochasticTablePolicy, CategoricalMLP, QREPSPolicy

from qfunction import  SimpleQFunction, NNQFunction
from sampler import Sampler, ExponentiatedGradientSampler
from feature_map import FeatureMap, FeatureConcatenation, OneHotFeature, IdentityFeature


from bsuite.utils import gym_wrapper

env = gym.make("CartPole-v0")

env = gym_wrapper.DMEnvFromGym(env)
num_obs = env.observation_spec().shape[0]
num_act = env.action_spec().num_values

config = {
    "eta": 0.0045764553781095264,
    "alpha": 0.02864615079338432,
    "beta": 0.015676210325628655,
    "saddle_point_steps": 450,
    "discount": 0.99,
}


def main(config):

    q_function = NNQFunction(
        obs_dim=num_obs, act_dim=num_act, feature_fn=IdentityFeature()
    )
    policy = QREPSPolicy(q_function=q_function, temp=config["eta"])


    writer = SummaryWriter()

    agent = QREPS(
        writer=writer,
        policy=policy,
        q_function=q_function,
        learner=torch.optim.Adam,
        reward_transformer=lambda r: r / 1000,
        optimize_policy=False,
        **config,
    )

    def obtain_episode(max_steps):
        timestep = env.reset()
        step = 0
        rewards = []
        while not timestep.last():
            # Generate an action from the agent's policy.
            action = agent.select_action(timestep)

            # Step the environment.
            new_timestep = env.step(action)

            if step == max_steps:
                new_timestep = truncation(new_timestep.reward, new_timestep.observation)

            # Tell the agent about what just happened.
            agent.update(timestep, action, new_timestep)

            rewards.append(new_timestep.reward)

            # Book-keeping.
            timestep = new_timestep
            step += 1
        return np.sum(rewards)
    
    def train(num_iterations, max_steps, number_rollouts=1, logging_callback=None):
        """Trains the set algorithm for num_episodes and limits the steps per episode on max_steps.
        Note that the episode is returned earlier if the environment switches to done"""
        iter = 0
        for iteration in range(num_iterations):
            for rollout in range(number_rollouts):
                reward = obtain_episode(max_steps)
                if logging_callback is not None:
                    logging_callback(reward)

            agent.update_policy(iter)
            # Count global iterations
            iter += 1
            print(reward)
    train(30, 200, 5)
    
main(config)