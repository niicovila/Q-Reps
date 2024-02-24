import logging
import sys, os
sys.path.append("../")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qreps.utilities.util import set_seed

sys.path.append("../")

import gym
import ray.tune as tune
import torch
from bsuite.utils import gym_wrapper
from torch.utils.tensorboard import SummaryWriter
from qreps.algorithms.sampler import BestResponseSampler

from qreps.algorithms import QREPS
from qreps.feature_functions import IdentityFeature
from qreps.policies import CategoricalMLP
from qreps.policies.qreps_policy import QREPSPolicy
from qreps.utilities.trainer import Trainer
from qreps.valuefunctions import NNQFunction

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
SEED_OFFSET = 0

config = {
    "eta": 4.8,
    "beta": 2e-2,
    "saddle_point_steps": 300,
    "policy_opt_steps": 300,
    "discount": 0.99,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def train(config: dict):

    env = gym.make("CartPole-v1")
    num_obs = env.observation_space.shape[0]
    num_act = env.action_space.n

    q_function = NNQFunction(
        obs_dim=num_obs, act_dim=num_act, feature_fn=IdentityFeature()
    )
    policy = CategoricalMLP(num_obs, 2)
    # policy = QREPSPolicy(q_function=q_function, temp=config["eta"])

    writer = SummaryWriter()

    agent = QREPS(
        writer=writer,
        policy=policy,
        q_function=q_function,
        learner=torch.optim.Adam,
        sampler=BestResponseSampler,
        optimize_policy=True,
        reward_transformer=lambda r: r / 2500,
        **config,
    )

    trainer = Trainer()
    trainer.setup(agent, env)
    trainer.train(num_iterations=100,max_steps=500,number_rollouts=5)


# Repeat 2 samples 10 times each.
train(config)