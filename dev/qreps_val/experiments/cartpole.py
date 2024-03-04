import argparse
import logging
import random
import sys, os
import time
import numpy as np

import wandb

sys.path.append("../")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import gym
import torch
from torch.utils.tensorboard import SummaryWriter

from qreps.algorithms import QREPS, SaddleQREPS
from qreps.algorithms.sampler import ExponentiatedGradientSampler
from qreps.algorithms.sampler import BestResponseSampler
from qreps.feature_functions import IdentityFeature
from qreps.policies.qreps_policy import QREPSPolicy
from qreps.policies.categorical_mlp import CategoricalMLP
from qreps.utilities.trainer import Trainer
from qreps.utilities.util import set_seed
from qreps.valuefunctions import NNQFunction, DiscreteMLPCritic
import itertools
import pandas as pd

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


def make_env(env_id, capture_video, run_name):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk()

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

SEED_OFFSET = 0

qreps_config = {
    "env_id": "LunarLander-v2",
    "track": False,
    "eta": 0.05,
    "beta": 2.5e-4,
    "saddle_point_steps": 300,
    "policy_opt_steps": 300,
    "discount": 0.99,
    "wandb_project_name": "qreps",
    "wandb_entity": None,
    "seed" : 0,
    "exp_name": os.path.basename(__file__)[: -len(".py")],
    "num_envs": 1,
    "capture_video": False,
}
 
## seed 4: "eta": 0.204177551020,  "beta": 0.001 QREPS


def train(config: dict):
    config = argparse.Namespace(**config)
    run_name = f"{config.env_id}__{config.exp_name}__{config.seed}__{int(time.time())}"
    if config.track:
        wandb.init(
                project=config.wandb_project_name,
                entity=config.wandb_entity,
                sync_tensorboard=True,
                config=vars(config),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
    writer = SummaryWriter(f"runs/{run_name}")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(config.env_id, config.capture_video, run_name)

    # env = gym.make("CartPole-v1", render_mode="rgb_array")
    # env = gym.wrappers.RecordVideo(env, f"videos/{'cartpole'}")

    num_obs = env.observation_space.shape[0]
    num_act = env.action_space.n

    q_function = NNQFunction(obs_dim=num_obs, act_dim=num_act, feature_fn=IdentityFeature()).to(device)
    policy = CategoricalMLP(num_obs, num_act).to(device)
    # policy = QREPSPolicy(q_function, temp=config.eta)
    writer = SummaryWriter()

    agent = SaddleQREPS(
        writer=writer,
        policy=policy,
        q_function=q_function,
        learner=torch.optim.Adam,
        optimize_policy=True,
        policy_lr=1e-2,
        sampler=ExponentiatedGradientSampler,
        reward_transformer=lambda r: r / 5,
        device = device,
        **vars(config))

    trainer = Trainer(config.seed)
    trainer.setup(agent, env)
    reward = trainer.train(num_iterations=30, max_steps=500, number_rollouts=5)
    env.close()
    writer.close()
    return reward

# train(qreps_config)

alphas = np.linspace(0.0001, 5, num=100)
learning_rates = [0.1, 1e-2, 1e-3, 1e-4]

results = []

for alpha, learning_rate in itertools.product(alphas, learning_rates):
    qreps_config["eta"] = alpha
    qreps_config["beta"] = learning_rate
    try: 
        print(f"Training with alpha: {alpha}, learning_rate: {learning_rate}")
        reward = train(qreps_config)
        results.append((alpha, learning_rate, reward))

    except:
        pass

best_combination = max(results, key=lambda x: x[2])

print("Best Combination:")
print("eta:", best_combination[0])
print("Learning Rate:", best_combination[1])
print("Metric Value:", best_combination[2])

df = pd.DataFrame(results, columns=["Alpha", "Learning Rate", "Reward"])
df.to_csv("results_saddle.csv", index=False)