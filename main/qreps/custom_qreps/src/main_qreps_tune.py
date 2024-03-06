import argparse
import logging
import random
import sys, os
import time
import numpy as np

import wandb

sys.path.append("../qreps/")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
script_dir = os.path.abspath(os.path.dirname(__file__))
qreps_parent_dir = os.path.join(script_dir, "..")
sys.path.append(qreps_parent_dir)

import gym
import torch
from torch.utils.tensorboard import SummaryWriter

from qreps.algorithms import QREPS, SaddleQREPS
from qreps.algorithms.sampler import ExponentiatedGradientSampler, BestResponseSampler
from qreps.feature_functions import IdentityFeature
from qreps.policies.qreps_policy import QREPSPolicy
from qreps.policies.categorical_mlp import CategoricalMLP
from qreps.utilities.trainer import Trainer
from qreps.utilities.util import set_seed
from qreps.valuefunctions import NNQFunction, DiscreteMLPCritic
import itertools
import pandas as pd



import ray.tune as tune
from ray.tune.search import Repeater
from ray.tune.search.hebo import HEBOSearch


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
SEED_OFFSET = 0

qreps_config = {
    "env_id": "CartPole-v1",
    "track": False,
    "eta": tune.loguniform(2e-3, 2e-1),
    "alpha": tune.loguniform(2e-3, 2e-1),
    "beta": tune.loguniform(2e-4, 2e-1),
    "saddle_point_steps": tune.choice([300, 450]),
    "policy_opt_steps": tune.choice([300, 450]),
    "policy_lr": tune.loguniform(2e-3, 2e-1),
    "discount": 0.99,
    "wandb_project_name": "qreps",
    "wandb_entity": None,
    "seed" : 4,
    "exp_name": os.path.basename(__file__)[: -len(".py")],
    "num_envs": 1,
    "capture_video": False,
}
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
        optimize_policy=True,
        policy_lr=0.02,
        reward_transformer=lambda r: r / 5,
        device = device,
        **vars(config))

    trainer = Trainer(config.seed)
    trainer.setup(agent, env)
    reward = trainer.train(num_iterations=30, max_steps=200, number_rollouts=5)
    env.close()
    writer.close()
    return reward


search_alg = HEBOSearch(metric="reward", mode="max")
re_search_alg = Repeater(search_alg, repeat=5)

# Repeat 2 samples 10 times each.
analysis = tune.run(
    train,
    num_samples=5,
    config=qreps_config,
    search_alg=re_search_alg,
    local_dir="/Users/nicolasvila/workplace/uni/tfg_v2/tests/results_tune_customqreps",
)

print("Best config: ", analysis.get_best_config(metric="reward", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.results_df

df.to_csv("qreps_analysis.csv")