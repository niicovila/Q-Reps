import logging
import sys, os

import numpy as np 
sys.path.append("../")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import gym
import torch
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from qreps.algorithms import QREPS
from qreps.algorithms.sampler import BestResponseSampler
from qreps.feature_functions import IdentityFeature
from qreps.policies.qreps_policy import QREPSPolicy
from qreps.utilities.trainer import Trainer
from qreps.valuefunctions import ResNet

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

SEED_OFFSET = 0

qreps_config = {
    "eta": 0.1,
    "beta": 2e-2,
    "saddle_point_steps": 300,
    "policy_opt_steps": 300,
    "discount": 0.99,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


def train(config: dict):
    gym_id = "ALE/Alien-v5"
    num_envs = 1
    run_name = "qreps"
    capture_video = False
    seed = 0
    env = make_env(gym_id, seed , 0, capture_video, run_name)()
    
    env = gym.make("ALE/Alien-v5")
    num_obs = env.observation_space.shape[0]
    num_act = env.action_space.n
    device = config["device"]

    q_function = ResNet(observation_space=num_obs, action_space=num_act, feature_fn=IdentityFeature()).to(device)
    policy = QREPSPolicy(q_function=q_function, temp=config["eta"])
    writer = SummaryWriter()

    agent = QREPS(
        writer=writer,
        policy=policy,
        q_function=q_function,
        learner=torch.optim.Adam,
        sampler=BestResponseSampler,
        optimize_policy=False,
        reward_transformer=lambda r: r ,
        **config,
    )

    trainer = Trainer()
    trainer.setup(agent, env)
    trainer.train(num_iterations=1, max_steps=20, number_rollouts=2)


train(qreps_config)

