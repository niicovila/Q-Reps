# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse
import itertools
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from replay_buffer import ReplayBufferAtari
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from ray import train
from ray.tune.search import Repeater
from ray.tune.search.hebo import HEBOSearch
import ray.tune as tune  # Import the missing package

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

config = {
  "exp_name": "QREPS",
  "seed": 1,
  "torch_deterministic": True,
  "cuda": True,
  "track": False,
  "wandb_project_name": "QREPS_CartPole-v1",
  "wandb_entity": 'TFG',
  "capture_video": False,
  "env_id": "BeamRiderNoFrameskip-v4",
  "total_timesteps": 128,
  "num_updates": 20,
  "buffer_size": 10000,
  "update_epochs": tune.choice([5, 50, 100, 300]),
  "update_policy_epochs": tune.choice([50, 300, 450]),
  "num_rollouts": 5,
  "num_envs": tune.choice([1, 4, 6]),
  "gamma": 0.99,
  "policy_lr": tune.choice([0.1, 2e-2, 2.5e-3]),
  "q_lr": tune.choice([0.1, 2e-2, 2.5e-3]),
  "alpha":  tune.choice([0.2, 0.5, 2, 4, 6]),
  "eta": None,
  "beta": tune.choice([0.1, 0.01, 0.002, 4e-5]),
  "use_linear_schedule":  tune.choice([True, False]),
  "saddle_point_optimization":  tune.choice([True, False]),
  "use_kl_loss": tune.choice([True, False]),
}

import logging
FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
SEED_OFFSET = 0

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

class ExponentiatedGradientSampler:
    def __init__(self, N, device, eta, beta=0.01):
        self.n = N
        self.eta = eta
        self.beta = beta

        self.h = torch.ones((self.n,)) / N
        self.z = torch.ones((self.n,)) / N

        self.prob_dist = Categorical(torch.ones((self.n,))/ N)
        self.device = device

    def reset(self):
        self.h = torch.ones((self.n,))
        self.z = torch.ones((self.n,))
        self.prob_dist = Categorical(torch.softmax(torch.ones((self.n,)), 0))
                                     
    def probs(self):
        return self.prob_dist.probs.to(self.device)
    
    def entropy(self):
        return self.prob_dist.entropy().to(self.device)
    
    def update(self, bellman):
        self.h = bellman -  self.eta * torch.log(self.n * self.probs())
        t = self.beta*self.h
        self.z = self.probs() * torch.exp(t)
        self.z = torch.clamp(self.z / (torch.sum(self.z)), min=1e-8, max=1.0)
        self.prob_dist = Categorical(self.z)

def empirical_bellman_error(observations, next_observations, actions, rewards, qf, policy, gamma):
    v_target = qf.get_values(next_observations, actions, policy)[1]
    q_features = qf.get_values(observations, actions, policy)[0]
    output = rewards + gamma * v_target - q_features
    return output

def saddle(eta, observations, next_observations, actions, rewards, qf, policy, gamma, sampler):
    errors = torch.sum(sampler.probs().detach() * (empirical_bellman_error(observations, next_observations, actions, rewards, qf, policy, gamma) - eta * torch.log(sampler.n * sampler.probs().detach()))) + (1 - gamma) * qf.get_values(observations, actions, policy)[1].mean()
    return errors

def ELBE(eta, observations, next_observations, actions, rewards, qf, policy, gamma, sampler=None):
    errors = eta * torch.logsumexp(
        empirical_bellman_error(observations, next_observations, actions, rewards, qf, policy, gamma) / eta, 0
    ) + torch.mean((1 - gamma) * qf.get_values(observations, actions, policy)[1], 0)
    return errors

def nll_loss(alpha, observations, next_observations, rewards, actions, log_likes, q_net, policy):
    weights = torch.clamp(q_net.get_values(observations, actions, policy)[0] / alpha, -20, 20)
    _, log_likes, _, _ = policy.get_action(observations, actions)
    nll = -torch.mean(torch.exp(weights.detach()) * log_likes)
    return nll

def kl_loss(alpha, observations, next_observations, rewards, actions, log_likes, q_net, policy):
    _, _, newlogprob, probs = policy.get_action(observations, actions)
    q_values = q_net.get_values(observations, policy=policy)[0]
    actor_loss = torch.mean(probs * (alpha * (newlogprob-log_likes.detach()) - q_values.detach()))
    return actor_loss

def optimize_critic(eta, observations, next_observations, actions, rewards, q_net, policy, gamma, sampler, optimizer, steps=300, loss_fn=ELBE):
    def closure():
        optimizer.zero_grad()
        loss = loss_fn(eta, observations, next_observations, actions, rewards, q_net , policy, gamma, sampler)
        loss.backward()
        if sampler is not None: sampler.update(empirical_bellman_error(observations, next_observations, actions, rewards, q_net, policy, gamma))
        # nn.utils.clip_grad_norm_([param for group in optimizer.param_groups for param in group['params']], 1.0)
        return loss

    for i in range(steps):
        optimizer.step(closure)

def optimize_actor(alpha, observations, next_observations, rewards, actions, log_likes, q_net, policy, optimizer, steps=300, loss_fn=nll_loss):
    def closure():
        optimizer.zero_grad()
        loss = loss_fn(alpha, observations, next_observations, rewards, actions, log_likes, q_net, policy)
        loss.backward()
        return loss

    for i in range(steps):
        optimizer.step(closure)

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.env = env
        self.alpha = args.alpha

        obs_shape = env.single_observation_space.shape

        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]
        
        self.critic = nn.Sequential(
            self.conv,
            layer_init(nn.Linear(output_dim, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, env.single_action_space.n), std=1),
        )

    def forward(self, x):
        return self.critic(x / 255.0)
    
    def get_values(self, x, action=None, policy=None):
        q = self(x)
        z = q / self.alpha
        if policy is None: pi_k = torch.ones(x.shape[0], self.env.single_action_space.n, device=x.device) / self.env.single_action_space.n
        else: _, _, _, pi_k = policy.get_action(x); pi_k = pi_k.detach()
        v = self.alpha * (torch.log(torch.sum(pi_k * torch.exp(z), dim=1))).squeeze(-1)
        if action is None:
            return q, v
        else:
            q = q.gather(-1, action.unsqueeze(-1).long()).squeeze(-1)
            return q, v
    
class QREPSPolicy(nn.Module):
    def __init__(self, env):
        super().__init__()

        obs_shape = env.single_observation_space.shape

        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.actor = nn.Sequential(
            self.conv,
            layer_init(nn.Linear(output_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, env.single_action_space.n), std=0.01),
        )

    def forward(self, x):
        return self.actor(x / 255.0)

    def get_action(self, x, action=None):
        logits = self(x)
        policy_dist = Categorical(logits=logits)
        if action is None: action = policy_dist.sample()
        action_probs = policy_dist.probs
        log_prob = torch.log(action_probs+1e-6)
        action_log_prob = policy_dist.log_prob(action)
        return action, action_log_prob, log_prob, action_probs


def main(config: dict):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    args = argparse.Namespace(**config)
    args.seed = config["__trial_index__"] + SEED_OFFSET
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    logging_callback=lambda r: train.report({'reward':r})
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    actor = QREPSPolicy(envs).to(device)
    qf = QNetwork(envs, args).to(device)

    q_optimizer = optim.Adam(list(qf.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)    
    alpha = args.alpha
    if args.eta is None: eta = args.alpha
    else: eta = args.eta

    rb = ReplayBufferAtari(args.buffer_size)
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    
    global_step = 0

    for T in range(1, args.num_updates+1):

        all_rewards = []

        if args.use_linear_schedule:
            frac = 1.0 - (T - 1.0) / args.num_updates
            lrnow = frac * args.q_lr
            q_optimizer.param_groups[0]["lr"] = lrnow

            frac = 1.0 - (T - 1.0) / args.num_updates
            lrnow = frac * args.policy_lr
            actor_optimizer.param_groups[0]["lr"] = lrnow

        for N in range(args.num_rollouts):
            obs, _ = envs.reset(seed=args.seed)
            episode_reward = []
            for step in range(args.total_timesteps):
                global_step += args.num_envs
                with torch.no_grad():
                    action, _, loglikes, _ = actor.get_action(torch.Tensor(obs).to(device))
                
                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, done, truncation, info = envs.step(action.cpu().numpy())
                reward, obs, next_obs, done = torch.tensor(reward).to(device).view(-1), torch.Tensor(obs).to(device), torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
                rb.push(obs, next_obs, action, reward, done, loglikes)
                obs = next_obs
                all_rewards.append(reward)
                episode_reward.append(reward)
                if done.any():
                    break

        # TRAINING PHASE         
        (
        observations, 
        next_observations, 
        actions, 
        rewards, 
        dones, 
        log_likes
        ) = rb.get_all()

        if args.saddle_point_optimization:
            sampler = ExponentiatedGradientSampler(observations.shape[0], device, eta, beta=0.01)
            optimize_critic(eta, observations, next_observations, actions, rewards, qf, actor, args.gamma, sampler, q_optimizer, steps=args.update_epochs, loss_fn=saddle)
        else:
            optimize_critic(eta, observations, next_observations, actions, rewards, qf, actor, args.gamma, None, q_optimizer, steps=args.update_epochs, loss_fn=ELBE)

        if args.use_kl_loss: optimize_actor(alpha, observations, next_observations, rewards, actions, log_likes, qf, actor, actor_optimizer, steps=args.update_policy_epochs, loss_fn=kl_loss)
        else: optimize_actor(alpha, observations, next_observations, rewards, actions, log_likes, qf, actor, actor_optimizer, steps=args.update_policy_epochs, loss_fn=nll_loss)
        rb.reset()
         
        writer.add_scalar("charts/episodic_return", np.sum([rew.cpu().numpy() for rew in all_rewards])/(args.num_rollouts*args.num_envs), T)
        logging_callback(torch.sum(torch.Tensor(all_rewards))/(args.num_rollouts*args.num_envs))
        

    envs.close()
    writer.close()

ray_init_config = {
    "gpu": 1,  # Adjust based on the number of available GPUs
    "cpu": 4,  # Number of CPU cores to allocate per trial
    # Additional Ray initialization options if needed
}

search_alg = HEBOSearch(metric="reward", mode="max")
re_search_alg = Repeater(search_alg, repeat=5)

analysis = tune.run(
    main,
    num_samples=200,
    config=config,
    search_alg=re_search_alg,
    # resources_per_trial=ray_init_config,
    local_dir="/Users/nicolasvila/workplace/uni/tfg_v2/tests/qreps/results_tune_qreps_v3",
)

print("Best config: ", analysis.get_best_config(metric="reward", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.results_df

df.to_csv("tuning_no_grad.csv")