import argparse
from copy import deepcopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from replay_buffer import ReplayBuffer
from agent import Agent

from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sampler import ExponentiatedGradientSampler, BestResponseSampler
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
  "env_id": "CartPole-v1",
  "total_timesteps": 500,
  "num_updates": 50,
  "buffer_size": 10000,
  "update_epochs": tune.choice([50, 100, 300]),
  "update_policy_epochs": tune.choice([50, 100, 300]),
  "num_rollouts": 5,
  "num_envs": tune.choice([1, 2, 5]),
  "gamma": tune.choice([0.99, 1]),
  "policy_lr_start": tune.loguniform(2.5e-3, 2e-1),
  "policy_lr_end": 2.5e-4,
  "q_lr_start": tune.loguniform(2.5e-3, 2.5e-1),
  "q_lr_end":2.5e-4,
  "alpha": tune.loguniform(2e-1, 10),
  "eta": None,
  "autotune": False,
  "target_entropy_scale": 0.1,
  "use_linear_schedule": True,
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
        env.action_space.seed(seed)
        return env
    return thunk

def empirical_bellman_error(observations, next_observations, actions, rewards, qf, policy, gamma):
    v_target = qf.get_values(next_observations, actions, policy)[1]
    q_features = qf.get_values(observations, actions, policy)[0]
    output = rewards + gamma * v_target - q_features
    return output

def ELBE(eta, observations, next_observations, actions, rewards, qf, policy, gamma):
    errors = eta * torch.logsumexp(
        empirical_bellman_error(observations, next_observations, actions, rewards, qf, policy, gamma) / eta, 0
    ) + torch.mean((1 - gamma) * qf.get_values(observations, actions, policy)[1], 0)
    return errors

def nll_loss(alpha, observations, next_observations, rewards, actions, q_net, policy):
    weights = torch.clamp(q_net.get_values(observations, actions, policy)[0] / alpha, -20, 20)
    _, log_likes, _, _ = policy.get_action(observations, actions)
    nll = -torch.mean(torch.exp(weights.detach()) * log_likes)
    return nll

def optimize_critic(eta, observations, next_observations, actions, rewards, q_net, policy, gamma, optimizer, steps=300, loss_fn=ELBE):
    def closure():
        optimizer.zero_grad()
        loss = loss_fn(eta, observations, next_observations, actions, rewards, q_net , policy, gamma)
        loss.backward()
        return loss

    for i in range(steps):
        optimizer.step(closure)

def optimize_actor(alpha, observations, next_observations, rewards, actions, q_net, policy, optimizer, steps=300, loss_fn=nll_loss):
    def closure():
        optimizer.zero_grad()
        loss = loss_fn(alpha, observations, next_observations, rewards, actions, q_net, policy)
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
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(env.single_observation_space.shape).prod(), 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, env.single_action_space.n), std=1),
        )

    def forward(self, x):
        return self.critic(x)
    
    def get_values(self, x, action=None, policy=None):
        q = self(x)
        z = q / self.alpha
        if policy is None: pi_k = torch.ones(x.shape[0], self.env.single_action_space.n, device=x.device) / self.env.single_action_space.n
        else: _, _, log_pi, pi_k = policy.get_action(x); pi_k = pi_k.detach()
        v = self.alpha * torch.log(torch.sum(pi_k * torch.exp(z), dim=1)).squeeze(-1)

        if action is None:
            return q, v
        else:
            q = q.gather(-1, action.unsqueeze(-1).long()).squeeze(-1)
            return q, v
    
class QREPSPolicy(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, env.single_action_space.n), std=0.01),
        )

    def forward(self, x):
        return self.actor(x)

    def get_action(self, x, action=None):
        logits = self(x)
        policy_dist = Categorical(logits=logits)
        if action is None: action = policy_dist.sample()
        action_probs = policy_dist.probs
        log_prob = torch.log(action_probs+1e-6)
        action_log_prob = policy_dist.log_prob(action)
        return action, action_log_prob, log_prob, action_probs
    

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

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

    q_optimizer = optim.SGD(list(qf.parameters()), lr=args.q_lr_start)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr_start)

    if args.autotune:
        target_entropy = -args.target_entropy_scale * torch.log(1 / torch.tensor(envs.single_action_space.n))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, eps=1e-4)
    else:
        alpha = args.alpha

    if args.eta is None: eta = args.alpha
    else: eta = args.eta

    rb = ReplayBuffer(args.buffer_size)
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    
    global_step = 0
    try: 
        for T in range(args.num_updates):
            all_rewards = []
            if args.use_linear_schedule:
                q_optimizer.param_groups[0]["lr"] = linear_schedule(start_e= args.q_lr, end_e=args.q_lr_end, duration=args.num_updates, t=T)
                actor_optimizer.param_groups[0]["lr"] = linear_schedule(start_e= args.policy_lr, end_e=args.policy_lr_end, duration=args.num_updates, t=T)

            for N in range(args.num_rollouts):
                obs, _ = envs.reset(seed=args.seed)
                for step in range(args.total_timesteps):
                    global_step += 1
                    with torch.no_grad():
                        actions, a_loglike, loglikes, probs = actor.get_action(torch.Tensor(obs).to(device))
                    action = actions.numpy()

                    # TRY NOT TO MODIFY: execute the game and log data.
                    next_obs, reward, done, truncation, info = envs.step(action)
                    rb.push(obs, next_obs, action, reward, done, loglikes)
                    obs = next_obs
                    all_rewards.append(reward)
                    step += 1
                    if done.any():
                        break

            # TRAINING PHASE         
            (
            observations, 
            next_observations, 
            actions, 
            rewards, 
            dones,
            loglikes
            ) = rb.get_all()
            rewards = rewards / 1000

            optimize_critic(args.eta, observations, next_observations, actions, rewards, qf, actor, args.gamma, q_optimizer, steps=args.update_epochs)            
            optimize_actor(alpha, observations, next_observations, rewards, actions, qf, actor, actor_optimizer, steps=args.update_policy_epochs)
            rb.reset()
            logging_callback(np.sum(all_rewards)/(args.num_rollouts*args.num_envs))
            
            # if args.autotune:
            #     # re-use action probabilities for temperature loss
            #     alpha_loss = (action_probs.detach() * (-log_alpha.exp() * (log_pi + target_entropy).detach())).mean()

            #     a_optimizer.zero_grad()
            #     alpha_loss.backward()
            #     a_optimizer.step()
            #     alpha = log_alpha.exp().item()
        
    except:
        logging_callback(0.0)
    envs.close()
    writer.close()

search_alg = HEBOSearch(metric="reward", mode="max")
re_search_alg = Repeater(search_alg, repeat=3)

analysis = tune.run(
    main,
    num_samples=500,
    config=config,
    search_alg=re_search_alg,
    local_dir="/Users/nicolasvila/workplace/uni/tfg_v2/tests/qreps/results_tune_qreps_v3",
)

print("Best config: ", analysis.get_best_config(metric="reward", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.results_df

df.to_csv("qreps_analysis_v3_4_500).csv")