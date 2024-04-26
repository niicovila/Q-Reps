import argparse
from copy import deepcopy
import random
import time

import pandas as pd
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from ray import train
from ray.tune.search import Repeater
from ray.tune.search.hebo import HEBOSearch
import ray.tune as tune  # Import the missing package
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class BestResponseSampler:
    def __init__(self, N, device, eta, beta=None):
        self.n = N
        self.eta = eta

        self.z = torch.ones((self.n,)) / N

        self.prob_dist = Categorical(self.z)
        self.device = device

    def reset(self):
        self.h = torch.ones((self.n,))
        self.z = torch.ones((self.n,))
        self.prob_dist = Categorical(torch.softmax(torch.ones((self.n,)), 0))
                                     
    def probs(self):
        return self.prob_dist.probs.to(self.device)
    
    def update(self, bellman):
        t = self.eta * bellman
        t = torch.clamp(t, -20, 20)
        self.z = self.probs() * torch.exp(t)
        self.z = torch.clamp(self.z / (torch.sum(self.z + 1e-8)), min=1e-8, max=1.0)
        self.prob_dist = Categorical(self.z)

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



config = {
    "exp_name": "QREPS",
    "seed": 0,
    "torch_deterministic": True,
    "cuda": True,
    "track": False,
    "wandb_project_name": "QREPS_cleanRL",
    "wandb_entity": None,
    "capture_video": False,
    "env_id": "CartPole-v1",
    "total_timesteps": 100000,
    "gamma": 0.99,
    "target_network_frequency": 0,
    "tau": 1.0, 

    "num_envs": tune.choice([4, 8]),
    "num_steps": tune.choice([128, 256, 350, 500]),

    "q_lr_start": tune.loguniform(1e-4, 1e-1),
    "beta": tune.loguniform(1e-4, 1e-1),

    "update_epochs": tune.choice([10, 50, 100, 150]),

    "q_optimizer": tune.choice(["Adam", "SGD", "Sigmoid"]),
    "q_activation": tune.choice(["Tanh", "ReLU", "Sigmoid"]),

    "q_hidden_size": tune.choice([64, 128, 256, 512]),
    "q_num_hidden_layers": tune.choice([2, 4, 8]),
    "eps": tune.choice([1e-8, 1e-4]),

    "anneal_lr": tune.choice([True, False]),
    "target_network": False,
    "q_histogram": False,
    "average_critics": tune.choice([True, False]),

    "batch_size": 0,
    "num_iterations": 0
}

import logging
FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
SEED_OFFSET = 1

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

class QNetwork(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.env = env

        self.critic = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), args.q_hidden_size),
            getattr(nn, args.q_activation)(),
            *[layer for _ in range(args.q_num_hidden_layers) for layer in (
                nn.Linear(args.q_hidden_size, args.q_hidden_size),
                getattr(nn, args.q_activation)()
            )],
            nn.Linear(args.q_hidden_size, env.single_action_space.n),
        )

    def forward(self, x):
        return self.critic(x)
    
    def get_values(self, x, action=None, policy=None):
        q = self(x)
        _, probs = policy.get_action(x)
        v = (torch.sum(probs * q, dim=1)).squeeze(-1)
        if action is None:
            return q, v
        else:
            q = q.gather(-1, action.unsqueeze(-1).long()).squeeze(-1)
            return q, v
    
class Policy:
    def __init__(self, q):
        self.q = q

    def set_q(self, q):
        self.q = q
    
    def get_action(self, x, action=None):
        logits = self.q(x)
        prob_dist = Categorical(logits=logits)
        
        if action is None:
            return prob_dist.sample(), prob_dist.probs
        else:
            return action, prob_dist.probs

class ExponentiatedGradientSampler:
    def __init__(self, N, device, beta=0.01):
        self.n = N
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
        self.h = bellman
        t = self.beta*self.h
        self.z = self.probs() * torch.exp(t)
        self.z = torch.clamp(self.z / (torch.sum(self.z)), min=1e-8, max=1.0)
        self.prob_dist = Categorical(self.z)


def main(config):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    args = argparse.Namespace(**config)
    args.seed = config["__trial_index__"] + SEED_OFFSET

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_iterations = args.total_timesteps // args.batch_size
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
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    qf = QNetwork(envs, args).to(device)

    if args.target_network:
        qf_target = QNetwork(envs, args).to(device)
        qf_target.load_state_dict(qf.state_dict())

    actor = Policy(qf)

    if args.q_optimizer == "Adam" or args.q_optimizer == "RMSprop":
        q_optimizer = getattr(optim, args.q_optimizer)(
            list(qf.parameters()), lr=args.q_lr_start, eps=args.eps
        )
    else:
        q_optimizer = getattr(optim, args.q_optimizer)(
            list(qf.parameters()), lr=args.q_lr_start
        )

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    next_observations = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)  # Added this line
    
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    reward_iteration = []

    try:
     for iteration in range(1, args.num_iterations + 1):

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.q_lr_start
            q_optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():        
                action, _ = actor.get_action(next_obs)

            actions[step] = action

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            reward = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            
            next_observations[step] = next_obs

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        reward_iteration.append(info["episode"]["r"])

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_next_obs = next_observations.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)

        b_rewards = rewards.flatten()
        b_dones = dones.flatten()
        weights_after_each_epoch = []

        if len(reward_iteration) > 5: 
            logging_callback(np.mean(reward_iteration))
            reward_iteration = []

        sampler = ExponentiatedGradientSampler(b_obs.shape[0], device, args.beta)

        for epoch in range(args.update_epochs):
            
            if args.target_network:
                delta = b_rewards.squeeze() + args.gamma * qf_target.get_values(b_next_obs, policy=actor)[1].detach() * (1 - b_dones.squeeze()) - qf.get_values(b_obs, b_actions, actor)[0]          
            else: delta = b_rewards.squeeze() + args.gamma * qf.get_values(b_next_obs, policy=actor)[1] * (1 - b_dones.squeeze()) - qf.get_values(b_obs, b_actions, actor)[0]

            bellman = delta.detach()
            z_n = sampler.probs() 
            critic_loss = torch.sum(z_n.detach() * delta) + (1 - args.gamma) * qf.get_values(b_obs, b_actions, actor)[1].mean()
        
            q_optimizer.zero_grad()
            critic_loss.backward()
            q_optimizer.step()

            sampler.update(bellman)
            if args.average_critics: weights_after_each_epoch.append(deepcopy(qf.state_dict()))
    
        if args.average_critics:
            avg_weights = {}
            for key in weights_after_each_epoch[0].keys():
                avg_weights[key] = sum(T[key] for T in weights_after_each_epoch) / len(weights_after_each_epoch)
            qf.load_state_dict(avg_weights)

            actor.set_q(qf)
            
            if args.target_network and iteration % args.target_network_frequency == 0:
                for param, target_param in zip(qf.parameters(), qf_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

    except:
        logging_callback(-1111.0)

    if len(reward_iteration) > 0: 
        logging_callback(np.mean(reward_iteration))
        
    envs.close()
    writer.close()

search_alg = HEBOSearch(metric="reward", mode="max")
re_search_alg = Repeater(search_alg, repeat=3)

# search_alg = OptunaSearch(metric="reward", mode="max")
# search_alg = ConcurrencyLimiter(search_alg, max_concurrent=4)
# re_search_alg = Repeater(search_alg, repeat=1)

ray_init = {"cpu": 1}

analysis = tune.run(
    main,
    num_samples=300,
    config=config,
    search_alg=re_search_alg,
    # resources_per_trial=ray_init,
    local_dir="/Users/nicolasvila/workplace/uni/tfg_v2/tests/results_tune",
)
df = analysis.results_df
df.to_csv("CartPole_qreps_mod_v2.csv")
