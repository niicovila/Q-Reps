import argparse
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from replay_buffer import ReplayBuffer

from torch.nn.functional import F
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
        t = torch.clamp(t, -50, 50)
        self.z = self.probs() * torch.exp(t)
        self.z = torch.clamp(self.z / (torch.sum(self.z + 1e-8)), min=1e-8, max=1.0)
        self.prob_dist = Categorical(self.z)

config = {
  "exp_name": "QREPS",
  "seed": 1,
  "torch_deterministic": True,
  "cuda": True,
  "track": False,
  "wandb_project_name": "QREPS_CartPole-v1",
  "wandb_entity": 'TFG',
  "capture_video": False,
  "env_id": "LunarLander-v2",
  "total_timesteps": 1000,
  "num_updates": 100,
  "buffer_size": 10000,
  "update_epochs": tune.choice([5, 50, 100, 300]),
  "update_policy_epochs": tune.choice([50, 300, 450]),
  "num_rollouts": tune.choice([4, 6, 8]),
  "num_envs": 1,
  "gamma": 0.99,
  "policy_lr_start": tune.choice([2e-2, 2.5e-3]),
  "q_lr_start": tune.choice([0.1, 2e-2, 2.5e-3]),
  "q_lr_end": 0,
  "policy_lr_end":  0,
  "alpha":  tune.choice([2, 4, 6, 8]),
  "eta": None,
  "beta": tune.choice([0.1, 0.01, 0.002, 4e-5]),
  "use_linear_schedule":  False,
  "anneal_lr": tune.choice([True, False]),
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
        env.action_space.seed(seed)

        return env

    return thunk

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

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
        if sampler is not None: sampler.update(empirical_bellman_error(observations, next_observations, actions, rewards, q_net, policy, gamma).detach())
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


class QNetwork(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.env = env
        self.alpha = args.alpha
        self.critic = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.single_action_space.n),
        )

    def forward(self, x):
        return self.critic(x)
    
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
        self.actor = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, env.single_action_space.n),
        )

    def forward(self, x):
        return self.actor(x)

    def get_action(self, x, action=None, critic=None):
        if critic is not None:
            logits = critic(x)
        else:
            logits = self(x)
        policy_dist = Categorical(logits=logits)
        if action is None: action = policy_dist.sample()
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(action_probs, dim=1)
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
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    actor = QREPSPolicy(envs).to(device)
    qf = QNetwork(envs, args).to(device)

    q_optimizer = optim.Adam(list(qf.parameters()), lr=args.q_lr_start, eps=1e-4)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr_start, eps=1e-4)

    alpha = args.alpha
    if args.eta is None: eta = args.alpha
    else: eta = args.eta

    rb = ReplayBuffer(args.buffer_size)

    obs, _ = envs.reset(seed=args.seed)
    global_step = 0
    try:
     for T in range(1, args.num_updates+1):
        all_rewards = []
        
        if args.use_linear_schedule:
            q_optimizer.param_groups[0]["lr"] = linear_schedule(start_e= args.q_lr_start, end_e=args.q_lr_end, duration=100, t=T)
            actor_optimizer.param_groups[0]["lr"] = linear_schedule(start_e= args.policy_lr_start, end_e=args.policy_lr_end, duration=100, t=T)
        
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (T - 1.0) / args.num_updates
            lrnow = frac * args.policy_lr_start
            actor_optimizer.param_groups[0]["lr"] = lrnow

            lrnow = frac * args.q_lr_start
            q_optimizer.param_groups[0]["lr"] = lrnow

        for N in range(args.num_rollouts):
            episode_reward = []
            obs, _ = envs.reset()
            for step in range(args.total_timesteps):
                global_step += args.num_envs
                with torch.no_grad():
                    actions, _, loglikes, _ = actor.get_action(torch.Tensor(obs).to(device))

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, done, _, infos = envs.step(actions.detach().cpu().numpy())
                reward, obs, next_obs, done = torch.tensor(reward).to(device).view(-1), torch.Tensor(obs).to(device), torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

                rb.push(obs, next_obs, actions, reward, done, loglikes)
                obs = next_obs
                all_rewards.append(reward)
                episode_reward.append(reward)
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
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
            sampler = ExponentiatedGradientSampler(observations.shape[0], device, eta, beta=args.beta)
            optimize_critic(eta, observations, next_observations, actions, rewards, qf, actor, args.gamma, sampler, q_optimizer, steps=args.update_epochs, loss_fn=saddle)
        else: optimize_critic(eta, observations, next_observations, actions, rewards, qf, actor, args.gamma, None, q_optimizer, steps=args.update_epochs, loss_fn=ELBE)

        if args.use_kl_loss: optimize_actor(alpha, observations, next_observations, rewards, actions, log_likes, qf, actor, actor_optimizer, steps=args.update_policy_epochs, loss_fn=kl_loss)
        else: optimize_actor(alpha, observations, next_observations, rewards, actions, log_likes, qf, actor, actor_optimizer, steps=args.update_policy_epochs, loss_fn=nll_loss)
        
        logging_callback(torch.sum(torch.Tensor(all_rewards))/(args.num_rollouts*args.num_envs))
        rb.reset()

    except:
     logging_callback(-1000.0)
    envs.close()
    writer.close()


search_alg = HEBOSearch(metric="reward", mode="max")
re_search_alg = Repeater(search_alg, repeat=1)

analysis = tune.run(
    main,
    num_samples=500,
    config=config,
    search_alg=re_search_alg,
    # resources_per_trial=ray_init_config,
    local_dir="/Users/nicolasvila/workplace/uni/tfg_v2/tests/qreps/results_tune_qreps_v3",
)

print("Best config: ", analysis.get_best_config(metric="reward", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.results_df

df.to_csv("LunarLander_tune_no_reset_v4.csv")