# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse 
from copy import deepcopy
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from ray import train
from ray.tune.search import Repeater
from ray.tune.search.hebo import HEBOSearch
import ray.tune as tune  # Import the missing package
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter

import logging
FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
SEED_OFFSET = 1
logging_callback=lambda r: train.report({'reward':r})

config = {
    "exp_name": "QREPS",
    "seed": 0,
    "torch_deterministic": True,
    "cuda": True,
    "track": False,
    "wandb_project_name": "CC",
    "wandb_entity": None,
    "capture_video": False,

    "env_id": "CartPole-v1",

    # Algorithm
    "total_timesteps": 50000,
    "num_envs": 16,
    "gamma": 0.99,

    "total_iterations": 2048,
    "num_minibatches": 32,
    "update_epochs": 50,

    "alpha": tune.choice([4, 8, 12, 32]),    
    "eta": None,

    # Learning rates
    "beta": tune.choice([3e-05, 0.0001, 0.00025, 0.0003, 0.001, 0.003]),
    "policy_lr": tune.choice([3e-05, 0.0001, 0.00025, 0.0003, 0.001, 0.003]),
    "q_lr": tune.choice([3e-05, 0.0001, 0.00025, 0.0003, 0.001, 0.003]),

    # Architecture
    "policy_activation": tune.choice(["Tanh", "ReLU", "Sigmoid", "ELU"]),
    "num_hidden_layers": tune.choice([2, 4, 8]),
    "hidden_size": tune.choice([32, 64, 128, 512]),
    "actor_last_layer_std": tune.choice([0.01, 0.1, 1.0]),

    "q_activation": tune.choice(["Tanh", "ReLU", "Sigmoid", "ELU"]),
    "q_num_hidden_layers": tune.choice([2, 4, 8]),
    "q_hidden_size": tune.choice([32, 64, 128, 512]),
    "q_last_layer_std": tune.choice([0.01, 0.1, 1.0]),

    "sampler_activation": tune.choice(["Tanh", "ReLU", "Sigmoid", "ELU"]),
    "sampler_num_hidden_layers": tune.choice([2, 4, 8]),
    "sampler_hidden_size": tune.choice([32, 64, 128, 512]),
    "sampler_last_layer_std": tune.choice([0.01, 0.1, 1.0]),

    "layer_init": tune.choice(["default", 
                               "orthogonal_gain", 
                               "orthogonal", 
                               "xavier_normal", 
                               "xavier_uniform", 
                               "kaiming_normal", 
                               "kaiming_uniform", 
                               "sparse"]),

    # Optimization
    "q_optimizer": "Adam",  # "Adam", "SGD", "RMSprop
    "actor_optimizer": "Adam",
    "sampler_optimizer": "Adam",
    "eps": 1e-8,

    # Options
    "average_critics": True,
    "normalize_delta": False,
    "use_kl_loss": False,
    "anneal_lr": False,
    "parametrized_sampler" : True,
    "saddle_point_optimization": True,
    "q_histogram": False,

    "target_network": False,
    "tau": 1.0,
    "target_network_frequency": 0, 

    "minibatch_size": 0,
    "num_iterations": 0,
    "num_steps": 0,
    "logging_callback": logging_callback,
}

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

def layer_init(layer, args, gain_ort=np.sqrt(2), bias_const=0.0, gain=1):

    if args.layer_init == "orthogonal_gain":
        torch.nn.init.orthogonal_(layer.weight, gain_ort)
    elif args.layer_init == "orthogonal":
        torch.nn.init.orthogonal_(layer.weight, gain)

    elif args.layer_init == "xavier_normal":
        torch.nn.init.xavier_normal_(layer.weight, gain)
    elif args.layer_init == "xavier_uniform":
        torch.nn.init.xavier_uniform_(layer.weight, gain)

    elif args.layer_init == "kaiming_normal":
        torch.nn.init.kaiming_normal_(layer.weight)
    elif args.layer_init == "kaiming_uniform":
        torch.nn.init.kaiming_uniform_(layer.weight)

    elif args.layer_init == "sparse":
        torch.nn.init.sparse_(layer.weight, sparsity=0.1)
    else:
        pass
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def nll_loss(alpha, observations, next_observations, rewards, actions, log_likes, q_net, policy):
    weights = torch.clamp(q_net.get_values(observations, actions, policy)[0] / alpha, -50, 50)
    _, log_likes, _, _ = policy.get_action(observations, actions)
    nll = -torch.mean(torch.exp(weights.detach()) * log_likes)
    return nll

def kl_loss(alpha, observations, next_observations, rewards, actions, log_likes, q_net, policy):
    _, _, newlogprob, probs = policy.get_action(observations, actions)
    q_values, v = q_net.get_values(observations, policy=policy)
    advantage = q_values - v.unsqueeze(1)
    actor_loss = torch.mean(probs * (alpha * (newlogprob-log_likes.detach()) - advantage.detach()))
    return actor_loss

class QNetwork(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.env = env
        self.alpha = args.alpha

        def init_layer(layer, gain_ort=np.sqrt(2), gain=1):
            if args.layer_init == "default":
                return layer
            else:
                return layer_init(layer, args, gain_ort=gain_ort, gain=gain)

        self.critic = nn.Sequential(
            init_layer(nn.Linear(np.array(env.single_observation_space.shape).prod(), args.q_hidden_size)),
            getattr(nn, args.q_activation)(),
            *[layer for _ in range(args.q_num_hidden_layers) for layer in (
                init_layer(nn.Linear(args.q_hidden_size, args.q_hidden_size)),
                getattr(nn, args.q_activation)()
            )],
            init_layer(nn.Linear(args.q_hidden_size, env.single_action_space.n), gain_ort=args.q_last_layer_std, gain=args.q_last_layer_std),
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
    def __init__(self, env, args):
        super().__init__()

        def init_layer(layer, gain_ort=np.sqrt(2), gain=1):
            if args.layer_init == "default":
                return layer
            else:
                return layer_init(layer, args, gain_ort=gain_ort, gain=gain)

            
        self.actor = nn.Sequential(
            init_layer(nn.Linear(np.array(env.single_observation_space.shape).prod(), args.hidden_size)),
            getattr(nn, args.policy_activation)(),
            *[layer for _ in range(args.num_hidden_layers) for layer in (
                init_layer(nn.Linear(args.hidden_size, args.hidden_size)),
                getattr(nn, args.policy_activation)()
            )],
            init_layer(nn.Linear(args.hidden_size, env.single_action_space.n), gain_ort=args.actor_last_layer_std, gain=args.actor_last_layer_std),
        )

    def forward(self, x):
        return self.actor(x)

    def get_action(self, x, action=None):
        logits = self(x)
        policy_dist = Categorical(logits=logits)
        if action is None: action = policy_dist.sample()
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        action_log_prob = policy_dist.log_prob(action)
        return action, action_log_prob, log_prob, action_probs
    
class Sampler(nn.Module):
    def __init__(self, args, N):
        super().__init__()
        self.n = N
            
        self.z = nn.Sequential(
            (nn.Linear(N, args.sampler_hidden_size)),
            getattr(nn, args.sampler_activation)(),
            *[layer for _ in range(args.sampler_num_hidden_layers) for layer in (
                (nn.Linear(args.sampler_hidden_size, args.sampler_hidden_size)),
                getattr(nn, args.sampler_activation)()
            )],
            (nn.Linear(args.sampler_hidden_size, N)),
        )

    def forward(self, x):
        return self.z(x)

    def get_probs(self, x):
        logits = self(x)
        sampler_dist = Categorical(logits=logits)
        return sampler_dist.probs
    
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


def main(config):
    import torch
    import torch.optim as optim
    
    args = argparse.Namespace(**config)
    args.seed = config["__trial_index__"] + SEED_OFFSET
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    args.minibatch_size = args.total_iterations // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.total_iterations
    args.num_steps = args.total_iterations // args.num_envs
    logging_callback = args.logging_callback

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

    actor = QREPSPolicy(envs, args).to(device)
    qf = QNetwork(envs, args).to(device)

    if args.target_network:
        qf_target = QNetwork(envs, args).to(device)
        qf_target.load_state_dict(qf.state_dict())

    if args.q_optimizer == "Adam" or args.q_optimizer == "RMSprop":
        q_optimizer = getattr(optim, args.q_optimizer)(
            list(qf.parameters()), lr=args.q_lr, eps=args.eps
        )
    else:
        q_optimizer = getattr(optim, args.q_optimizer)(
            list(qf.parameters()), lr=args.q_lr
        )
    if args.actor_optimizer == "Adam" or args.actor_optimizer == "RMSprop":
        actor_optimizer = getattr(optim, args.actor_optimizer)(
            list(actor.parameters()), lr=args.policy_lr, eps=args.eps
        )
    else:

        actor_optimizer = getattr(optim, args.actor_optimizer)(
            list(actor.parameters()), lr=args.policy_lr
        )
    alpha = args.alpha
    if args.eta is None: eta = args.alpha
    else: eta = torch.Tensor([args.eta]).to(device)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs, envs.single_action_space.n)).to(device)
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
            lrnow = frac * args.policy_lr
            actor_optimizer.param_groups[0]["lr"] = lrnow

            lrnow = frac * args.q_lr
            q_optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():        
                action, _, logprob, _ = actor.get_action(next_obs)

            actions[step] = action
            logprobs[step] = logprob

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
                        reward_iteration.append(info["episode"]["r"])

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_next_obs = next_observations.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape((-1, envs.single_action_space.n))
        b_rewards = rewards.flatten()
        b_dones = dones.flatten()
        b_inds = np.arange(args.total_iterations)

        weights_after_each_epoch = []

        if len(reward_iteration) > 5: 
            logging_callback(np.mean(reward_iteration))
            reward_iteration = []

        np.random.shuffle(b_inds)
        for start in range(0, args.total_iterations, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            if args.saddle_point_optimization:
                if args.parametrized_sampler:
                    sampler = Sampler(args, N=b_obs[mb_inds].shape[0]).to(device)

                    if args.sampler_optimizer == "Adam" or args.sampler_optimizer == "RMSprop":
                        sampler_optimizer = getattr(optim, args.sampler_optimizer)(
                            list(sampler.parameters()), lr=args.beta, eps=args.eps
                        )
                    else:
                        sampler_optimizer = getattr(optim, args.sampler_optimizer)(
                            list(sampler.parameters()), lr=args.beta
                        )
                else:
                    sampler = ExponentiatedGradientSampler(b_obs[mb_inds].shape[0], device, eta, args.beta)

            for epoch in range(args.update_epochs):            
                
                if args.target_network:
                    delta = b_rewards[mb_inds].squeeze() + args.gamma * qf_target.get_values(b_next_obs[mb_inds], policy=actor)[1].detach() * (1 - b_dones[mb_inds].squeeze()) - qf.get_values(b_obs[mb_inds], b_actions[mb_inds], actor)[0]          
                
                else: delta = b_rewards[mb_inds].squeeze() + args.gamma * qf.get_values(b_next_obs[mb_inds], policy=actor)[1] * (1 - b_dones[mb_inds].squeeze()) - qf.get_values(b_obs[mb_inds], b_actions[mb_inds], actor)[0]

                if args.normalize_delta: delta = (delta - delta.mean()) / (delta.std() + 1e-9)

                if args.saddle_point_optimization:
                    bellman = delta.detach()
                    if args.parametrized_sampler: z_n = sampler.get_probs(bellman) 
                    else: z_n = sampler.probs() 
                    critic_loss = torch.sum(z_n.detach() * (delta - eta * torch.log(sampler.n * z_n.detach()))) + (1 - args.gamma) * qf.get_values(b_obs[mb_inds], b_actions[mb_inds], actor)[1].mean()
                
                else: 
                    critic_loss = eta * torch.log(torch.mean(torch.exp(delta / eta), 0)) + torch.mean((1 - args.gamma) * qf.get_values(b_obs[mb_inds], b_actions[mb_inds], actor)[1], 0)

                q_optimizer.zero_grad()
                critic_loss.backward()
                q_optimizer.step()

                if args.saddle_point_optimization:

                    if args.parametrized_sampler:
                        sampler_loss = - (torch.sum(z_n * (bellman - eta * torch.log(sampler.n * z_n))) + (1 - args.gamma) * qf.get_values(b_obs[mb_inds], b_actions[mb_inds], actor)[1].mean().detach())
                        
                        sampler_optimizer.zero_grad()
                        sampler_loss.backward()
                        sampler_optimizer.step()

                    else: sampler.update(bellman)

                if args.use_kl_loss: actor_loss = kl_loss(alpha, b_obs[mb_inds], b_next_obs[mb_inds], b_rewards[mb_inds], b_actions[mb_inds], b_logprobs[mb_inds], qf, actor)
                else: actor_loss = nll_loss(alpha, b_obs[mb_inds], b_next_obs[mb_inds], b_rewards[mb_inds], b_actions[mb_inds], b_logprobs[mb_inds], qf, actor)
                
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

            if args.average_critics: weights_after_each_epoch.append(deepcopy(qf.state_dict()))
        
        if args.average_critics:
            avg_weights = {}
            for key in weights_after_each_epoch[0].keys():
                avg_weights[key] = sum(T[key] for T in weights_after_each_epoch) / len(weights_after_each_epoch)
            qf.load_state_dict(avg_weights)

        if args.target_network and iteration % args.target_network_frequency == 0:
            for param, target_param in zip(qf.parameters(), qf_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
    except:
        logging_callback(-1.0)

    if len(reward_iteration) > 0:
        logging_callback(np.mean(reward_iteration))

    envs.close()
    writer.close()



search_alg = HEBOSearch(metric="reward", mode="max")
re_search_alg = Repeater(search_alg, repeat=2)

# search_alg = OptunaSearch(metric="reward", mode="max")
# search_alg = ConcurrencyLimiter(search_alg, max_concurrent=4)
# re_search_alg = Repeater(search_alg, repeat=3)


start_time = time.time()
analysis = tune.run(
    main,
    num_samples=800,
    config=config,
    search_alg=re_search_alg,
    local_dir="/Users/nicolasvila/workplace/uni/tfg_v2/tests/results_tune",
)
print("Time taken: ", time.time() - start_time)

df = analysis.results_df
df.to_csv("CartPole_Qreps_main.csv")