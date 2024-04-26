# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
from copy import deepcopy
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
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

Q_HIST = []
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 2
    """seed of the experiment"""
    run_multiple_seeds: bool = False
    """if toggled, this script will run with multiple seeds"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "CartPole-QREPS-Saddle-Benchmark"
    """the wandb's project name"""
    wandb_entity = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 100000
    """total timesteps of the experiments"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 300
    """the number of steps to run in each environment per policy rollout"""
    gamma: float = 0.99
    """the discount factor gamma"""
    update_epochs: int = 100
    """the number of epochs for the policy and value networks"""
    num_minibatches: int = 8
    """the number of minibatches to train the policy and value networks"""
    q_last_layer_std: float = 1.0
    """the standard deviation of the last layer of the Q network"""
    actor_last_layer_std: float = 0.01
    """the standard deviation of the last layer of the Q network"""
    layer_init: bool = True
    """if toggled, the layers will be initialized"""


    policy_lr_start: float = 0.0022370478556036
    """the learning rate of the policy network optimizer"""
    q_lr_start: float = 0.0023558824219659
    """the learning rate of the Q network network optimizer"""
    beta: float = 0.0070811761546232
    """coefficient for the saddle point optimization"""
    alpha: float = 8.0
    """Entropy regularization coefficient."""
    eta = None
    """coefficient for the kl reg"""

    
    # Network params
    policy_activation: str = "ReLU"
    """the activation function of the policy network"""
    hidden_size: int = 128
    """the hidden size of the policy network"""
    num_hidden_layers: int = 4
    """the number of hidden layers of the policy network"""
    q_activation: str = "Tanh"
    """the activation function of the Q network"""
    q_hidden_size: int = 128
    """the hidden size of the Q network"""
    q_num_hidden_layers: int = 4
    """the number of hidden layers of the Q network"""

    # Optimizer params
    q_optimizer: str = "SGD"
    """the optimizer of the Q network"""
    actor_optimizer: str = "Adam"
    """the optimizer of the policy network"""
    eps: float = 1e-8
    """the epsilon value for the optimizer"""

    # Options
    anneal_lr: bool = False
    """if toggled, the learning rate will decrease linearly"""
    saddle_point_optimization: bool = True
    """if toggled, the saddle point optimization will be used"""
    parametrized_sampler: bool = False
    """if toggled, the sampler will be parametrized"""
    use_kl_loss: bool = False
    """if toggled, the kl loss will be used"""
    q_histogram: bool = False
    """if toggled, the q function histogram will be plotted"""
    average_critics: bool = False   
    """if toggled, the critics will be averaged"""
    
    target_network: bool = False
    """if toggled, the target network will be used"""
    target_network_frequency: int = 100
    """the frequency of updating the target network"""
    tau: float = 1.0
    
    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    minibatch_size: int = 0
    """the minibatch size (computed in runtime)"""

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

        def init_layer(layer, std=np.sqrt(2)):
            if args.layer_init:
                return layer_init(layer, std=std)
            else:
                return layer

        self.critic = nn.Sequential(
            init_layer(nn.Linear(np.array(env.single_observation_space.shape).prod(), args.q_hidden_size)),
            getattr(nn, args.q_activation)(),
            *[layer for _ in range(args.q_num_hidden_layers) for layer in (
                init_layer(nn.Linear(args.q_hidden_size, args.q_hidden_size)),
                getattr(nn, args.q_activation)()
            )],
            init_layer(nn.Linear(args.q_hidden_size, env.single_action_space.n), std=args.q_last_layer_std),
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

        def init_layer(layer, std=np.sqrt(2)):
            if args.layer_init:
                return layer_init(layer, std=std)
            else:
                return layer
            
        self.actor = nn.Sequential(
            init_layer(nn.Linear(np.array(env.single_observation_space.shape).prod(), args.hidden_size)),
            getattr(nn, args.policy_activation)(),
            *[layer for _ in range(args.num_hidden_layers) for layer in (
                init_layer(nn.Linear(args.hidden_size, args.hidden_size)),
                getattr(nn, args.policy_activation)()
            )],
            init_layer(nn.Linear(args.hidden_size, env.single_action_space.n), std=args.actor_last_layer_std),
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
            layer_init(nn.Linear(N, args.sampler_hidden_size)),
            getattr(nn, args.sampler_activation)(),
            *[layer for _ in range(args.sampler_num_hidden_layers) for layer in (
                layer_init(nn.Linear(args.sampler_hidden_size, args.sampler_hidden_size)),
                getattr(nn, args.sampler_activation)()
            )],
            layer_init(nn.Linear(args.sampler_hidden_size, N), std=0.01),
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


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

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

    actor = QREPSPolicy(envs, args).to(device)
    qf = QNetwork(envs, args).to(device)

    if args.target_network:
        qf_target = QNetwork(envs, args).to(device)
        qf_target.load_state_dict(qf.state_dict())

    if args.q_optimizer == "Adam" or args.q_optimizer == "RMSprop":
        q_optimizer = getattr(optim, args.q_optimizer)(
            list(qf.parameters()), lr=args.q_lr_start, eps=args.eps
        )
    else:
        q_optimizer = getattr(optim, args.q_optimizer)(
            list(qf.parameters()), lr=args.q_lr_start
        )
    if args.actor_optimizer == "Adam" or args.actor_optimizer == "RMSprop":
        actor_optimizer = getattr(optim, args.actor_optimizer)(
            list(actor.parameters()), lr=args.policy_lr_start, eps=args.eps
        )
    else:

        actor_optimizer = getattr(optim, args.actor_optimizer)(
            list(actor.parameters()), lr=args.policy_lr_start
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
    rewards_df = pd.DataFrame(columns=["Step", "Reward"])

    for iteration in range(1, args.num_iterations + 1):

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.policy_lr_start
            actor_optimizer.param_groups[0]["lr"] = lrnow

            lrnow = frac * args.q_lr_start
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
                rs = []
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        reward_iteration.append(info["episode"]["r"])
                        rs.append(info["episode"]["r"])

                if len(rs)>0:
                    rewards_df = rewards_df._append({"Step": global_step, "Reward": np.mean(rs)}, ignore_index=True)

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_next_obs = next_observations.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape((-1, envs.single_action_space.n))

        b_rewards = rewards.flatten()
        b_dones = dones.flatten()
        b_inds = np.arange(args.batch_size)
        weights_after_each_epoch = []

        if len(reward_iteration) > 5: 
            print(f"Iteration {global_step}, mean episodic return: {np.mean(reward_iteration)}")
            reward_iteration = []

        np.random.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                if args.saddle_point_optimization:
                    if args.parametrized_sampler:
                        sampler = Sampler(args, N=b_obs[mb_inds].shape[0]).to(device)
                        sampler_optimizer = optim.Adam(list(sampler.parameters()), lr=args.beta)
                    else:
                        sampler = ExponentiatedGradientSampler(b_obs[mb_inds].shape[0], device, eta, args.beta)

                for epoch in range(args.update_epochs):            
                    if args.target_network:
                        delta = b_rewards[mb_inds].squeeze() + args.gamma * qf_target.get_values(b_next_obs[mb_inds], policy=actor)[1].detach() * (1 - b_dones[mb_inds].squeeze()) - qf.get_values(b_obs[mb_inds], b_actions[mb_inds], actor)[0]          
                    else: delta = b_rewards[mb_inds].squeeze() + args.gamma * qf.get_values(b_next_obs[mb_inds], policy=actor)[1] * (1 - b_dones[mb_inds].squeeze()) - qf.get_values(b_obs[mb_inds], b_actions[mb_inds], actor)[0]
        
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

    if len(reward_iteration) > 0:
        rewards_df = rewards_df._append({"Step": global_step, "Reward": np.mean(reward_iteration)}, ignore_index=True)
        print(f"Iteration: {global_step}, Avg Reward: {np.mean(reward_iteration)}")
    rewards_df.to_csv(f"rewards_{run_name}.csv")
    envs.close()
    writer.close()



    if args.q_histogram:
        import matplotlib.pyplot as plt
        Q_HIST = [item for array in Q_HIST for item in array.flatten()]
        plt.hist(Q_HIST, bins=30, edgecolor='black')  # Adjust the number of bins as needed
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Q Function Histogram')
        plt.grid(True)
        plt.savefig(f'histogram_{args.seed}.png')  # Change the file extension as needed
