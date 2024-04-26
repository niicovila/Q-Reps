# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
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
    wandb_project_name: str = "CartPole-v1-QREPS-Benchmark"
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
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    gamma: float = 0.99
    """the discount factor gamma"""

    q_lr_start: float = 0.00020571135648186006
    """the learning rate of the Q network network optimizer"""
    beta: float = 0.00016618354222092168
    """coefficient for the saddle point optimization"""

    update_epochs: int = 300
    """the number of epochs for the policy and value networks"""

    # Network params
    q_activation: str = "Tanh"
    """the activation function of the Q network"""
    q_hidden_size: int = 512
    """the hidden size of the Q network"""
    q_num_hidden_layers: int = 4
    """the number of hidden layers of the Q network"""

    target_network_frequency: int = 2
    """the frequency of updating the target network"""
    tau: float = 1.0
    """the soft update coefficient"""

    # Optimizer params
    q_optimizer: str = "Adam"
    """the optimizer of the Q network"""
    eps: float = 1e-8
    """the epsilon value for the optimizer"""

    # Options
    anneal_lr: bool = False
    """if toggled, the learning rate will decrease linearly"""
    q_histogram: bool = False
    """if toggled, the q function histogram will be plotted"""
    target_network: bool = False
    """if toggled, the target network will be used"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

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


if __name__ == "__main__":

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    args.batch_size = int(args.num_envs * args.num_steps)
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
    rewards_df = pd.DataFrame(columns=["Step", "Reward"])
    reward_iteration = []

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

        b_rewards = rewards.flatten()
        b_dones = dones.flatten()

        if len(reward_iteration) > 5: 
            print(f"Iteration: {global_step}, Avg Reward: {np.mean(reward_iteration)}")
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

        actor.set_q(qf)

        if args.target_network and iteration % args.target_network_frequency == 0:
            for param, target_param in zip(qf.parameters(), qf_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        
    rewards_df.to_csv(f"rewards_{run_name}.csv")
    envs.close()
    writer.close()


