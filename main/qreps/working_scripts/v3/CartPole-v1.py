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
from replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 0
    """seed of the experiment"""
    run_multiple_seeds: bool = False
    """if toggled, this script will run with multiple seeds"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "QREPS_LunarLander-v2"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "LunarLander-v2"
    """the id of the environment"""
    total_timesteps: int = 1000
    """total timesteps of the experiments"""
    num_updates: int = 50
    """the number of updates"""
    num_rollouts: int = 5
    """the number of rollouts before each update"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e4)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    policy_lr: float = 0.02
    """the learning rate of the policy network optimizer"""
    policy_lr_end: float = 0.001
    """the last learning rate of the policy network optimizer"""
    q_lr: float = 0.02
    """the learning rate of the Q network network optimizer"""
    q_lr_end: float = 0.001
    """the last learning rate of the Q network network optimizer"""
    alpha: float = 2.0
    """Entropy regularization coefficient."""
    eta = None
    """coefficient for the kl reg"""
    update_epochs: int = 50
    """the number of epochs for the policy and value networks"""
    update_policy_epochs: int = 300
    """the number of epochs for the policy network"""
    beta: float = 4e-5
    """the sampler step size"""
    autotune: bool =  False
    """automatic tuning of the entropy coefficient"""
    target_entropy_scale: float = 0.2000
    """coefficient for scaling the autotune entropy target"""
    use_linear_schedule: bool = True
    """if toggled, the learning rate will decrease linearly"""
    saddle_point_optimization: bool = True
    """if toggled, the saddle point optimization will be used"""
    use_kl_loss: bool = False
    """if toggled, the kl loss will be used"""

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
    actor_loss = torch.mean(probs * (alpha * (newlogprob-log_likes) - q_values.detach()))
    return actor_loss

def optimize_critic(eta, observations, next_observations, actions, rewards, q_net, policy, gamma, sampler, optimizer, steps=300, loss_fn=ELBE):
    def closure():
        optimizer.zero_grad()
        loss = loss_fn(eta, observations, next_observations, actions, rewards, q_net , policy, gamma, sampler)
        loss.backward()
        if sampler is not None: sampler.update(empirical_bellman_error(observations, next_observations, actions, rewards, q_net, policy, gamma))
        # nn.utils.clip_grad_norm_([param for group in optimizer.param_groups for param in group['params']], 1.0)
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
    




def main(args):
    import stable_baselines3 as sb3

    # assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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

    q_optimizer = optim.SGD(list(qf.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

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
        log_likes
        ) = rb.get_all()
        # rewards = rewards

        if args.saddle_point_optimization:
            sampler = ExponentiatedGradientSampler(observations.shape[0], device, eta, beta=args.beta)
            optimize_critic(eta, observations, next_observations, actions, rewards, qf, actor, args.gamma, sampler, q_optimizer, steps=args.update_epochs, loss_fn=saddle)
        else:
            optimize_critic(eta, observations, next_observations, actions, rewards, qf, actor, args.gamma, None, q_optimizer, steps=args.update_epochs, loss_fn=ELBE)

        if args.use_kl_loss: optimize_actor(alpha, observations, next_observations, rewards, actions, log_likes, qf, actor, actor_optimizer, steps=args.update_policy_epochs, loss_fn=kl_loss)
        else: optimize_actor(alpha, observations, next_observations, rewards, actions, log_likes, qf, actor, actor_optimizer, steps=args.update_policy_epochs, loss_fn=nll_loss)

        rb.reset()
        print("Iteation:", T, "reward:", np.sum(all_rewards)/(args.num_rollouts*args.num_envs))
        writer.add_scalar("charts/episodic_return", np.sum(all_rewards)/(args.num_rollouts*args.num_envs), T)

        
        if args.autotune:
            actions, a_loglike, loglikes, probs = actor.get_action(torch.Tensor(obs).to(device))
            # re-use action probabilities for temperature loss
            alpha_loss = (probs.detach() * (-log_alpha.exp() * (loglikes + target_entropy).detach())).mean()

            a_optimizer.zero_grad()
            alpha_loss.backward()
            a_optimizer.step()
            alpha = log_alpha.exp().item()

    envs.close()
    writer.close()
    return np.sum(all_rewards)/(args.num_rollouts*args.num_envs)

# ### HP SEARCH
args = tyro.cli(Args)

if args.run_multiple_seeds:
    rewards = []
    n_seeds = 10

    for i in range(n_seeds):
        args.seed = i
        reward = main(args)
        rewards.append(reward)
    print("Average reward:", np.mean(rewards), "stddev:", np.std(rewards))

else:
    main(args)

# use_linear_schedule = False # False
# saddle_point_optimization = True # False
# use_kl_loss = True # True

# alphas = [0.2, 2, 3, 5]
# q_lrs = [0.1, 2.5e-2, 2.5e-3]
# policy_lrs = [0.1, 2.5e-2, 2.5e-3]

# update_epochs = 100
# update_policy_epochs = 300
# gammas = 0.99
# num_envs = 4
# num_rollouts = 5

# results = []

# for alpha, q_lr, policy_lr in itertools.product(alphas, q_lrs, policy_lrs):
#     args.alpha = alpha
#     args.q_lr = q_lr
#     args.policy_lr = policy_lr
#     args.update_epochs = update_epochs
#     args.update_policy_epochs = update_policy_epochs
#     args.gamma = gammas
#     args.num_envs = num_envs
#     args.num_rollouts = num_rollouts
#     args.use_kl_loss = use_kl_loss
#     args.use_linear_schedule = use_linear_schedule
#     args.saddle_point_optimization = saddle_point_optimization

#     try: 
#         print("Iteration with hyperparameters:", alpha, q_lr, policy_lr)
#         reward = main(args)
#         results.append((alpha, q_lr, policy_lr, reward))
#     except:
#         pass

# best_combination = max(results, key=lambda x: x[3])
# print("Best combination:", best_combination)

# df = pd.DataFrame(results, columns=["Alpha", "Q Learning Rate", "Policy Learning Rate", "Reward"])
# df.to_csv("results_qreps_elbe_nll_lunar_lander.csv", index=False)