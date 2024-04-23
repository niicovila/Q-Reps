# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_ataripy
import argparse
import os
import random
import time
from dataclasses import dataclass
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from ray import train
from ray.tune.search import Repeater
from ray.tune.search.hebo import HEBOSearch
import ray.tune as tune  # Import the missing package
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
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
    "total_timesteps": 5000000,
    "buffer_size": 10000,
    "batch_size": 64,
    "learning_starts": 2000,
    "update_frequency": 4,

    "update_epochs": tune.choice([5, 50, 100, 300]),
    "update_policy_epochs": tune.choice([50, 300, 450]),
    "num_rollouts": tune.choice([2, 5, 8]),
    "num_envs": tune.choice([1, 4, 6]),
    "gamma": 0.99,
    "policy_lr": tune.choice([0.1, 2e-2, 2.5e-3]),
    "q_lr": tune.choice([0.1, 2e-2, 2.5e-3]),
    "alpha":  tune.choice([0.2, 0.5, 2, 4, 6]),
    "eta": None,
    "beta": tune.choice([0.1, 0.01, 0.002, 4e-5]),
    "autotune":  True,
    "target_entropy_scale": tune.choice([0.2, 0.35, 0.5, 0.89]),
    "use_linear_schedule":  tune.choice([True, False]),
    "saddle_point_optimization":  tune.choice([True, False]),
    "use_kl_loss": tune.choice([True, False]),
    "target_network_frequency": tune.choice([2, 4, 8, 16]),
    "tau": 1.0,
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

def empirical_bellman_error(observations, next_observations, actions, rewards, qnet, policy, gamma):
    qf, qf2, qf_target, qf2_target = qnet
    with torch.no_grad():
        v_target = torch.min(qf_target.get_values(next_observations, policy=policy)[1], qf2_target.get_values(next_observations, policy=policy)[1])
    # v_target = qf.get_values(next_observations, actions, policy)[1]
    q_features_1 = qf.get_values(observations, actions, policy)[0]
    q_features_2 = qf2.get_values(observations, actions, policy)[0]

    loss_1 = rewards.reshape(-1) + gamma * v_target - q_features_1
    loss_2 = rewards.reshape(-1) + gamma * v_target - q_features_2
    loss = loss_1 + loss_2
    # Q_HIST.append(q_features_1.flatten().detach().numpy())
    return loss

def saddle(eta, observations, next_observations, actions, rewards, qnets, policy, gamma, sampler):
    qf, qf2, _, _ = qnets
    discount_term_1 = (1 - gamma) * qf.get_values(observations, actions, policy)[1].mean()
    discount_term_2 = (1 - gamma) * qf2.get_values(observations, actions, policy)[1].mean()
    discount_term = discount_term_1 + discount_term_2

    errors = torch.sum(sampler.probs().detach() * (empirical_bellman_error(observations, next_observations, actions, rewards, qnets, policy, gamma) - eta * torch.log(sampler.n * sampler.probs().detach()))) + discount_term
    return errors

def ELBE(eta, observations, next_observations, actions, rewards, qnets, policy, gamma, sampler=None):
    qf, qf2, _, _ = qnets
    discount_term_1 = (1 - gamma) * qf.get_values(observations, actions, policy)[1].mean()
    discount_term_2 = (1 - gamma) * qf2.get_values(observations, actions, policy)[1].mean()
    discount_term = discount_term_1 + discount_term_2
    print(discount_term.shape)
    errors = eta * torch.logsumexp(
        empirical_bellman_error(observations, next_observations, actions, rewards, qnets, policy, gamma) / eta, 0
    ) + discount_term
    print(errors)
    return errors

def nll_loss(alpha, observations, next_observations, rewards, actions, log_likes, q_net, policy):
    qf, qf2, _, _ = q_net
    with torch.no_grad():
        min_q = torch.min(qf.get_values(observations, actions, policy)[0], qf2.get_values(observations, actions, policy)[0])

    weights = torch.clamp(min_q / alpha, -20, 20)
    _, log_likes, _, _ = policy.get_action(observations, actions)
    nll = -torch.mean(torch.exp(weights) * log_likes)
    return nll

def kl_loss(alpha, observations, next_observations, rewards, actions, q_net, policy):
    qf, qf2, _, _ = q_net
    with torch.no_grad():
        min_q = torch.min(qf.get_values(observations, policy=policy)[0], qf2.get_values(observations, policy=policy)[0])
    _, _, newlogprob, probs = policy.get_action(observations, actions)
    actor_loss = torch.mean(probs * (alpha * (newlogprob) - min_q.detach()))
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
            nn.ReLU(),
            layer_init(nn.Linear(output_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, env.single_action_space.n), std=1),
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
            q = q.gather(-1, action.long()).squeeze(-1)
            return q, v
    

class QREPSPolicy(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
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

        self.fc1 = layer_init(nn.Linear(output_dim, 512))
        self.fc_logits = layer_init(nn.Linear(512, envs.single_action_space.n))

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.relu(self.fc1(x))
        logits = self.fc_logits(x)
        return logits


    def get_action(self, x, action=None):
        logits = self(x / 255.0)
        policy_dist = Categorical(logits=logits)
        if action is None: action = policy_dist.sample()
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1) # log_prob = torch.log(action_probs+1e-6)
        action_log_prob = policy_dist.log_prob(action)
        return action, action_log_prob, log_prob, action_probs


def main(config):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import stable_baselines3 as sb3

    # assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    args = argparse.Namespace(**config)
    args.seed = config["__trial_index__"] + SEED_OFFSET
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    logging_callback=lambda r: train.report({'reward':r})
    
    if args.eta is None: args.eta = args.alpha
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
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    
    if args.eta is None: eta = args.alpha
    else: eta = args.eta

    actor = QREPSPolicy(envs).to(device)
    qf = QNetwork(envs, args).to(device)
    qf2 = QNetwork(envs, args).to(device)

    qf_target = QNetwork(envs, args).to(device)
    qf2_target = QNetwork(envs, args).to(device)

    qf_target.load_state_dict(qf.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(list(qf.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, eps=1e-4)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -args.target_entropy_scale * torch.log(1 / torch.tensor(envs.single_action_space.n))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, eps=1e-4)
    else:
        alpha = args.alpha

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    try:
     for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if "episode" not in info:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                logging_callback(info["episode"]["r"])
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.update_frequency == 0:

                data = rb.sample(args.batch_size)
                # CRITIC training
                
                q_nets = [qf, qf2, qf_target, qf2_target]
                if args.saddle_point_optimization:
                    sampler = ExponentiatedGradientSampler(data.observations.shape[0], device, eta, beta=args.beta)
                    optimize_critic(eta, data.observations, data.next_observations, data.actions.long(), data.rewards, q_nets, actor, args.gamma, sampler, q_optimizer, steps=args.update_epochs, loss_fn=saddle)
                else:
                    optimize_critic(eta, data.observations, data.next_observations, data.actions.long(), data.rewards, q_nets, actor, args.gamma, None, q_optimizer, steps=args.update_epochs, loss_fn=ELBE)

                if args.use_kl_loss: optimize_actor(alpha, data.observations, data.next_observations, data.rewards, data.actions.long(), q_nets, actor, actor_optimizer, steps=args.update_policy_epochs, loss_fn=kl_loss)
                else: optimize_actor(alpha, data.observations, data.next_observations, data.rewards, data.actions.long(), None, q_nets, actor, actor_optimizer, steps=args.update_policy_epochs, loss_fn=nll_loss)
                
                if args.autotune:
                    _, _, log_pi, action_probs = actor.get_action(data.observations)

                    # re-use action probabilities for temperature loss
                    alpha_loss = (action_probs.detach() * (-log_alpha.exp() * (log_pi + target_entropy).detach())).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf.parameters(), qf_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                # writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                # writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                # writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                # writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                # writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                # writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
    except:
        logging_callback(0.0)
    envs.close()
    writer.close()


ray_init_config = {
    "num_gpus": 1,  # Adjust based on the number of available GPUs
    "num_cpus": 4,  # Number of CPU cores to allocate per trial
    # Additional Ray initialization options if needed
}

search_alg = HEBOSearch(metric="reward", mode="max")
re_search_alg = Repeater(search_alg, repeat=5)

analysis = tune.run(
    main,
    num_samples=200,
    config=config,
    search_alg=re_search_alg,
    # resources_per_trial=ray_init_config,
    local_dir="/Users/nicolasvila/workplace/uni/tfg_v2/tests/qreps/results_tune_qreps_v3",
)

print("Best config: ", analysis.get_best_config(metric="reward", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.results_df

df.to_csv("tuning_atari_v1.csv")