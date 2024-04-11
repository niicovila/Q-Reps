import argparse
import random
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
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



config = {
    "exp_name": "QREPS",
    "seed": 0,
    "torch_deterministic": True,
    "cuda": True,
    "track": False,
    "wandb_project_name": "QREPS_cleanRL",
    "wandb_entity": None,
    "capture_video": False,
    "env_id": "LunarLander-v2",
    "total_timesteps": 50000,
    "num_envs": tune.choice([4, 8]),
    "num_steps": tune.choice([128, 256, 500]),
    "anneal_lr": tune.choice([True, False]),
    "gamma": 0.99,
    "num_minibatches": tune.choice([2, 4, 8, 16]),
    "policy_lr_start": tune.choice([0.1, 0.001, 2.5e-3, 2.5e-4]),
    "q_lr_start": tune.choice([0.1, 0.001, 2.5e-3, 2.5e-4]),
    "alpha": tune.choice([0.5, 1.5, 2, 4, 6, 8]),
    "eta": None,
    "beta": tune.choice([0.01, 0.1, 2e-3, 2e-4, 4e-5]),
    "update_epochs": tune.choice([10, 50, 100, 300]),
    "autotune": tune.choice([True, False]),
    "target_entropy_scale": tune.choice([0.3, 0.5, 0.7, 0.89]),
    "saddle_point_optimization": True,
    "use_kl_loss": tune.choice([True, False]),
    "sampler": ExponentiatedGradientSampler,
    "parametrized_sampler" : True,
    "sampler_lr": tune.choice([0.1, 0.01, 0.001, 0.0001]),
    "q_histogram": False,
    "batch_size": 0,
    "minibatch_size": 0,
    "num_iterations": 0
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

def nll_loss(alpha, observations, next_observations, rewards, actions, log_likes, q_net, policy):
    weights = torch.clamp(q_net.get_values(observations, actions, policy)[0] / alpha, -20, 20)
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
    
class Sampler(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.n = N
        self.z = nn.Sequential(
            layer_init(nn.Linear(N, 56)),
            nn.Tanh(),
            layer_init(nn.Linear(56, 56)),
            nn.Tanh(),
            layer_init(nn.Linear(56, N), std=0.01),
        )

    def forward(self, x):
        return self.z(x)

    def get_probs(self, x):
        logits = self(x)
        sampler_dist = Categorical(logits=logits)
        return sampler_dist.probs
    
def main(config):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    args = argparse.Namespace(**config)
    args.seed = config["__trial_index__"] + SEED_OFFSET
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    logging_callback=lambda r: train.report({'reward':r})

    # assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
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

    actor = QREPSPolicy(envs).to(device)
    qf = QNetwork(envs, args).to(device)

    q_optimizer = optim.Adam(list(qf.parameters()), lr=args.q_lr_start, eps=1e-4)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr_start, eps=1e-4)
 
    if args.autotune:
        target_entropy = -args.target_entropy_scale * torch.log(1 / torch.tensor(envs.single_action_space.n))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr_start, eps=1e-4)
    else:
        alpha = args.alpha

    if args.eta is None: eta = args.alpha
    else: eta = args.eta

    start_time = time.time()

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs, envs.single_action_space.n)).to(device)
    # values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # qs = torch.zeros((args.num_steps, args.num_envs, envs.single_action_space.n)).to(device)
    next_observations = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)  # Added this line
    
    if args.eta is None: eta = args.alpha
    else: eta = torch.Tensor([args.eta]).to(device)
    
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    try: 
     for iteration in range(1, args.num_iterations + 1):
        reward_iteration = []
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
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        reward_iteration.append(info["episode"]["r"])
                        # logging_callback(info["episode"]["r"])

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_next_obs = next_observations.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape((-1, envs.single_action_space.n))

        b_rewards = rewards.flatten()
        b_dones = dones.flatten()
        b_inds = np.arange(args.batch_size)

        if len(reward_iteration) > 0: logging_callback(np.mean(reward_iteration))

        if args.saddle_point_optimization:
            if args.parametrized_sampler:
                sampler = Sampler(N=b_obs.shape[0]).to(device)
                sampler_optimizer = optim.Adam(list(sampler.parameters()), lr=args.sampler_lr, eps=1e-4)
            else:
                sampler = ExponentiatedGradientSampler(b_obs.shape[0], device, eta, args.beta)

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            
            delta = b_rewards.squeeze() + args.gamma * qf.get_values(b_next_obs, policy=actor)[1] * (1 - b_dones.squeeze()) - qf.get_values(b_obs, b_actions, actor)[0]

            if args.saddle_point_optimization:
                bellman = delta.detach()
                if args.parametrized_sampler: z_n = sampler.get_probs(bellman) 
                else: z_n = sampler.probs() 
                critic_loss = torch.sum(z_n.detach() * (delta - eta * torch.log(sampler.n * z_n.detach()))) + (1 - args.gamma) * qf.get_values(b_obs, b_actions, actor)[1].mean()
            
            else: 
                # critic_loss = ELBE(eta, b_obs[mb_inds], b_next_obs[mb_inds], b_actions[mb_inds], b_rewards[mb_inds], qf, actor, args.gamma)
                critic_loss = eta * torch.log(torch.mean(torch.exp(delta / eta), 0)) + torch.mean((1 - args.gamma) * qf.get_values(b_obs, b_actions, actor)[1], 0)

            q_optimizer.zero_grad()
            critic_loss.backward()
            q_optimizer.step()

            if args.saddle_point_optimization:

                if args.parametrized_sampler:
                    sampler_loss = - (torch.sum(z_n * (bellman - eta * torch.log(sampler.n * z_n))) + (1 - args.gamma) * qf.get_values(b_obs, b_actions, actor)[1].mean().detach())
                    
                    sampler_optimizer.zero_grad()
                    sampler_loss.backward()
                    sampler_optimizer.step()

                else: sampler.update(bellman)

            if args.use_kl_loss: actor_loss = kl_loss(alpha, b_obs, b_next_obs, b_rewards, b_actions, b_logprobs, qf, actor)
            else: actor_loss = nll_loss(alpha, b_obs, b_next_obs, b_rewards, b_actions, b_logprobs, qf, actor)
            
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            if args.autotune:
                _, _, loglikes, probs = actor.get_action(torch.Tensor(b_obs).to(device))
                
                # re-use action probabilities for temperature loss
                alpha_loss = (probs.detach() * (-log_alpha.exp() * (loglikes + target_entropy).detach())).mean()

                a_optimizer.zero_grad()
                alpha_loss.backward()
                a_optimizer.step()
                alpha = log_alpha.exp().item()
    except:
        logging_callback(-1000)
    envs.close()
    writer.close()

search_alg = HEBOSearch(metric="reward", mode="max")
re_search_alg = Repeater(search_alg, repeat=1)

ray_init = {"cpu": 1}

analysis = tune.run(
    main,
    num_samples=100,
    config=config,
    search_alg=re_search_alg,
    resources_per_trial=ray_init,
    local_dir="/Users/nicolasvila/workplace/uni/tfg_v2/tests/results_tune",
)
df = analysis.results_df
df.to_csv("LunarLander_qreps_analysis_v2.csv")
