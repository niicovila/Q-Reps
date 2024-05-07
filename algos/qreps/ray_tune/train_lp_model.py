# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse 
from copy import deepcopy
import random
import time
import gymnasium as gym
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from ray import train
from utils import make_env
from common_utils import Sampler, nll_loss
import torch
import torch.nn as nn
from utils import layer_init
import logging
from utils_lp import Policy, ExponentiatedGradientSampler

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
SEED_OFFSET = 1

class QNetwork(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.env = env
        self.alpha = args.alpha
        self.use_policy = args.use_policy
        self.policy_opt = args.policy_opt
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

        if self.policy_opt:
            q = self(x)
            z = q / self.alpha
            if self.use_policy:
                if policy is None: pi_k = torch.ones(x.shape[0], self.env.single_action_space.n, device=x.device) / self.env.single_action_space.n
                else: _, _, _, pi_k = policy.get_action(x); pi_k = pi_k.detach()
                v = self.alpha * (torch.log(torch.sum(pi_k * torch.exp(z), dim=1))).squeeze(-1)
            else:
                v = self.alpha * torch.log(torch.mean(torch.exp(z), dim=1)).squeeze(-1)
            if action is None:
                return q, v
            else:
                q = q.gather(-1, action.unsqueeze(-1).long()).squeeze(-1)
                return q, v
        else:
            q = self(x)
            _, probs = policy.get_action(x)
            v = (torch.sum(probs * q, dim=1)).squeeze(-1)
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


def lp_algo(config):
    import torch
    import torch.optim as optim
    
    args = argparse.Namespace(**config)
    if "__trial_index__" in config: args.seed = config["__trial_index__"] + SEED_OFFSET
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    args.minibatch_size = args.total_iterations // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.total_iterations
    args.num_steps = args.total_iterations // args.num_envs

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


    qf = QNetwork(envs, args).to(device)

    if args.policy_opt: 
        actor = QREPSPolicy(envs, args).to(device)
        if args.actor_optimizer == "Adam" or args.actor_optimizer == "RMSprop":
            actor_optimizer = getattr(optim, args.actor_optimizer)(
                list(actor.parameters()), lr=args.policy_lr, eps=args.eps
            )
        else:
            actor_optimizer = getattr(optim, args.actor_optimizer)(
                list(actor.parameters()), lr=args.policy_lr
            )
    else: actor = Policy(qf)

    if args.q_optimizer == "Adam" or args.q_optimizer == "RMSprop":
        q_optimizer = getattr(optim, args.q_optimizer)(
            list(qf.parameters()), lr=args.q_lr, eps=args.eps
        )
    else:
        q_optimizer = getattr(optim, args.q_optimizer)(
            list(qf.parameters()), lr=args.q_lr
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
    if args.target_network:
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)
        qs = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    reward_iteration = []
    if args.save_learning_curve: rewards_df = pd.DataFrame(columns=["Step", "Reward"])

    #try:
    for iteration in range(1, args.num_iterations + 1):

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.policy_lr
            if args.policy_opt: actor_optimizer.param_groups[0]["lr"] = lrnow

            lrnow = frac * args.q_lr
            q_optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs


            # ALGO LOGIC: action logic
            with torch.no_grad():        
                if args.policy_opt: action, _, logprob, _ = actor.get_action(next_obs)
                else: action, _ = actor.get_action(next_obs)

                if args.target_network:
                    q, v = qf.get_values(next_obs, action, actor)
                    qs[step] = q
                    values[step] = v

            actions[step] = action
            if args.policy_opt: logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            reward = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            
            next_observations[step] = next_obs
            dones[step] = next_done

            if "final_info" in infos:
                rs = []
                ls = []
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        ls.append(info["episode"]["l"])
                        rs.append(info["episode"]["r"])
                        reward_iteration.append(info["episode"]["r"])

                if len(rs)>0:
                    writer.add_scalar("charts/episodic_length", np.mean(ls), global_step)
                    writer.add_scalar("charts/episodic_return", np.mean(rs), global_step)
                    
                    if args.save_learning_curve: 
                        rewards_df = rewards_df._append({"Step": global_step, "Reward": np.mean(rs)}, ignore_index=True)
                        print(f'Iteration: {global_step}, Reward: {np.mean(rs)}')

        if len(reward_iteration) > 5: 
            logging_callback(np.mean(reward_iteration))
            print(f'Iteration: {global_step}, Reward: {np.mean(reward_iteration)}')
            reward_iteration = []

        if args.target_network:
            with torch.no_grad():
                next_value = qf.get_values(next_obs, policy=actor)[1]
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - qs[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            b_returns = returns.reshape(-1)

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_next_obs = next_observations.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        if args.policy_opt: b_logprobs = logprobs.reshape((-1, envs.single_action_space.n))
        b_rewards = rewards.flatten()
        b_dones = dones.flatten()
        b_inds = np.arange(args.total_iterations)

        weights_after_each_epoch = []
        
        if args.policy_opt:
            if not args.gae: np.random.shuffle(b_inds)
            for start in range(0, args.total_iterations, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                if args.parametrized_sampler:
                    sampler = Sampler(args, N=args.minibatch_size).to(device)

                    if args.sampler_optimizer == "Adam" or args.sampler_optimizer == "RMSprop":
                        sampler_optimizer = getattr(optim, args.sampler_optimizer)(
                            list(sampler.parameters()), lr=args.beta, eps=args.eps
                        )
                    else:
                        sampler_optimizer = getattr(optim, args.sampler_optimizer)(
                            list(sampler.parameters()), lr=args.beta
                        )
                else:
                    sampler = ExponentiatedGradientSampler(args, args.minibatch_size, device, eta, args.beta)
                
                for epoch in range(args.update_epochs):       
                    q_val, val = qf.get_values(b_obs[mb_inds], b_actions[mb_inds], actor)
                    values_next = qf.get_values(b_next_obs[mb_inds], b_actions[mb_inds], actor)[1]

                    if args.target_network:
                        delta =  b_returns[mb_inds] - q_val  
                    
                    elif args.gae:
                        delta = torch.zeros_like(b_rewards[mb_inds]).to(device)
                        lastgaelam = 0
                        for t in reversed(range(args.minibatch_size)):
                            nextnonterminal = 1.0 - b_dones[mb_inds][t]
                            nextvalues = values_next[t]
                            delta_t = b_rewards[mb_inds][t] + args.gamma * nextvalues * nextnonterminal - q[t]
                            delta[t] = lastgaelam = delta_t + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam

                    else: delta = b_rewards[mb_inds].squeeze() + args.gamma * values_next * (1 - b_dones[mb_inds].squeeze()) - q_val

                    if args.normalize_delta: delta = (delta - delta.mean()) / (delta.std() + 1e-8)

                    bellman = delta.detach()
                    if args.parametrized_sampler: z_n = sampler.get_probs(bellman) 
                    else: z_n = sampler.probs() 
                    
                    critic_loss = torch.sum(z_n.detach() * (delta - eta * torch.log(sampler.n * z_n.detach()))) + (1 - args.gamma) * val.mean()
                
                    q_optimizer.zero_grad()
                    critic_loss.backward()
                    q_optimizer.step()

                    if args.parametrized_sampler:
                        sampler_loss = - (torch.sum(z_n * (bellman - eta * torch.log(sampler.n * z_n))) + (1 - args.gamma) * val.mean().detach())
                        
                        sampler_optimizer.zero_grad()
                        sampler_loss.backward()
                        sampler_optimizer.step()

                    else: sampler.update(bellman)

                    if args.joint_opt:
                        if args.use_kl_loss: 
                            _, _, newlogprob, probs = actor.get_action(b_obs[mb_inds])
                            with torch.no_grad():
                                q_state_action = qf.get_values(b_obs[mb_inds], policy=actor)[0]
                                adv = q_state_action - val.unsqueeze(1)
                            actor_loss = torch.mean(probs * (alpha * (newlogprob-b_logprobs[mb_inds].detach()) - adv))

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
            
            if not args.joint_opt:
                for epoch in range(args.update_epochs_policy):
                    np.random.shuffle(b_inds)
                    for start in range(0, args.total_iterations, args.minibatch_size):
                            end = start + args.minibatch_size
                            mb_inds = b_inds[start:end]

                            if args.use_kl_loss:
                                _, _, newlogprob, probs = actor.get_action(b_obs[mb_inds])
                                with torch.no_grad():
                                    q_state_action, val = qf.get_values(b_obs[mb_inds], policy=actor)
                                    adv = q_state_action - val.unsqueeze(1)
                                actor_loss = torch.mean(probs * (alpha * (newlogprob-b_logprobs[mb_inds].detach()) - adv))
                            
                            else: actor_loss = nll_loss(alpha, b_obs[mb_inds], b_next_obs[mb_inds], b_rewards[mb_inds], b_actions[mb_inds], b_logprobs[mb_inds], qf, actor)
                            
                            actor_optimizer.zero_grad()
                            actor_loss.backward()
                            actor_optimizer.step()
        else:
            if not args.gae: np.random.shuffle(b_inds)
            for start in range(0, args.total_iterations, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                if args.parametrized_sampler:
                    sampler = Sampler(args, N=args.minibatch_size).to(device)

                    if args.sampler_optimizer == "Adam" or args.sampler_optimizer == "RMSprop":
                        sampler_optimizer = getattr(optim, args.sampler_optimizer)(
                            list(sampler.parameters()), lr=args.beta, eps=args.eps
                        )
                    else:
                        sampler_optimizer = getattr(optim, args.sampler_optimizer)(
                            list(sampler.parameters()), lr=args.beta
                        )
                else:
                    sampler = ExponentiatedGradientSampler(args, args.minibatch_size, device, eta, args.beta)
                
                for epoch in range(args.update_epochs):       
                    q_val, val = qf.get_values(b_obs[mb_inds], b_actions[mb_inds], actor)
                    values_next = qf.get_values(b_next_obs[mb_inds], b_actions[mb_inds], actor)[1]

                    if args.target_network:
                        delta =  b_returns[mb_inds] - q_val  

                    elif args.gae:
                        delta = torch.zeros_like(b_rewards[mb_inds]).to(device)
                        lastgaelam = 0
                        for t in reversed(range(args.minibatch_size)):
                            nextnonterminal = 1.0 - b_dones[mb_inds][t]
                            nextvalues = values_next[t]
                            delta_t = b_rewards[mb_inds][t] + args.gamma * nextvalues * nextnonterminal - q[t]
                            delta[t] = lastgaelam = delta_t + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam

                    else: delta = b_rewards[mb_inds].squeeze() + args.gamma * values_next * (1 - b_dones[mb_inds].squeeze()) - q_val

                    if args.normalize_delta: delta = (delta - delta.mean()) / (delta.std() + 1e-8)

                    bellman = delta.detach()
                    if args.parametrized_sampler: z_n = sampler.get_probs(bellman) 
                    else: z_n = sampler.probs() 
                    
                    critic_loss = torch.sum(z_n.detach() * delta) + (1 - args.gamma) * val.mean()
                
                    q_optimizer.zero_grad()
                    critic_loss.backward()
                    q_optimizer.step()

                    if args.parametrized_sampler:
                        sampler_loss = - (torch.sum(z_n * bellman) + (1 - args.gamma) * val.mean().detach())
                        
                        sampler_optimizer.zero_grad()
                        sampler_loss.backward()
                        sampler_optimizer.step()

                    else: sampler.update(bellman)

                if args.average_critics: weights_after_each_epoch.append(deepcopy(qf.state_dict()))
            
            if args.average_critics:
                avg_weights = {}
                for key in weights_after_each_epoch[0].keys():
                    avg_weights[key] = sum(T[key] for T in weights_after_each_epoch) / len(weights_after_each_epoch)
                qf.load_state_dict(avg_weights)
             
            actor.set_q(qf)

    if len(reward_iteration) > 0:
        logging_callback(np.mean(reward_iteration))

    envs.close()
    writer.close()

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
    "policy_opt": False,
    "joint_opt": False,

    # Algorithm
    "total_timesteps": 100000,
    "num_envs": 16,
    "gamma": 0.99,

    "total_iterations": 1024,
    "num_minibatches": 16,
    "update_epochs": 50,
    "update_epochs_policy": 50,

    "alpha": 16,  
    "eta": 16,

    # Learning rates
    "beta": 3e-4,
    "policy_lr": 3e-3,
    "q_lr": 3e-4,
    "anneal_lr": True,

    # Layer Init
    "layer_init": "kaiming_uniform",
    # Architecture
    "policy_activation": "Tanh",
    "num_hidden_layers": 2,
    "hidden_size": 128,
    "actor_last_layer_std": 0.01,

    "q_activation": "Tanh",
    "q_num_hidden_layers": 4,
    "q_hidden_size": 128,
    "q_last_layer_std": 1.0,

    "average_critics": True,
    "use_policy": False,

    "parametrized_sampler" : False,
    "sampler_activation": "Tanh",
    "sampler_num_hidden_layers": 2,
    "sampler_hidden_size": 128,
    "sampler_last_layer_std": 0.01,


    # Optimization
    "q_optimizer": "Adam",  # "Adam", "SGD", "RMSprop
    "actor_optimizer": "Adam", 
    "sampler_optimizer": "Adam",
    "eps": 1e-8,

    # Options
    "normalize_delta": True,
    "gae": True,
    "gae_lambda": 0.95,
    "use_kl_loss": True,
    "q_histogram": False,
    "save_learning_curve": False,
    "target_network": True,
    "tau": 1.0,
    "target_network_frequency": 0, 

    "minibatch_size": 0,
    "num_iterations": 0,
    "num_steps": 0,
}



lp_algo(config)