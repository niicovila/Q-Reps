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
from losses import empirical_logistic_bellman, S, log_gumbel
from agent import Agent

from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sampler import ExponentiatedGradientSampler, BestResponseSampler
from ray import train
from ray.tune.search import Repeater
from ray.tune.search.hebo import HEBOSearch
import ray.tune as tune  # Import the missing package



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
  "total_timesteps": 30,
  "learning_rate": tune.loguniform(2e-4, 2e-1),
  "policy_lr": tune.loguniform(2e-3, 2e-1),
  "num_envs": 5,
  "num_steps": 200,
  "anneal_lr": True,
  "gamma": 0.99,
  "num_minibatches": 4,
  "update_epochs": tune.choice([300, 450]),
  "update_policy_epochs": tune.choice([300, 450]),
  "max_grad_norm": 0.5,
  "alpha": tune.loguniform(2e-1, 10),
  "eta": 0.0,
  "parametrized": True,
  "saddle": False,
  "gumbel": False,
  "nll_loss": True,
  "sampler": BestResponseSampler,
  "average_critics": False,
  "batch_size": 0,
  "minibatch_size": 0,
  "num_iterations": 0
}

import logging
FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
SEED_OFFSET = 0

def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk



def main(config: dict):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    args.seed = config["__trial_index__"] + SEED_OFFSET
    args = argparse.Namespace(**config)
    
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps # // args.batch_size
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
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs, args).to(device)
    critic_optimizer = optim.SGD(agent.critic.parameters(), lr=args.learning_rate)
    actor_optimizer = optim.SGD(agent.actor.parameters(), lr=args.policy_lr)
    alpha = torch.Tensor([args.alpha]).to(device)
    if args.eta == 0: args.eta = args.alpha
    eta = torch.Tensor([args.eta]).to(device)
    if args.saddle: sampler = args.sampler(args.minibatch_size, device, eta=eta)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    qs = torch.zeros((args.num_steps, args.num_envs, envs.single_action_space.n)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    try: 
        for iteration in range(1, args.num_iterations + 1):
            
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                actor_optimizer.param_groups[0]["lr"] = lrnow
                critic_optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():        
                    action, action_logprob, log_probs, action_probs = agent.get_action(next_obs)
                    q, value = agent.get_value(next_obs)
                    qs[step] = q
                    values[step] = value.flatten()

                actions[step] = action

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                            logging_callback(info['episode']['r'])
            # bootstrap value if not done
            with torch.no_grad():
                q, next_value = agent.get_value(next_obs)
                next_value = next_value.reshape(1, -1)
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextvalues * nextnonterminal


            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
            b_qs = qs.reshape((-1, envs.single_action_space.n))

            b_inds = np.arange(args.batch_size)
            clipfracs = []
            weights_after_each_epoch = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                        end = start + args.minibatch_size
                        mb_inds = b_inds[start:end]

                        newqvalue, newvalue = agent.get_value(b_obs[mb_inds])
                        new_q_a_value = newqvalue.gather(1, b_actions.long()[mb_inds].unsqueeze(-1)).squeeze(-1)
                        
                        if args.saddle: loss = S(new_q_a_value, b_returns[mb_inds], sampler, newvalue, args.eta, args.gamma)
                        elif args.gumbel: loss = log_gumbel(new_q_a_value, b_returns[mb_inds], args.eta, newvalue, args.gamma)
                        else: loss = empirical_logistic_bellman(new_q_a_value, b_returns[mb_inds], args.eta, newvalue, args.gamma)
                        
                        critic_optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm)
                        critic_optimizer.step()
                        if args.saddle: sampler.update(new_q_a_value.detach(), b_returns[mb_inds])

                weights_after_each_epoch.append(deepcopy(agent.critic.state_dict()))
            
            if args.average_critics:
                avg_weights = {}
                for key in weights_after_each_epoch[0].keys():
                    avg_weights[key] = sum(T[key] for T in weights_after_each_epoch) / len(weights_after_each_epoch)
                agent.critic.load_state_dict(avg_weights)

            if args.parametrized:
                for epoch in range(args.update_policy_epochs):
                    np.random.shuffle(b_inds)
                    for start in range(0, args.batch_size, args.minibatch_size):
                            end = start + args.minibatch_size
                            mb_inds = b_inds[start:end]
                            with torch.no_grad():
                                newqvalue, newvalue = agent.get_value(b_obs[mb_inds])
                                new_q_a_value = newqvalue.gather(1, b_actions.long()[mb_inds].unsqueeze(1)).view(-1)
                                weights = torch.clamp(new_q_a_value / alpha, -50, 50)

                            _, newlogprob, newlogprobs, action_probs = agent.get_action(b_obs[mb_inds])

                            if args.nll_loss: actor_loss = torch.mean(torch.exp(weights) * newlogprob)
                            else: actor_loss = (action_probs * (alpha * newlogprobs - newqvalue)).mean()
                    
                            actor_optimizer.zero_grad()
                            actor_loss.backward()
                            actor_optimizer.step()

            if args.saddle: sampler.reset()
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", critic_optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("losses/critic_loss", loss, global_step)
            writer.add_scalar("losses/actor_loss", actor_loss, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
    except:
        logging_callback(0.0)

    envs.close()
    writer.close()

search_alg = HEBOSearch(metric="reward", mode="max")
re_search_alg = Repeater(search_alg, repeat=2)

analysis = tune.run(
    main,
    num_samples=100,
    config=config,
    search_alg=re_search_alg,
    local_dir="/Users/nicolasvila/workplace/uni/tfg_v2/tests/results_tune",
)

print("Best config: ", analysis.get_best_config(metric="reward", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.results_df

df.to_csv("qreps_analysis_v2.csv")