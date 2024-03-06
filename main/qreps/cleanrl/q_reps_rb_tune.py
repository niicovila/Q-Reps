# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import argparse
import os
import random
import time
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from replay_buffer import ReplayBuffer, update
from agent import Agent
from losses import empirical_logistic_bellman, optimize_loss, S, nll_loss, sac_loss, log_gumbel
from sampler import BestResponseSampler, ExponentiatedGradientSampler

from ray import train
from ray.tune.search import Repeater
from ray.tune.search.hebo import HEBOSearch
import ray.tune as tune  # Import the missing package



config = {
  "exp_name": "config",
  "seed": 0,
  "torch_deterministic": True,
  "cuda": True,
  "track": False,
  "wandb_project_name": "QREPS_cleanRL_RB",
  "wandb_entity": None,
  "capture_video": False,
  "env_id": "CartPole-v1",
  "total_timesteps": 30,
  "learning_rate": tune.loguniform(1e-4, 5e-2),
  "policy_lr": tune.loguniform(2e-3, 2e-1),
  "num_envs": 5,
  "num_steps": 200,
  "anneal_lr": False,
  "gamma": 0.99,
  "num_minibatches": 1,
  "update_epochs": tune.choice([300, 450]),
  "alpha": tune.loguniform(2e-1, 5),
  "eta": 0,
  "parametrized": True,
  "saddle": True,
  "sampler": BestResponseSampler,
  "nll_loss": True,
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
    
    args = argparse.Namespace(**config)
    
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps 
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    args.seed = config["__trial_index__"] + SEED_OFFSET
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
    critic_optimizer = optim.Adam(agent.critic.parameters(), lr=args.learning_rate, eps=1e-5)
    actor_optimizer = optim.Adam(agent.actor.parameters(), lr=args.policy_lr, eps=1e-5)
    alpha = torch.Tensor([args.alpha]).to(device)
    if args.eta == 0: args.eta = args.alpha
    eta = torch.Tensor([args.eta]).to(device)
    if args.saddle: sampler = args.sampler(args.minibatch_size, device, eta=eta)
    buffer = ReplayBuffer(args.num_steps * args.num_envs)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    actor_loss = 0
    try:
        for iteration in range(1, args.num_iterations + 1): # K

            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                actor_optimizer.param_groups[0]["lr"] = lrnow
                critic_optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps): # N
                global_step += args.num_envs
                obs = next_obs
                dones = next_done
                with torch.no_grad():        
                    action, _, _, _ = agent.get_action(next_obs)

                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                reward = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

                update(buffer, obs, next_obs, action, reward, next_done) # Try also with dones instead of next_done

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                            logging_callback(info['episode']['r'])

            # Optimize critic
            if args.saddle:
                optimize_loss(buffer=buffer, loss_fn=S, optimizer=critic_optimizer, agent=agent, args=args, optimizer_steps=args.update_epochs, sampler=sampler)
            else: 
                optimize_loss(buffer=buffer, loss_fn=empirical_logistic_bellman, optimizer=critic_optimizer, agent=agent, args=args, optimizer_steps=args.update_epochs)
            
            # Optimize actor
            if args.parametrized: 
                if args.nll_loss: optimize_loss(buffer=buffer, loss_fn=nll_loss, optimizer=critic_optimizer, agent=agent, args=args, optimizer_steps=args.update_epochs)
                else: optimize_loss(buffer=buffer, loss_fn=sac_loss, optimizer=critic_optimizer, agent=agent, args=args, optimizer_steps=args.update_epochs)
            buffer.reset()

            if args.saddle: sampler.reset()

            writer.add_scalar("losses/actor_loss", actor_loss, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
    except:
        logging_callback(0.0)

    envs.close()
    writer.close()

search_alg = HEBOSearch(metric="reward", mode="max")
re_search_alg = Repeater(search_alg, repeat=1)

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

df.to_csv("qreps_analysis_rb_saddle_param_nll.csv")