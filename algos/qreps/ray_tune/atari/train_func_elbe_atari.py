# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse
from copy import deepcopy
import random
import time
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from ray import train
from utils_atari import QREPSAgent
from common_utils import nll_loss, kl_loss
import logging

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
SEED_OFFSET = 1

def make_env(env_id, idx, capture_video, run_name):
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
        return env

    return thunk


def tune_elbe_atari(config):
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
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = QREPSAgent(envs, args).to(device)

    if args.optimizer == "Adam" or args.optimizer == "RMSprop":
        optimizer = getattr(optim, args.optimizer)(
            list(agent.parameters()), lr=args.learning_rate, eps=args.eps
        )
    else:
        optimizer = getattr(optim, args.optimizer)(
            list(agent.parameters()), lr=args.learning_rate
        )

    if args.target_network:
        agent_target = QREPSAgent(envs, args).to(device)
        agent_target.load_state_dict(agent.state_dict())

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
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs

            # ALGO LOGIC: action logic
            with torch.no_grad():        
                action, _, logprob, _ = agent.get_action(next_obs)

            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            reward = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            
            next_observations[step] = next_obs
            dones[step] = next_done

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        reward_iteration.append(info["episode"]["r"])

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_next_obs = next_observations.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape((-1, envs.single_action_space.n))
        b_rewards = rewards.flatten()
        b_dones = dones.flatten()
        b_inds = np.arange(args.total_iterations)
        weights_after_each_epoch = []

        if len(reward_iteration)>5: 
            writer.add_scalar("charts/episodic_return", np.mean(np.mean(reward_iteration)), global_step)
            logging_callback(np.mean(reward_iteration))
            reward_iteration = []

        for epoch in range(args.update_epochs):
            # np.random.shuffle(b_inds)
            for start in range(0, args.total_iterations, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    
                    q, values = agent.get_values(b_obs[mb_inds], b_actions[mb_inds])
                    values_next = agent.get_values(b_next_obs[mb_inds], b_actions[mb_inds])[1]

                    if args.target_network:
                        delta = b_rewards[mb_inds].squeeze() + args.gamma * agent_target.get_values(b_next_obs[mb_inds])[1].detach() * (1 - b_dones[mb_inds].squeeze()) - q        
                        
                    elif args.gae:
                        delta = torch.zeros_like(b_rewards[mb_inds]).to(device)
                        lastgaelam = 0
                        for t in reversed(range(args.minibatch_size)):
                            nextnonterminal = 1.0 - b_dones[mb_inds][t]
                            nextvalues = values_next[t]
                            delta_t = b_rewards[mb_inds][t] + args.gamma * nextvalues * nextnonterminal - q[t]
                            delta[t] = lastgaelam = delta_t + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                        returns = delta + q
                    
                    else: delta = b_rewards[mb_inds].squeeze() + args.gamma * values_next * (1 - b_dones[mb_inds].squeeze()) - q

                    if args.normalize_delta: delta = (delta - delta.mean()) / (delta.std() + 1e-8)

                    critic_loss = eta * torch.log(torch.mean(torch.exp(delta / eta), 0)) + torch.mean((1 - args.gamma) * values, 0)

                    optimizer.zero_grad()
                    critic_loss.backward()
                    optimizer.step()

                    if args.use_kl_loss: actor_loss = kl_loss(alpha, b_obs[mb_inds], b_next_obs[mb_inds], b_rewards[mb_inds], b_actions[mb_inds], b_logprobs[mb_inds], agent, agent)
                    else: actor_loss = nll_loss(alpha, b_obs[mb_inds], b_next_obs[mb_inds], b_rewards[mb_inds], b_actions[mb_inds], b_logprobs[mb_inds], agent, agent)
                    
                    optimizer.zero_grad()
                    actor_loss.backward()
                    optimizer.step()

            if args.average_critics: weights_after_each_epoch.append(deepcopy(agent.critic.state_dict()))
        
        if args.average_critics:
            avg_weights = {}
            for key in weights_after_each_epoch[0].keys():
                avg_weights[key] = sum(T[key] for T in weights_after_each_epoch) / len(weights_after_each_epoch)
            agent.critic.load_state_dict(avg_weights)

        if args.target_network and iteration % args.target_network_frequency == 0:
            for param, target_param in zip(agent.critic.parameters(), agent_target.critic.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
    except:
        logging_callback(-1)

    if len(reward_iteration)>0:
        logging_callback(np.mean(reward_iteration))

    envs.close()
    writer.close()


def tune_elbe_atari_decoupled(config):
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
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = QREPSAgent(envs, args).to(device)

    if args.optimizer == "Adam" or args.optimizer == "RMSprop":
        optimizer = getattr(optim, args.optimizer)(
            list(agent.parameters()), lr=args.learning_rate, eps=args.eps
        )
    else:
        optimizer = getattr(optim, args.optimizer)(
            list(agent.parameters()), lr=args.learning_rate
        )

    if args.target_network:
        agent_target = QREPSAgent(envs, args).to(device)
        agent_target.load_state_dict(agent.state_dict())

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
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs

            # ALGO LOGIC: action logic
            with torch.no_grad():        
                action, _, logprob, _ = agent.get_action(next_obs)

            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            reward = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            
            next_observations[step] = next_obs
            dones[step] = next_done
            # reward_iteration.append(reward)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        reward_iteration.append(info["episode"]["r"])

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_next_obs = next_observations.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape((-1, envs.single_action_space.n))
        b_rewards = rewards.flatten()
        b_dones = dones.flatten()
        b_inds = np.arange(args.total_iterations)
        weights_after_each_epoch = []

        if len(reward_iteration)>5: 
            writer.add_scalar("charts/episodic_return", np.mean(np.mean(reward_iteration)), global_step)
            logging_callback(np.mean(reward_iteration))
            print(np.mean(reward_iteration))
            reward_iteration = []

        for epoch in range(args.update_epochs):
            # np.random.shuffle(b_inds)
            for start in range(0, args.total_iterations, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    
                    q, values = agent.get_values(b_obs[mb_inds], b_actions[mb_inds])
                    values_next = agent.get_values(b_next_obs[mb_inds], b_actions[mb_inds])[1]

                    if args.target_network:
                        delta = b_rewards[mb_inds].squeeze() + args.gamma * agent_target.get_values(b_next_obs[mb_inds])[1].detach() * (1 - b_dones[mb_inds].squeeze()) - q        
                        
                    elif args.gae:
                        delta = torch.zeros_like(b_rewards[mb_inds]).to(device)
                        lastgaelam = 0
                        for t in reversed(range(args.minibatch_size)):
                            nextnonterminal = 1.0 - b_dones[mb_inds][t]
                            nextvalues = values_next[t]
                            delta_t = b_rewards[mb_inds][t] + args.gamma * nextvalues * nextnonterminal - q[t]
                            delta[t] = lastgaelam = delta_t + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                        returns = delta + q
                    
                    else: delta = b_rewards[mb_inds].squeeze() + args.gamma * values_next * (1 - b_dones[mb_inds].squeeze()) - q

                    if args.normalize_delta: delta = (delta - delta.mean()) / (delta.std() + 1e-8)

                    critic_loss = eta * torch.log(torch.mean(torch.exp(delta / eta), 0)) + torch.mean((1 - args.gamma) * values, 0)

                    optimizer.zero_grad()
                    critic_loss.backward()
                    optimizer.step()

            if args.average_critics: weights_after_each_epoch.append(deepcopy(agent.critic.state_dict()))
        
        if args.average_critics:
            avg_weights = {}
            for key in weights_after_each_epoch[0].keys():
                avg_weights[key] = sum(T[key] for T in weights_after_each_epoch) / len(weights_after_each_epoch)
            agent.critic.load_state_dict(avg_weights)

        for epoch in range(args.update_epochs_policy):
            np.random.shuffle(b_inds)
            for start in range(0, args.total_iterations, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    if args.use_kl_loss: actor_loss = kl_loss(alpha, b_obs[mb_inds], b_next_obs[mb_inds], b_rewards[mb_inds], b_actions[mb_inds], b_logprobs[mb_inds], agent, agent)
                    else: actor_loss = nll_loss(alpha, b_obs[mb_inds], b_next_obs[mb_inds], b_rewards[mb_inds], b_actions[mb_inds], b_logprobs[mb_inds], agent, agent)
                    
                    optimizer.zero_grad()
                    actor_loss.backward()
                    optimizer.step()


        if args.target_network and iteration % args.target_network_frequency == 0:
            for param, target_param in zip(agent.critic.parameters(), agent_target.critic.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
    except:
        logging_callback(-1)

    if len(reward_iteration)>0:
        logging_callback(np.mean(reward_iteration))

    envs.close()
    writer.close()


config_atari = {
    "exp_name": "QREPS",
    "seed": 3,
    "torch_deterministic": True,
    "cuda": True,
    "track": False,
    "wandb_project_name": "CC",
    "wandb_entity": None,
    "capture_video": False,

    "env_id":  "ALE/Breakout-v5",

    # Algorithm
    "total_timesteps": 100000,
    "num_envs": 256,
    "gamma": 0.99,

    "total_iterations": 4096,
    "num_minibatches": 64,
    "update_epochs": 10,
    "update_epochs_policy": 10,

    "alpha": 32,  
    "eta": 32,

    # Learning rates
    "beta": 3e-4,
    "learning_rate": 3e-4,
    "anneal_lr": False,

    # Layer Init
    "layer_init": "orthogonal_gain",

    # Architecture
    "policy_activation": "Tanh",
    "num_hidden_layers": 2,
    "hidden_size": 64,
    "actor_last_layer_std": 0.01,

    "q_activation": "Tanh",
    "q_num_hidden_layers": 4,
    "q_hidden_size": 128,
    "q_last_layer_std": 1.0,

    "average_critics": True,
    "use_policy": True,


    # Optimization
    "optimizer": "Adam",
    "eps": 1e-8,

    # Options
    "normalize_delta": False,
    "gae": False,
    "gae_lambda": 0.95,
    "use_kl_loss": True,
    "q_histogram": False,

    "target_network": False,
    "tau": 1.0,
    "target_network_frequency": 0, 

    "minibatch_size": 0,
    "num_iterations": 0,
    "num_steps": 0,
}


