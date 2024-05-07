# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse 
from copy import deepcopy
import random
import time
import gymnasium as gym
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from ray import train
from utils import make_env, QNetwork, QREPSPolicy
from common_utils import Sampler, ExponentiatedGradientSampler, nll_loss, kl_loss
import logging
FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
SEED_OFFSET = 1

def tune_saddle(config):
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
    mean_rewards = []
    if args.save_learning_curve: rewards_df = pd.DataFrame(columns=["Step", "Reward"])

    #try:
    for iteration in range(1, args.num_iterations + 1):
        reward_iteration = []
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
            dones[step] = next_done
            if "final_info" in infos:
                rs = []
                ls = []
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        ls.append(info["episode"]["l"])
                        reward_iteration.append(info["episode"]["r"])
                        mean_rewards.append(info["episode"]["r"])
                        rs.append(info["episode"]["r"])

                if len(rs)>0:
                    writer.add_scalar("charts/episodic_length", np.mean(ls), global_step)
                    writer.add_scalar("charts/episodic_return", np.mean(rs), global_step)

        if len(mean_rewards) > 5: 
            logging_callback(np.mean(mean_rewards))
            mean_rewards = []

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_next_obs = next_observations.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape((-1, envs.single_action_space.n))
        b_rewards = rewards.flatten()
        b_dones = dones.flatten()
        b_inds = np.arange(args.total_iterations)

        weights_after_each_epoch = []

        # np.random.shuffle(b_inds)
        for start in range(0, args.total_iterations, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

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
                sampler = ExponentiatedGradientSampler(args.minibatch_size, device, eta, args.beta)
            
            for epoch in range(args.update_epochs):            
                q, values = qf.get_values(b_obs[mb_inds], b_actions[mb_inds], actor)
                values_next = qf.get_values(b_next_obs[mb_inds], b_actions[mb_inds], actor)[1]

                if args.target_network:
                    delta = b_rewards[mb_inds].squeeze() + args.gamma * qf_target.get_values(b_next_obs[mb_inds], policy=actor)[1].detach() * (1 - b_dones[mb_inds].squeeze()) - q        
                    
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

                if args.normalize_delta: delta = (delta - delta.mean()) / (delta.std() + 1e-9)

                bellman = delta.detach()
                if args.parametrized_sampler: z_n = sampler.get_probs(bellman) 
                else: z_n = sampler.probs() 
                
                critic_loss = torch.sum(z_n.detach() * (delta - eta * torch.log(sampler.n * z_n.detach()))) + (1 - args.gamma) * values.mean()
            
                q_optimizer.zero_grad()
                critic_loss.backward()
                q_optimizer.step()


                if args.parametrized_sampler:
                    sampler_loss = - (torch.sum(z_n * (bellman - eta * torch.log(sampler.n * z_n))) + (1 - args.gamma) * values.mean().detach())
                    
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

        if iteration % 10000  == 0:
            print(f'Iteration: {iteration}, Reward: {np.mean(reward_iteration)}')
            if args.save_learning_curve: rewards_df = rewards_df._append({"Step": global_step, "Reward": np.mean(reward_iteration)}, ignore_index=True)
    
    # except:
    #     logging_callback(-2000)

    if len(mean_rewards) > 0:
        logging_callback(np.mean(reward_iteration))

    envs.close()
    writer.close()
    if args.save_learning_curve: 
        rewards_df.to_csv(f"rewards_{run_name}.csv")   
        return rewards_df


def tune_saddle_decoupled(config):
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
    if args.save_learning_curve: rewards_df = pd.DataFrame(columns=["Step", "Reward"])

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
            dones[step] = next_done

            if "final_info" in infos:
                rs = []
                ls = []
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        ls.append(info["episode"]["l"])
                        reward_iteration.append(info["episode"]["r"])
                        rs.append(info["episode"]["r"])

                if len(rs)>0:
                    writer.add_scalar("charts/episodic_length", np.mean(ls), global_step)
                    writer.add_scalar("charts/episodic_return", np.mean(rs), global_step)
                    
                    if args.save_learning_curve: 
                        rewards_df = rewards_df._append({"Step": global_step, "Reward": np.mean(rs)}, ignore_index=True)
                        print(f'Iteration: {global_step}, Reward: {np.mean(rs)}')

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

        # np.random.shuffle(b_inds)
        
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
                sampler = ExponentiatedGradientSampler(args.minibatch_size, device, eta, args.beta)

            for epoch in range(args.update_epochs):      

                q, values = qf.get_values(b_obs[mb_inds], b_actions[mb_inds], actor)
                values_next = qf.get_values(b_next_obs[mb_inds], b_actions[mb_inds], actor)[1]
     
                if args.target_network:
                    delta = b_rewards[mb_inds].squeeze() + args.gamma * qf_target.get_values(b_next_obs[mb_inds], policy=actor)[1].detach() * (1 - b_dones[mb_inds].squeeze()) - qf.get_values(b_obs[mb_inds], b_actions[mb_inds], actor)[0]          
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

                if args.normalize_delta: delta = (delta - delta.mean()) / (delta.std() + 1e-9)

                bellman = delta.detach()
                if args.parametrized_sampler: z_n = sampler.get_probs(bellman) 
                else: z_n = sampler.probs() 
                critic_loss = torch.sum(z_n.detach() * (delta - eta * torch.log(sampler.n * z_n.detach()))) + (1 - args.gamma) * values.mean()

                q_optimizer.zero_grad()
                critic_loss.backward()
                q_optimizer.step()


                if args.parametrized_sampler:
                    sampler_loss = - (torch.sum(z_n * (bellman - eta * torch.log(sampler.n * z_n))) + (1 - args.gamma) * values.mean().detach())
                    
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

        for epoch in range(args.update_epochs_policy):
            np.random.shuffle(b_inds)
            for start in range(0, args.total_iterations, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    if args.use_kl_loss: actor_loss = kl_loss(alpha, b_obs[mb_inds], b_next_obs[mb_inds], b_rewards[mb_inds], b_actions[mb_inds], b_logprobs[mb_inds], qf, actor)
                    else: actor_loss = nll_loss(alpha, b_obs[mb_inds], b_next_obs[mb_inds], b_rewards[mb_inds], b_actions[mb_inds], b_logprobs[mb_inds], qf, actor)
                    
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

        if args.target_network and iteration % args.target_network_frequency == 0:
            for param, target_param in zip(qf.parameters(), qf_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
    except:
        logging_callback(-2000)

    if len(reward_iteration) > 0:
        logging_callback(np.mean(reward_iteration))
        if args.save_learning_curve: rewards_df = rewards_df._append({"Step": global_step, "Reward": np.mean(reward_iteration)}, ignore_index=True)

    envs.close()
    writer.close()
    if args.save_learning_curve: 
        rewards_df.to_csv(f"rewards_{run_name}.csv")   
        return rewards_df
    

def tune_saddle_v2(config):
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

    actor = QREPSPolicy(envs, args).to(device)
    qf = QNetwork(envs, args).to(device)

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
            actor_optimizer.param_groups[0]["lr"] = lrnow

            lrnow = frac * args.q_lr
            q_optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs


            # ALGO LOGIC: action logic
            with torch.no_grad():        
                action, _, logprob, _ = actor.get_action(next_obs)
                q, v = qf.get_values(next_obs, action, actor)
                qs[step] = q
                values[step] = v

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
                rs = []
                ls = []
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        ls.append(info["episode"]["l"])
                        reward_iteration.append(info["episode"]["r"])
                        rs.append(info["episode"]["r"])

                if len(rs)>0:
                    writer.add_scalar("charts/episodic_length", np.mean(ls), global_step)
                    writer.add_scalar("charts/episodic_return", np.mean(rs), global_step)
                    
                    if args.save_learning_curve: 
                        rewards_df = rewards_df._append({"Step": global_step, "Reward": np.mean(rs)}, ignore_index=True)
                        print(f'Iteration: {global_step}, Reward: {np.mean(rs)}')

        if len(reward_iteration) > 5: 
            logging_callback(np.mean(reward_iteration))
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
        b_logprobs = logprobs.reshape((-1, envs.single_action_space.n))
        b_rewards = rewards.flatten()
        b_dones = dones.flatten()
        b_inds = np.arange(args.total_iterations)

        weights_after_each_epoch = []
        np.random.shuffle(b_inds)

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
                sampler = ExponentiatedGradientSampler(args.minibatch_size, device, eta, args.beta)
            
            for epoch in range(args.update_epochs):       
                q_val, val = qf.get_values(b_obs[mb_inds], b_actions[mb_inds], actor)
                values_next = qf.get_values(b_next_obs[mb_inds], b_actions[mb_inds], actor)[1]

                if args.target_network:
                    delta =  b_returns[mb_inds] - q_val  

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

    # except:
    #     logging_callback(-2000)

    if len(reward_iteration) > 0:
        logging_callback(np.mean(reward_iteration))
        if args.save_learning_curve: rewards_df = rewards_df._append({"Step": global_step, "Reward": np.mean(reward_iteration)}, ignore_index=True)

    envs.close()
    writer.close()
    if args.save_learning_curve: 
        rewards_df.to_csv(f"rewards_{run_name}.csv")   
        return rewards_df