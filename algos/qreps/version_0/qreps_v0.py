# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
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
from losses import empirical_logistic_bellman, S, log_gumbel, empirical_logistic_bellman_trick
from agent import Agent

from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sampler import ExponentiatedGradientSampler, BestResponseSampler
## Try adding the replay buffer // checking if the data storage of the transition data is done right

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 0
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "QREPS_cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 100
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-3
    """the learning rate of the optimizer"""
    policy_lr: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 500
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 50
    """the K epochs to update the policy"""
    update_policy_epochs: int = 50
    """the K epochs to update the policy"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    alpha: float = 2
    """the entropy regularization coefficient"""
    eta: float = None
    """the KL regularization coefficient"""

    
    parametrized: bool = True
    """if toggled, the policy will be parametrized"""
    saddle: bool = False
    """if toggled, will use saddle point optimization"""
    nll_loss: bool = False
    """if toggled, will use nll for policy optimization"""
    sampler = ExponentiatedGradientSampler
    """the sampler to use for the optimization -either ExponentiatedGradientSampler or BestResponseSampler-"""
    average_critics: bool = True
    """if toggled, will average the critics after each iteration"""
    exp_clip: float = 8.0
    """the clipping value for the exps"""
    mse_loss: bool = False
    """if toggled, will use mse loss for the critic"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

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

def empirical_bellman_error(observations, next_observations, actions, rewards, qf, policy, gamma):
    v_target = qf.get_values(next_observations, actions, policy)[1]
    q_features = qf.get_values(observations, actions, policy)[0]
    output = rewards + gamma * v_target - q_features
    return output

def ELBE(eta, observations, next_observations, actions, rewards, qf, policy, gamma, sampler=None):
    errors = eta * torch.logsumexp(
        empirical_bellman_error(observations, next_observations, actions, rewards, qf, policy, gamma) / eta, 0
    ) + torch.mean((1 - gamma) * qf.get_values(observations, actions, policy)[1], 0)
    return errors


def optimize_critic(eta, observations, next_observations, actions, rewards, q_net, policy, gamma, sampler, optimizer, steps=300, loss_fn=ELBE, max_grad_norm=None):
    def closure():
        optimizer.zero_grad()
        loss = loss_fn(eta, observations, next_observations, actions, rewards, q_net , policy, gamma, sampler)
        loss.backward()
        if sampler is not None: sampler.update(empirical_bellman_error(observations, next_observations, actions, rewards, q_net, policy, gamma))
        if max_grad_norm is not None: nn.utils.clip_grad_norm_([param for group in optimizer.param_groups for param in group['params']], max_grad_norm)
        return loss

    for i in range(steps):
        optimizer.step(closure)

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps # // args.batch_size
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
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs, args).to(device)
    
    critic_optimizer = optim.Adam(agent.critic.parameters(), lr=args.learning_rate, eps=1e-5)
    actor_optimizer = optim.Adam(agent.actor.parameters(), lr=args.policy_lr, eps=1e-5)
    
    alpha = torch.Tensor([args.alpha]).to(device)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs, envs.single_action_space.n)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    qs = torch.zeros((args.num_steps, args.num_envs, envs.single_action_space.n)).to(device)
    if args.eta is None: eta = args.alpha
    else: eta = torch.Tensor([args.eta]).to(device)
    sampler = args.sampler(args.minibatch_size, device, eta=eta)
    
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    actor_loss=0
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
                action, a_logprob, logprob, action_probs = agent.get_action(next_obs)
                q, value = agent.get_value(next_obs)
                qs[step] = q
                values[step] = value.flatten()
                all_logprobs = logprob

            actions[step] = action
            logprobs[step] = all_logprobs

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        
        # bootstrap value if not done
        # with torch.no_grad():
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
        b_logprobs = logprobs.reshape((-1, envs.single_action_space.n))
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
                    
                    next_observations = b_obs[b_inds[start + 1:end +1]]

                    newqvalue, newvalue = agent.get_value(b_obs[mb_inds])
                    new_q_a_value = newqvalue.gather(1, b_actions.long()[mb_inds].unsqueeze(-1)).squeeze(-1)

                    loss = empirical_logistic_bellman(new_q_a_value, b_returns[mb_inds], eta, newvalue, args.gamma, args.exp_clip)
                    critic_optimizer.zero_grad(retain_graph=True)
                    loss.backward(retain_graph=True)
                    critic_optimizer.step()

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

                        _, a_newlogprob, newlogprob, action_probs = agent.get_action(b_obs[mb_inds])
                        newlogprobs = torch.log(action_probs)

                        actor_loss = (action_probs * (alpha * (newlogprob - b_logprobs[mb_inds]) - b_qs[mb_inds])).sum()
                
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

    envs.close()
    writer.close()