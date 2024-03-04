# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
from copy import deepcopy
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
from losses import empirical_logistic_bellman, optimize_loss, S, nll_loss
from sampler import Sampler

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
    wandb_project_name: str = "QREPS_cleanRL_RB"
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
    learning_rate: float = 2.5e-4
    """the learning rate of the critic"""
    policy_lr: float = 1e-3
    """the learning rate of the actor"""
    num_envs: int = 5
    """the number of parallel game environments"""
    num_steps: int = 500
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 1.0
    """the discount factor gamma"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 300
    """the K epochs to update the policy"""
    update_policy_epochs: int = 300
    """the K epochs to update the policy"""
    alpha: float = 0.0 #0.02 was current best
    """the entropy regularization coefficient"""
    eta: float = 0
    """the entropy regularization coefficient"""
    parametrized: bool = True
    """if toggled, the policy will be parametrized"""
    saddle: bool = False
    """if toggled, will use saddle point optimization"""

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

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps 
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
    if args.eta == 0: args.eta = args.alpha
    eta = torch.Tensor([args.eta]).to(device)
    if args.saddle: sampler = Sampler(args.minibatch_size, device, eta=eta, beta=0.01)
    buffer = ReplayBuffer(args.num_steps * args.num_envs)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    actor_loss = 0
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

            update(buffer, obs, next_obs, action, reward, next_done)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # Optimize critic
        (   _observations,
            _actions,
            _rewards,
            _next_observations,
            _next_done,
        ) = buffer.get_all()

        with torch.no_grad():
            b_returns = _rewards + args.gamma * agent.get_value(_next_observations)[1] * (1 - _next_done)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        weights_after_each_epoch = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    newqvalue, newvalue = agent.get_value(_observations[mb_inds])
                    new_q_a_value = newqvalue.gather(1, _actions.long()[mb_inds].unsqueeze(-1)).squeeze(-1)
                    
                    if args.saddle:
                        loss = S(new_q_a_value, b_returns[mb_inds], sampler, newvalue, args.eta, args.gamma)
                        sampler.update(new_q_a_value.detach(), b_returns[mb_inds])
                    else: loss = empirical_logistic_bellman(new_q_a_value, b_returns[mb_inds], args.eta, newvalue, args.gamma)
                    
                    critic_optimizer.zero_grad()
                    loss.backward()
                    # print(loss.item())
                    critic_optimizer.step()

            weights_after_each_epoch.append(deepcopy(agent.critic.state_dict()))

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
                            newqvalue, newvalue = agent.get_value(_observations[mb_inds])
                            new_q_a_value = newqvalue.gather(1, _actions.long()[mb_inds].unsqueeze(1)).view(-1)
                            weights = args.alpha * (new_q_a_value)

                        _, newlogprob, newlogprobs, action_probs = agent.get_action(_observations[mb_inds])  
                        actor_loss = -torch.mean(torch.exp(weights)*newlogprob)

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()

        buffer.reset()

        writer.add_scalar("losses/actor_loss", actor_loss, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()