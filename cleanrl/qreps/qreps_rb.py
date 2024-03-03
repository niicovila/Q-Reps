# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
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
    learning_rate: float = 2.5e-3
    """the learning rate of the critic"""
    policy_lr: float = 1e-3
    """the learning rate of the actor"""
    num_envs: int = 5
    """the number of parallel game environments"""
    num_steps: int = 500
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    num_minibatches: int = 1
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    alpha: float = 0.02 #0.02 was current best
    """the entropy regularization coefficient"""
    eta: float = 0
    """the entropy regularization coefficient"""
    parametrized: bool = True
    """if toggled, the policy will be parametrized"""
    saddle: bool = True
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

            update(buffer, obs, next_obs, action, reward, next_done) # Try also with dones instead of next_done

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # Optimize critic
        if args.saddle:
            optimize_loss(buffer=buffer, loss_fn=S, optimizer=critic_optimizer, agent=agent, args=args, optimizer_steps=args.update_epochs, sampler=sampler)
        else: 
            optimize_loss(buffer=buffer, loss_fn=empirical_logistic_bellman, optimizer=critic_optimizer, agent=agent, args=args, optimizer_steps=args.update_epochs)
        
        # Optimize actor
        if args.parametrized: optimize_loss(buffer=buffer, loss_fn=nll_loss, optimizer=critic_optimizer, agent=agent, args=args, optimizer_steps=args.update_epochs)
        
        buffer.reset()

        writer.add_scalar("losses/actor_loss", actor_loss, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()