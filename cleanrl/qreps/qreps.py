# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
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

from ray.tune.search import Repeater
from ray.tune.search.hebo import HEBOSearch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import ray.tune as tune  # Import the missing package

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
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 0.08
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 300
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    policy_freq: int = 1
    """the frequency of updating the policy"""
    alpha: float = 0.01 #0.02 was current best
    """the entropy regularization coefficient"""
    clip_gradient_val: float = float("Inf")


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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, args):
        super(Agent, self).__init__()
        self.alpha =args.alpha
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )
        self.sampler = self.sampler_0 = Categorical(logits=torch.Tensor([1]*args.minibatch_size))
        self.beta_hat = 0.1

    def update_sampler(self, td):
        with torch.no_grad():
            print(self.sampler.probs)
            importance = len(self.sampler.probs)
            error = td - torch.log(self.sampler.probs / self.sampler_0.probs)/self.alpha
            error = importance * error
            logits = (self.beta_hat * error)
            logits = logits
            self.sampler = Categorical(logits=logits)

    def get_value(self, x):
        # _, _, log_probs, _ = self.get_action(x)
        q = self.critic(x)
        v = torch.logsumexp(q * self.alpha , dim=-1) / self.alpha
        return q, v

    def get_action(self, x, action=None):
        logits, v = self.get_value(x)
        # logits = self.actor(x)
        policy_dist = Categorical(logits=logits*self.alpha)
        log_probs = F.log_softmax(logits*self.alpha, dim=1)
        action_probs = policy_dist.probs
        if action is None:
            action = policy_dist.sample()
        return action, policy_dist.log_prob(action), log_probs, action_probs
    
def empirical_logistic_bellman(eta, td, values, discount):
    errors = 1 / eta * torch.log(torch.exp(eta * td).mean()) + torch.mean((1 - discount) * values, 0)
    return errors

def S(sampler, td, values, args):
    dual = sampler * (td - torch.log(len(sampler)*sampler)/args.alpha)
    errors = dual.sum() + (1-args.gamma) * values.mean()
    return errors

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
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
    actor_optimizer = optim.Adam(agent.actor.parameters(), lr=args.learning_rate, eps=1e-5)
    alpha = args.alpha

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
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
    update_policy = False
    for iteration in range(1, args.num_iterations + 1):

        if args.num_iterations % args.policy_freq == 0:
            update_policy = True
        
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            actor_optimizer.param_groups[0]["lr"] = lrnow
            critic_optimizer.param_groups[0]["lr"] = lrnow
            # policy_optimizer.param_groups[0]["lr"] = lrnow

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
            logprobs[step] = action_logprob

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
        with torch.no_grad():
            q, next_value = agent.get_value(next_obs)
            next_value = next_value.reshape(1, -1)
            returns = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
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
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    newqvalue, newvalue = agent.get_value(b_obs[mb_inds])
                    new_q_a_value = newqvalue.gather(1, b_actions.long()[mb_inds].unsqueeze(1)).view(-1)
                    td = b_returns[mb_inds] - new_q_a_value
                    # critic_loss = empirical_logistic_bellman(eta=args.alpha, td=td, values=newvalue, discount=args.gamma)
                    # critic_loss = critic_loss.mean()
                    # critic_loss = delta ** 2

                    loss = S(agent.sampler.probs, td=td, values=newvalue, args=args)
                    # agent.update_sampler(td=td) ---> Nans

                    critic_optimizer.zero_grad()
                    loss.backward()
                    critic_optimizer.step()
                    

            # if update_policy:
            #     np.random.shuffle(b_inds)
            #     for start in range(0, args.batch_size, args.minibatch_size):
            #         with torch.no_grad():  
            #             newqvalue, newvalue = agent.get_value(b_obs[mb_inds])
            #             new_q_a_value = newqvalue.gather(1, b_actions.long()[mb_inds].unsqueeze(1)).view(-1)
            #             weights = torch.clamp(args.alpha * (new_q_a_value-newvalue), -20, 20)

            #         _, newlogprob, newlogprobs, action_probs = agent.get_action(b_obs[mb_inds], b_actions.long()[mb_inds])  
            #         actor_loss = -torch.mean(torch.exp(weights)*newlogprob)
                    
            #         actor_optimizer.zero_grad()
            #         actor_loss.backward()
            #         actor_optimizer.step()
    
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", critic_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/critic_loss", loss, global_step)
        #writer.add_scalar("losses/actor_loss", actor_loss, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()