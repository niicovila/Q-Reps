# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "SAC_Discrete"
    """the wandb's project name"""
    wandb_entity: str = 'TFG'
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500
    """total timesteps of the experiments"""
    num_envs: int = 2
    """the number of parallel game environments"""
    buffer_size: int = int(1e5)
    """the replay memory buffer size"""  # smaller than in original paper but evaluation is done only for 100k steps anyway
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """target smoothing coefficient (default: 1)"""
    batch_size: int = 64
    """the batch size of sample from the reply memory"""
    learning_starts: int = 2e4
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-3
    """the learning rate of the Q network network optimizer"""
    update_frequency: int = 1
    """the frequency of training updates"""
    target_network_frequency: int = 4
    """the frequency of updates for the target networks"""
    alpha: float = 4.9
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    eta: float = 4.9
    """coefficient for the logistic loss"""
    target_entropy_scale: float = 0.1
    """coefficient for scaling the autotune entropy target"""

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

def empirical_bellman_error(observations, next_observations, actions, rewards, q_net_target, gamma):
    q_1_target, q_2_target = q_net_target
    v_target = torch.min(q_1_target.get_values(next_observations)[1], q_2_target.get_values(next_observations)[1])
    q_features = torch.min(q_1_target.get_values(observations)[0], q_2_target.get_values(observations)[0])
    q_features = q_features.gather(1, actions.long()).squeeze(-1)
    output = rewards.flatten() + gamma * v_target - q_features
    print("output:", output.mean()  )
    return output

def ELBE(eta, observations, next_observations, actions, rewards, q_nets, gamma):
    q_1, q_2, q_1_target, q_2_target = q_nets
    q_net_target = (q_1_target, q_2_target)

    errors = eta * torch.log(
    torch.mean(torch.exp(empirical_bellman_error(observations, next_observations, actions, rewards, q_net_target, gamma) / eta)),
    ) + torch.mean((1 - gamma) * torch.min(q_1.get_values(observations)[1], q_2.get_values(observations)[1]), 0)
    return errors

def optimize_critic(eta, observations, next_observations, actions, rewards, q_nets , gamma, optimizer, steps=300, loss_fn=ELBE):
    def closure():
        optimizer.zero_grad()
        loss = loss_fn(eta, observations, next_observations, actions, rewards, q_nets , gamma)
        loss.backward()
        return loss

    for i in range(steps):
        optimizer.step(closure)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, args, policy):
        super().__init__()
        self.alpha = args.alpha
        self.policy = policy
        self.critic = nn.Sequential(
            (nn.Linear(np.array(env.single_observation_space.shape).prod(), 128)),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            (nn.Linear(128, env.single_action_space.n)),
        )

    def forward(self, x):
        return self.critic(x)
    
    def get_values(self, x):
        q = self(x)
        z = q / self.alpha
        _, _, _, pi_k = self.policy.get_action(x)
        max_z = torch.max(z, dim=-1, keepdim=True)[0]
        max_z = torch.where(max_z < -1.0, torch.tensor(-1.0, dtype=torch.float, device=max_z.device), max_z)
        v = self.alpha * (torch.log(torch.sum(pi_k.detach()*torch.exp(z-max_z), dim=1)))
        return q, v
    
class QREPSPolicy(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.actor = nn.Sequential(
            (nn.Linear(np.array(envs.single_observation_space.shape).prod(), 128)),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            (nn.Linear(128, envs.single_action_space.n)),
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
    

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    import stable_baselines3 as sb3

    args = tyro.cli(Args)
    # assert args.num_envs == 1, "vectorized envs are not supported at the moment"
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
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    actor = QREPSPolicy(envs).to(device)

    qf1 = QNetwork(envs, args, actor).to(device)
    qf2 = QNetwork(envs, args, actor).to(device)
    qf1_target = QNetwork(envs, args, actor).to(device)
    qf2_target = QNetwork(envs, args, actor).to(device)

    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, eps=1e-4)

    if args.autotune:
        target_entropy = -args.target_entropy_scale * torch.log(1 / torch.tensor(envs.single_action_space.n))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, eps=1e-4)
    else:
        alpha = args.alpha
        eta = args.eta

    rb = ReplayBuffer(args.buffer_size)
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    rollouts = 5
    iterations = 100
    global_step = 0
    for T in range(iterations):
        all_rewards = []
        ##Collect sample transitions
        for N in range(rollouts):
            for step in range(args.total_timesteps):
                global_step += 1
                actions, _, _, _ = actor.get_action(torch.Tensor(obs).to(device))
                action = actions.detach().cpu().numpy()

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, termination, truncation, info = envs.step(action)
                
                rb.push(obs, next_obs, action, reward, termination)

                obs = next_obs
                all_rewards.append(reward)

                if termination.any() == True:
                    break

        # TRAINING PHASE         
        if global_step % args.update_frequency == 0:
            (
            observations, 
            next_observations, 
            actions, 
            rewards, 
            dones
            ) = rb.get_all()

            q_nets = (qf1, qf2, qf1_target, qf2_target)
            optimize_critic(args.eta, observations, next_observations, actions, rewards, q_nets , args.gamma, q_optimizer)
            qreps_loss = ELBE(args.eta, observations, next_observations, actions, rewards, q_nets , args.gamma)
            print("qreps_loss:", qreps_loss.mean().item())
            
            # ACTOR training
            _, action_log_pi, log_pi, action_probs = actor.get_action(observations)
            with torch.no_grad():
                qf1_values = qf1(observations)
                qf2_values = qf2(observations)
                min_qf_values = torch.min(qf1_values, qf2_values)

            # no need for reparameterization, the expectation can be calculated for discrete actions
                
            actor_loss = (action_probs * ((alpha * log_pi) - min_qf_values)).mean()
            # actor_optimizer.zero_grad()
            # actor_loss.backward()
            # actor_optimizer.step()

            if args.autotune:
                # re-use action probabilities for temperature loss
                alpha_loss = (action_probs.detach() * (-log_alpha.exp() * (log_pi + target_entropy).detach())).mean()

                a_optimizer.zero_grad()
                alpha_loss.backward()
                a_optimizer.step()
                alpha = log_alpha.exp().item()

        # update target network
        if global_step % args.target_network_frequency == 0:
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        rb.reset()
        print("iteation:", T, "reward:", np.sum(all_rewards)/(rollouts))

    envs.close()
    writer.close()