from .base import Algorithm
from research.networks.base import ActorCriticValuePolicy
from research.utils import utils
from functools import partial
import torch.distributions as D
import torch
import numpy as np
import itertools

from research.networks.base import ActorCriticPolicy
from research.utils.utils import to_tensor, to_device

class Sampler:
    def __init__(self, N):
        self.n = N
        self.prob_dist = D.Categorical( probs=torch.ones(N) / N)
        self.entropy = self.prob_dist.entropy()
    def probs(self):
        return self.prob_dist.probs
    def update(self, probs):
        pass



def qreps_loss(delta, eta, clip, V, batch_size, discount, action_space):
    t1 = sum(torch.exp(eta * delta))
    t1 /= (batch_size * action_space)
    loss = (1/eta) * torch.log(t1)
    loss += (1 - discount) * V.sum() / batch_size
    return loss

def mse_loss(pred, label):
    return (label - pred)**2

def S(sampler, label, pred, values, beta, discount):
    z = label - pred
    dual = sampler.probs() * z 
    errors = dual.sum() - (sampler.entropy + np.log((sampler.n)))*beta
    # errors +=  (1-discount) * values.mean()
    return errors

class QREPSDiscrete(Algorithm):

    def __init__(self, env, network_class, dataset_class, 
                       tau=0.005,
                       policy_noise=0.1,
                       init_temperature=0.1,
                       target_noise=0.2,
                       noise_clip=0.5,
                       env_freq=1,
                       critic_freq=1,
                       actor_freq=2,
                       target_freq=2,
                       init_steps=1000,
                       eta = 0.5,
                       beta=4.0,
                       exp_clip=10,
                       use_target_actor=True,
                       max_grad_value=None,
                       alpha=None,
                       **kwargs):
        

        self.init_temperature = init_temperature
        self._alpha = alpha
        super().__init__(env, network_class, dataset_class, **kwargs)
        assert isinstance(self.network, ActorCriticPolicy)


        self.beta = beta
        self.exp_clip = exp_clip
        self.use_target_actor = use_target_actor
        self.max_grad_value = max_grad_value
        self.eta = eta

        self.tau = tau
        self.policy_noise = policy_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.env_freq = env_freq
        self.critic_freq = critic_freq
        self.actor_freq = actor_freq
        self.target_freq = target_freq
        self.action_range = range(self.env.action_space.n)
        self.action_range_tensor = to_device(to_tensor(self.action_range), self.device)
        self.init_steps = init_steps
        # self.sampler = D.Categorical(logits=torch.ones(self.dataset.batch_size))

    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def setup_network(self, network_class, network_kwargs):
        self.network = network_class(self.env.observation_space, self.env.action_space, 
                                     **network_kwargs).to(self.device)
        self.target_network = network_class(self.env.observation_space, self.env.action_space, 
                                     **network_kwargs).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False

    def setup_optimizers(self, optim_class, optim_kwargs):
        # Default optimizer initialization
        self.optim['actor'] = optim_class(self.network.actor.parameters(), **optim_kwargs)
        # Update the encoder with the critic.
        critic_params = itertools.chain(self.network.critic.parameters(), self.network.encoder.parameters())        
        self.optim['critic'] = optim_class(critic_params, **optim_kwargs)

        self.target_entropy = - 1
        if self._alpha is None:
            # Setup the learned entropy coefficients. This has to be done first so its present in the setup_optim call.
            self.log_alpha = torch.tensor(np.log(self.init_temperature), dtype=torch.float).to(self.device)
            self.log_alpha.requires_grad = True
            self.optim['log_alpha'] = optim_class([self.log_alpha], **optim_kwargs)
        else:
            self.log_alpha = torch.tensor(np.log(self._alpha), dtype=torch.float).to(self.device)
            self.log_alpha.requires_grad = False
            

    def _update_critic(self, batch):
        with torch.no_grad():
            target_q = self.target_network.critic(batch['next_obs'])
            closed_solution_v = 1/self.eta * (torch.logsumexp( self.eta * target_q, dim=1))
            target_q = batch['reward'] + batch['discount'] * closed_solution_v

        q = self.network.critic(batch['obs'])
        target_q = torch.cat([target_q.unsqueeze(1)] * self.env.action_space.n, dim=1)
        delta = target_q - q

        delta = delta.reshape(delta.shape[0]*delta.shape[1])


        loss_fn = partial(qreps_loss, eta=self.eta, clip=self.exp_clip, V=closed_solution_v, batch_size=self.dataset.batch_size, discount=self.dataset.discount, action_space=self.env.action_space.n)
        q_loss = loss_fn(delta)

        self.optim['critic'].zero_grad(set_to_none=True)
        
        q_loss.backward()

        if self.max_grad_value is not None:
            torch.nn.utils.clip_grad_value_(self.network.critic.parameters(), self.max_grad_value)
        self.optim['critic'].step()

        return dict(q_loss=q_loss.item(), target_q=target_q.mean().item())

    
    def _update_sampler(self, batch):
        pass

    def _update_actor(self, batch):
            obs = batch['obs'].detach()
            with torch.no_grad():
                target_q = self.target_network.critic(batch['obs'])
                closed_solution_v = 1/self.eta * (torch.logsumexp( self.eta * target_q, dim=1))
            
            closed_solution_v = torch.cat([closed_solution_v.unsqueeze(1)] * self.env.action_space.n, dim=1)
            qs_pi = self.network.critic(obs)
            advantadge = qs_pi - closed_solution_v

            _, action_probs, log_prob = self.network.actor(obs)
            
            term_1 = torch.sum(action_probs * self.alpha.detach()*log_prob, dim=1)
            term_2 = torch.sum(action_probs * advantadge, dim=1)
            actor_loss = torch.sum(term_1 - term_2)

            self.optim['actor'].zero_grad()
            actor_loss.backward(retain_graph=True)
            self.optim['actor'].step()

            # Alpha Loss
            if self._alpha is None:
                # Update the learned temperature
                self.optim['log_alpha'].zero_grad(set_to_none=True)
                alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
                alpha_loss.backward()
                self.optim['log_alpha'].step()
                entropy = (-log_prob.mean())

            return dict(actor_loss=actor_loss.item(), entropy=entropy.item(), 
                    alpha_loss=alpha_loss.item(), alpha=self.alpha.detach().item())

     
    def _step_env(self):
        # Step the environment and store the transition data.
        metrics = dict()
        if self._env_steps < self.init_steps:
            action = self.env.action_space.sample()
        else:
            self.eval_mode()
            with torch.no_grad():
                action = self.predict(self._current_obs, sample=True)
            action += int(self.policy_noise * np.random.randn(1))
            self.train_mode()
        action = np.clip(action, self.action_range[0], self.action_range[-1])
        next_obs, reward, done, truncated, info = self.env.step(action)

        self._episode_length += 1
        self._episode_reward += reward

        if 'discount' in info:
            discount = info['discount']
        elif hasattr(self.env, "_max_episode_steps") and self._episode_length == self.env._max_episode_steps:
            discount = 1.0
        else:
            discount = 1 - float(done)

        # Store the consequences
        self.dataset.add(next_obs, action, reward, done, discount)
        
        if done:
            self._num_ep += 1
            # update metrics
            metrics['reward'] = self._episode_reward
            print(self._episode_reward)
            metrics['length'] = self._episode_length
            metrics['num_ep'] = self._num_ep
            # Reset the environment
            # self._current_obs, info = self.env.reset()
            self._current_obs = next_obs

            self.dataset.add(self._current_obs) # Add the first timestep
            self._episode_length = 0
            self._episode_reward = 0
        else:
            self._current_obs = next_obs

        self._env_steps += 1
        metrics['env_steps'] = self._env_steps
        return metrics

    def _setup_train(self):
        # Now setup the logging parameters
        self._current_obs, info = self.env.reset()
        self._episode_reward = 0
        self._episode_length = 0
        self._num_ep = 0
        self._env_steps = 0
        self.dataset.add(self._current_obs) # Store the initial reset observation!

    def _train_step(self, batch):
        all_metrics = {}

        if self.steps % self.env_freq == 0 or self._env_steps < self.init_steps:
            # step the environment with freq env_freq or if we are before learning starts
            metrics = self._step_env()
            all_metrics.update(metrics)
            if self._env_steps < self.init_steps:
                return all_metrics # return here.
        
        if 'obs' not in batch:
            return all_metrics

        updating_critic = self.steps % self.critic_freq == 0
        updating_actor = self.steps % self.actor_freq == 0

        if updating_actor or updating_critic:
            batch['obs'] = self.network.encoder(batch['obs'])
            with torch.no_grad():
                batch['next_obs'] = self.target_network.encoder(batch['next_obs'])
        
        if updating_critic:
            metrics = self._update_critic(batch)
            all_metrics.update(metrics)
            # metrics_sampler = self._update_sampler(batch)
            # all_metrics.update(metrics_sampler)

        if updating_actor:
            metrics = self._update_actor(batch)
            all_metrics.update(metrics)

        if self.steps % self.target_freq == 0:
            with torch.no_grad():
                for param, target_param in zip(self.network.encoder.parameters(), self.target_network.encoder.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.network.critic.parameters(), self.target_network.critic.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


        return all_metrics

    def _validation_step(self, batch):
        raise NotImplementedError("RL Algorithm does not have a validation dataset.")