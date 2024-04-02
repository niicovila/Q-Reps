import random
import torch
import numpy as np
import itertools

from cmath import log
import os
import time
import torch
import numpy as np
import random
import copy
from torch.distributions import Categorical
from abc import ABC, abstractmethod
from collections import defaultdict

import research
from research.processors.base import IdentityProcessor
from research.utils.logger import Logger
from research.utils import utils
from research.utils.evaluate import eval_policy

import wandb
from torch.utils.tensorboard import SummaryWriter

from .base import Algorithm
from research.networks.base import ActorCriticValuePolicy
from research.utils import utils
from functools import partial

def log_from_dict(logger, metric_lists, prefix):
    keys_to_remove = []
    for metric_name, metric_value in metric_lists.items():
        if isinstance(metric_value, list) and len(metric_value) > 0:
            logger.record(prefix + "/" + metric_name, np.mean(metric_value))
            keys_to_remove.append(metric_name)
        else:
            logger.record(prefix + "/" + metric_name, metric_value)
            keys_to_remove.append(metric_name)
    for key in keys_to_remove:
        del metric_lists[key]

def _worker_init_fn(worker_id):
    state = np.random.get_state()
    new_state = list(state)
    new_state[2] += worker_id
    np.random.set_state(tuple(new_state))
    random.seed(new_state[2])

MAX_VALID_METRICS = {"reward", "accuracy", "success", "is_success"}

class ExponentiatedGradientSampler:
    def __init__(self, N, device, eta, beta=0.01):
        self.n = N
        self.eta = eta
        self.beta = beta

        self.h = torch.ones((self.n,)) / N
        self.z = torch.ones((self.n,)) / N

        self.prob_dist = Categorical(torch.ones((self.n,))/ N)
        self.device = device

    def reset(self):
        self.h = torch.ones((self.n,))
        self.z = torch.ones((self.n,))
        self.prob_dist = Categorical(torch.softmax(torch.ones((self.n,)), 0))
                                     
    def probs(self):
        return self.prob_dist.probs.to(self.device)
    
    def entropy(self):
        return self.prob_dist.entropy().to(self.device)
    
    def update(self, bellman):
        self.h = bellman -  self.eta * torch.log(self.n * self.probs())
        t = self.beta*self.h
        self.z = self.probs() * torch.exp(t)
        self.z = torch.clamp(self.z / (torch.sum(self.z)), min=1e-8, max=1.0)
        self.prob_dist = Categorical(self.z)

def gumbel_loss(pred, label, beta, clip):
    assert pred.shape == label.shape, "Shapes were incorrect"
    z = (label - pred)/beta
    if clip is not None:
        z = torch.clamp(z, -clip, clip)
    loss = torch.exp(z) - z - 1
    return loss

def gumbel_rescale_loss(pred, label, beta, clip):
    assert pred.shape == label.shape, "Shapes were incorrect"
    z = (label - pred)/beta
    if clip is not None:
        z = torch.clamp(z, -clip, clip)
    max_z = torch.max(z)
    max_z = torch.where(max_z < -1.0, torch.tensor(-1.0, dtype=torch.float, device=max_z.device), max_z)
    max_z = max_z.detach() # Detach the gradients
    loss = torch.exp(z - max_z) - z*torch.exp(-max_z) - torch.exp(-max_z)    
    return loss

def gumbel_log_loss_v3(pred, label, beta, clip):
    assert pred.shape == label.shape, "Shapes were incorrect"
    z = (label - pred)/beta
    if clip is not None:
        z = torch.clamp(z, -clip, clip)
    loss = torch.exp(z)
    loss_2 =  z + 1
    loss = loss.mean() - loss_2.mean()
    return torch.log(loss)

def mse_loss(pred, label):
    return (label - pred)**2

def qreps_loss(pred, label, beta, clip, gamma, value):
    assert pred.shape == label.shape, "Shapes were incorrect"
    z = (label - pred)/beta
    if clip is not None:
        z = torch.clamp(z, -clip, clip)
    loss = torch.exp(z)
    loss = loss.mean()

    return beta * torch.log(torch.mean(loss)) + (1-gamma)*value.mean()

def saddle(label, pred, sampler, eta, gamma, values):
    bellman = label - pred
    errors = torch.sum(sampler.probs().detach() * (bellman - eta * torch.log(sampler.n * sampler.probs().detach()))) + (1 - gamma) * values.mean()
    
    return errors

class GumbelSACREPS(Algorithm):

    def __init__(self, env, network_class, dataset_class, 
                       tau=0.005,
                       init_temperature=0.1,
                       critic_freq=1,
                       value_freq=1,
                       actor_freq=2,
                       target_freq=2,
                       env_freq=1,
                       init_steps=1000,
                       alpha=None,
                       exp_clip=10,
                       beta=1.0,
                       loss="gumbel",
                       value_action_noise=0.0,
                       use_value_log_prob=False,
                       max_grad_norm=None,
                       max_grad_value=None,
                       **kwargs):
        '''
        Note that regular SAC (with value function) is recovered by loss="mse" and use_value_log_prob=True
        '''
        # Save values needed for optim setup.
        self.init_temperature = init_temperature
        self._alpha = alpha
        self.use_saddle = False
        # Initialize wandb
        run_name = f'xsac_{beta}_QREPS'
        wandb.init(
            project='XSAC_REPS',
            entity=None,
            sync_tensorboard=True,
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        self.writer = SummaryWriter(f"runs/{run_name}")
        super().__init__(env, network_class, dataset_class, **kwargs)
        assert isinstance(self.network, ActorCriticValuePolicy)

        # Save extra parameters
        self.tau = tau
        self.critic_freq = critic_freq
        self.value_freq = value_freq
        self.actor_freq = actor_freq
        self.target_freq = target_freq
        self.env_freq = env_freq
        self.init_steps = init_steps
        self.exp_clip = exp_clip
        self.beta = beta
        self.loss = loss
        self.value_action_noise = value_action_noise
        self.use_value_log_prob = use_value_log_prob
        self.action_range = (self.env.action_space.low, self.env.action_space.high)
        self.action_range_tensor = utils.to_device(utils.to_tensor(self.action_range), self.device)
        # Gradient clipping
        self.max_grad_norm = max_grad_norm
        self.max_grad_value = max_grad_value

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
        self.optim['value'] = optim_class(self.network.value.parameters(), **optim_kwargs)
        
        self.target_entropy = -np.prod(self.env.action_space.low.shape)
        if self._alpha is None:
            # Setup the learned entropy coefficients. This has to be done first so its present in the setup_optim call.
            self.log_alpha = torch.tensor(np.log(self.init_temperature), dtype=torch.float).to(self.device)
            self.log_alpha.requires_grad = True
            self.optim['log_alpha'] = optim_class([self.log_alpha], **optim_kwargs)
        else:
            self.log_alpha = torch.tensor(np.log(self._alpha), dtype=torch.float).to(self.device)
            self.log_alpha.requires_grad = False
    
    def _step_env(self):
        # Step the environment and store the transition data.
        metrics = dict()
        if self._env_steps < self.init_steps:
            action = self.env.action_space.sample()
        else:
            self.eval_mode()
            with torch.no_grad():
                action = self.predict(self._current_obs, sample=True)
            self.train_mode()
        
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        next_obs, reward, done, info = self.env.step(action)
        self._episode_length += 1
        self._episode_reward += reward

        if 'discount' in info:
            discount = info['discount']
        elif hasattr(self.env, "_max_episode_steps") and self._episode_length == self.env._max_episode_steps:
            discount = 1.0
        else:
            discount = 1 - float(done)

        # Store the consequences.
        self.dataset.add(next_obs, action, reward, done, discount)

        if done:
            self._num_ep += 1
            # update metrics
            metrics['reward'] = self._episode_reward
            metrics['length'] = self._episode_length
            metrics['num_ep'] = self._num_ep
            # Reset the environment
            self._current_obs = self.env.reset()
            self.dataset.add(self._current_obs) # Add the first timestep
            self._episode_length = 0
            self._episode_reward = 0
        else:
            self._current_obs = next_obs

        self._env_steps += 1
        metrics['env_steps'] = self._env_steps
        return metrics

    def _setup_train(self):
        self._current_obs = self.env.reset()
        self._episode_reward = 0
        self._episode_length = 0
        self._num_ep = 0
        self._env_steps = 0
        self.dataset.add(self._current_obs) # Store the initial reset observation!

    def _train_step(self, batch, sampler):
        all_metrics = {}

        if self.steps % self.env_freq == 0 or self._env_steps < self.init_steps:
            # step the environment with freq env_freq or if we are before learning starts
            metrics = self._step_env()
            all_metrics.update(metrics)
            if self._env_steps < self.init_steps:
                return all_metrics # return here.
        
        if 'obs' not in batch:
            return all_metrics
        
        batch['obs'] = self.network.encoder(batch['obs'])
        with torch.no_grad():
            batch['next_obs'] = self.target_network.encoder(batch['next_obs'])

        if self.steps % self.critic_freq == 0:
            # Q Loss:
            with torch.no_grad():
                value = self.target_network.value(batch['obs'])
                target_v = self.target_network.value(batch['next_obs'])
                target_q = batch['reward'] + batch['discount']*target_v

            qs = self.network.critic(batch['obs'], batch['action'])
            # Note: Could also just compute the mean over a broadcasted target. TO investigate later.
            #loss_fn = gumbel_rescale_loss # qreps_loss
            
            if self.use_saddle: q_loss = sum([saddle(qs[i], target_q, sampler, self.beta, batch['discount'][0], value) for i in range(qs.shape[0])])
            else: q_loss = sum([qreps_loss(qs[i], target_q, self.beta, self.exp_clip, batch['discount'][0], value) for i in range(qs.shape[0])])
            self.optim['critic'].zero_grad(set_to_none=True)
            q_loss.backward()
            self.optim['critic'].step()
            
            bellman = sum([target_q - qs[i] for i in range(qs.shape[0])])
            sampler.update(bellman)

            all_metrics['q_loss'] = q_loss.item()
            all_metrics['target_q'] = target_q.mean().item()
        
        # Get the Q value of the current policy. This is used for the value and actor
        dist = self.network.actor(batch['obs'])  
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        qs_pi = self.network.critic(batch['obs'], action)
        q_pred = torch.min(qs_pi, dim=0)[0]

        if self.steps % self.value_freq == 0:

            if self.value_action_noise > 0.0:
                with torch.no_grad():
                    noise = (torch.randn_like(action) * self.value_action_noise).clamp(-0.5, 0.5)
                    noisy_action = (action.detach() + noise).clamp(*self.action_range_tensor)
                    qs_noisy = self.network.critic(batch['obs'], noisy_action)
                    q_noisy = torch.min(qs_noisy, dim=0)[0]
                    target_v_pi = (q_noisy).detach()
            else:
                target_v_pi = (q_pred).detach()

            if self.use_value_log_prob:
                target_v_pi = (target_v_pi  - self.alpha.detach() * log_prob).detach()
        
            v_pred = self.network.value(batch['obs'])
            if self.loss == "gumbel":
                value_loss_fn = partial(gumbel_loss, beta=self.beta, clip=self.exp_clip) # (v_pred, target_v_pi, beta, self.exp_clip)
            elif self.loss == "gumbel_rescale":
                value_loss_fn = partial(gumbel_rescale_loss, beta=self.beta, clip=self.exp_clip)
            elif self.loss == "gumbel_log_v3":
                value_loss_fn = partial(gumbel_log_loss_v3, beta=self.beta, clip=self.exp_clip)
            elif self.loss == "mse":
                value_loss_fn = mse_loss
            else:
                raise ValueError("Invalid loss specified.")
            value_loss = value_loss_fn(v_pred, target_v_pi)
            value_loss = value_loss.mean()

            self.optim['value'].zero_grad(set_to_none=True)
            value_loss.backward()
            # Gradient clipping
            if self.max_grad_value is not None:
                torch.nn.utils.clip_grad_value_(self.network.value.parameters(), self.max_grad_value)
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.network.value.parameters(), self.max_grad_norm)
            self.optim['value'].step()

            all_metrics['value_loss'] = value_loss.item()
            all_metrics['target_v'] = target_v_pi.mean().item()

        if self.steps % self.actor_freq == 0:
            # Actor Loss
            actor_loss = (self.alpha.detach() * log_prob - q_pred).mean()

            self.optim['actor'].zero_grad(set_to_none=True)
            actor_loss.backward()
            self.optim['actor'].step()

            # Alpha Loss
            if self._alpha is None:
                # Update the learned temperature
                self.optim['log_alpha'].zero_grad(set_to_none=True)
                alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
                alpha_loss.backward()
                self.optim['log_alpha'].step()
                all_metrics['alpha_loss'] = alpha_loss.item()
        
            all_metrics['actor_loss'] = actor_loss.item()
            all_metrics['entropy'] = (-log_prob.mean()).item()
            all_metrics['alpha'] = self.alpha.detach().item()
        
        if self.steps % self.target_freq == 0:
            # Only update the critic and encoder for speed. Ignore the actor.
            with torch.no_grad():
                for param, target_param in zip(self.network.encoder.parameters(), self.target_network.encoder.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.network.critic.parameters(), self.target_network.critic.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.network.value.parameters(), self.target_network.value.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return all_metrics
    
    def train(self, path, total_steps, schedule=None, schedule_kwargs={}, 
                    log_freq=100, eval_freq=1000, max_eval_steps=-1, workers=4, loss_metric="loss", 
                    eval_ep=-1, profile_freq=-1, use_wandb=False, x_axis="steps"):
        
        writers = ['tb', 'csv']
        if use_wandb:
            writers.append('wandb')
        logger = Logger(path=path, writers=writers)

        # Construct the dataloaders.
        self.setup_datasets()
        shuffle = not issubclass(self.dataset_class, torch.utils.data.IterableDataset)
        pin_memory = self.device.type == "cuda"
        worker_init_fn = _worker_init_fn if workers > 0 else None
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, 
                                                 shuffle=shuffle, 
                                                 num_workers=workers, worker_init_fn=worker_init_fn,
                                                 pin_memory=pin_memory, 
                                                 collate_fn=self.collate_fn)
        if self.validation_dataset is not None:
            validation_dataloader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=self.batch_size, 
                                                            shuffle=shuffle, 
                                                            num_workers=0, 
                                                            pin_memory=pin_memory,
                                                            collate_fn=self.collate_fn)
        else:
            validation_dataloader = None

        # Create schedulers for the optimizers
        schedulers = {}
        if schedule is not None:
            for name, opt in self.optim.items():
                schedulers[name] = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=self.schedule_fn(total_steps, **schedule_kwargs))

        # Setup model metrics.
        self._steps = 0
        self._epochs = 0
        self._total_steps = total_steps
        current_step = 0
        train_metric_lists = defaultdict(list)
        best_validation_metric = -1*float('inf') if loss_metric in MAX_VALID_METRICS else float('inf')
        last_train_log = 0
        last_validation_log = 0
        
        # Setup training
        self._setup_train()
        self.network.train()

        # Setup profiling immediately before we start the loop.
        start_time = current_time = time.time()
        profiling_metric_lists = defaultdict(list)
        stop = False
        
        while current_step < total_steps:
            
            for batch in dataloader:
             sampler = ExponentiatedGradientSampler(1024, self.device, self.beta, beta=0.002)
             for epoch in range(100):
                # Profiling
                if profile_freq > 0 and self._steps % profile_freq == 0:
                    stop_time = time.time()
                    profiling_metric_lists['dataset'].append(stop_time - current_time)
                    current_time = stop_time

                batch = self._format_batch(batch)

                if profile_freq > 0 and self._steps % profile_freq == 0:
                    stop_time = time.time()
                    profiling_metric_lists['preprocess'].append(stop_time - current_time)
                    current_time = stop_time

                # Train the network
                assert self.network.training, "Network was not in training mode and trainstep was called."
                train_metrics = self._train_step(batch, sampler)
                for metric_name, metric_value in train_metrics.items():
                    train_metric_lists[metric_name].append(metric_value)

                if profile_freq > 0 and self._steps % profile_freq == 0:
                    stop_time = time.time()
                    profiling_metric_lists['train_step'].append(stop_time - current_time)

                # Increment the number of training steps.
                self._steps += 1

                # Update the schedulers
                for scheduler in schedulers.values():
                    scheduler.step()

                # Compute the current step. This is so we can use other metrics
                if x_axis in train_metrics:
                    current_step = train_metrics[x_axis]
                elif x_axis == "epoch":
                    current_step = self.epochs
                else:
                    current_step = self._steps

                if (current_step - last_train_log) >= log_freq:
                    # Timing metrics
                    current_time = time.time()
                    logger.record("time/steps", self._steps)
                    logger.record("time/epochs", self._epochs)
                    logger.record("time/steps_per_second", (current_step - last_train_log) / (current_time - start_time))
                    start_time = current_time
                    # Record Other metrics
                    for name, scheduler in schedulers.items():
                        logger.record("lr/" + name, scheduler.get_last_lr()[0])
                    log_from_dict(logger, profiling_metric_lists, "time")
                    log_from_dict(logger, train_metric_lists, "train")
                    logger.dump(step=current_step)
                    last_train_log = current_step

                if (current_step - last_validation_log) >= eval_freq:
                    self.eval_mode()
                    current_validation_metric = None
                    if not validation_dataloader is None:
                        eval_steps = 0
                        validation_metric_lists = defaultdict(list)
                        for batch in validation_dataloader:
                            batch = self._format_batch(batch)
                            losses = self._validation_step(batch)
                            for metric_name, metric_value in losses.items():
                                validation_metric_lists[metric_name].append(metric_value)
                            eval_steps += 1
                            if eval_steps == max_eval_steps:
                                break

                        if loss_metric in validation_metric_lists:
                            current_validation_metric = np.mean(validation_metric_lists[loss_metric])
                        log_from_dict(logger, validation_metric_lists, "valid")

                    # Now run any extra validation steps, independent of the validation dataset.
                    validation_extras = self._validation_extras(path, self._steps, validation_dataloader)
                    if loss_metric in validation_extras:
                        current_validation_metric = validation_extras[loss_metric]
                    log_from_dict(logger, validation_extras, "valid")

                    # Evaluation episodes
                    if self.eval_env is not None and eval_ep > 0:
                        eval_metrics = eval_policy(self.eval_env, self, eval_ep)
                        # if eval_metrics['reward'] < 100 and current_step >= 125000:
                        #     print("Stopping early due to poor performance")
                        #     stop = True
                        #     break
                        if loss_metric in eval_metrics:
                            current_validation_metric = eval_metrics[loss_metric]
                        log_from_dict(logger, eval_metrics, "eval")

                    if current_validation_metric is None:
                        pass
                    elif loss_metric in MAX_VALID_METRICS and current_validation_metric > best_validation_metric:
                        self.save(path, "best_model")
                        best_validation_metric = current_validation_metric
                    elif current_validation_metric < best_validation_metric:
                        self.save(path, "best_model")
                        best_validation_metric = current_validation_metric

                    # Eval Logger Dump to CSV
                    logger.dump(step=current_step, eval=True) # Mark True on the eval flag
                    last_validation_log = current_step
                    self.save(path, "final_model") # Also save the final model every eval period.
                    self.train_mode()

                # Profiling
                if profile_freq > 0 and self._steps % profile_freq == 0:
                    current_time = time.time()

                if current_step >= total_steps:
                    break
                
            self._epochs += 1
            if stop: break
        logger.close()

    def _validation_step(self, batch):
        raise NotImplementedError("RL Algorithm does not have a validation dataset.")