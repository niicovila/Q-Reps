import collections
import random
import numpy as np
import torch

class ReplayBuffer(object):
    """A simple Python replay buffer."""

    def __init__(self, capacity, device):
        self._prev = None
        self._action = None
        self._latest = None
        self.buffer = collections.deque(maxlen=capacity)
        self.device = device

    def push(self, obs, new_observation, action, reward, done, log_likes):

        self.buffer.append(
            (
                obs,
                new_observation,
                action,
                reward,
                done, 
                log_likes,
            )
        )

    def sample(self, batch_size):
        obs_tm1, obs_t, a_tm1, r_t, dones = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            torch.tensor(obs_tm1).float(),
            torch.tensor(obs_t).float(),
            torch.tensor(a_tm1).float(),
            torch.tensor(r_t).float(),
            torch.tensor(dones).float(),
        )

    def get_all(self):
        obs_tm1, obs_t, a_tm1, r_t, dones_t, log_likes_t = zip(*self.buffer)

        obs_tm1 = np.concatenate(obs_tm1, axis=0); obs_tm1 = obs_tm1.reshape(obs_tm1.shape[0], -1); obs_tm1 = torch.tensor(obs_tm1, device=self.device).float()
        a_tm1 = np.concatenate(a_tm1, axis=0); a_tm1 = a_tm1.reshape(a_tm1.shape[0], -1); a_tm1 = torch.tensor(a_tm1, device=self.device).float().flatten()
        r_t = np.concatenate(r_t, axis=0); r_t = r_t.reshape(r_t.shape[0], -1); r_t = torch.tensor(r_t, device=self.device).float().squeeze(-1)
        obs_t = np.concatenate(obs_t, axis=0); obs_t = obs_t.reshape(obs_t.shape[0], -1); obs_t = torch.tensor(obs_t, device=self.device).float()
        dones_t = np.concatenate(dones_t, axis=0); dones_t = dones_t.reshape(dones_t.shape[0], -1); dones_t = torch.tensor(dones_t, device=self.device).float()
        log_likes_t = torch.cat(log_likes_t, axis=0); log_likes_t = log_likes_t.reshape(log_likes_t.shape[0], -1).float()#; log_likes_t = torch.tensor(log_likes_t, device=self.device).float()
        
        return obs_tm1, obs_t, a_tm1, r_t, dones_t, log_likes_t

    def is_ready(self, batch_size):
        return batch_size <= len(self.buffer)

    def reset(self):
        self.buffer.clear()

    def full(self):
        return self.buffer.maxlen == len(self.buffer)
