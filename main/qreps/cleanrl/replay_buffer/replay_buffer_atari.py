import collections
import random
import numpy as np
import torch

class ReplayBufferAtari(object):
    """A simple Python replay buffer."""

    def __init__(self, capacity):
        self._prev = None
        self._action = None
        self._latest = None
        self.buffer = collections.deque(maxlen=capacity)

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
        
        obs_tm1 = torch.tensor(np.concatenate(obs_tm1, axis=0)).float()
        a_tm1 = torch.tensor(np.concatenate(a_tm1, axis=0).reshape(-1)).float()  # Flattening directly
        r_t = torch.tensor(np.concatenate(r_t, axis=0).reshape(-1)).float()  # Flattening directly
        obs_t = torch.tensor(np.concatenate(obs_t, axis=0)).float()
        dones_t = torch.tensor(np.concatenate(dones_t, axis=0).reshape(-1)).float()  # Flattening directly
        log_likes_t = (torch.cat(log_likes_t, axis=0).reshape(-1)).float()  # Flattening directly
        
        return obs_tm1, obs_t, a_tm1, r_t, dones_t, log_likes_t

    def is_ready(self, batch_size):
        return batch_size <= len(self.buffer)

    def reset(self):
        self.buffer.clear()

    def full(self):
        return self.buffer.maxlen == len(self.buffer)