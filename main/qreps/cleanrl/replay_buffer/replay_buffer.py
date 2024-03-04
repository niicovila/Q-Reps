import collections
import random
import torch

def update(buffer, obs, new_observation, action, reward, done):
        """Sampling: Obtain N samples (s_i, a_i, s_i', r_i)

        Currently done using the Agent interface of DM.
        """
        buffer.push(obs, new_observation, action, reward, done)

class ReplayBuffer(object):
    """A simple Python replay buffer."""

    def __init__(self, capacity):
        self._prev = None
        self._action = None
        self._latest = None
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, obs, new_observation, action, reward, done):

        self.buffer.append(
            (
                obs,
                action,
                reward,
                new_observation,
                done
            )
        )

    def sample(self, batch_size):
        obs_tm1, a_tm1, r_t, discount_t, obs_t = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            torch.tensor(obs_tm1).float(),
            torch.tensor(a_tm1).float(),
            torch.tensor(r_t).float(),
            torch.tensor(discount_t).float(),
            torch.tensor(obs_t).float(),
        )

    def get_all(self):
        obs_tm1, a_tm1, r_t, obs_t, dones_t = zip(*self.buffer)
        
        if any(isinstance(x, tuple) for x in obs_tm1):
            print(True)
            obs_tm1 = [x[0] if isinstance(x, tuple) else x for x in obs_tm1]
        if any(isinstance(x, tuple) for x in obs_t):
            obs_t = [x[0] if isinstance(x, tuple) else x for x in obs_t]

        obs_tm1 = torch.cat(list(obs_tm1), dim=0).float()
        a_tm1 = torch.cat(list(a_tm1), dim=0).float()
        r_t = torch.cat(list(r_t), dim=0).float()
        obs_t = torch.cat(list(obs_t), dim=0).float()
        dones_t = torch.cat(list(dones_t), dim=0).float()
        
        return obs_tm1, a_tm1, r_t, obs_t, dones_t

    def is_ready(self, batch_size):
        return batch_size <= len(self.buffer)

    def reset(self):
        self.buffer.clear()

    def full(self):
        return self.buffer.maxlen == len(self.buffer)
