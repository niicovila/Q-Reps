import random
from collections import deque
import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
        self.size = buffer_size

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: int, reward: float, terminated: bool) -> None:

        self.buffer.append((obs, next_obs, action, reward, terminated))

    def sample(self, batch_size):
        (obs, next_obs, action, reward, terminated) = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.tensor(obs).float(),
            torch.tensor(next_obs).float(),
            torch.tensor(action).float(),
            torch.tensor(reward).float(),
            torch.tensor(terminated).float(),
        )
    
    def get_all(self):
        (obs, next_obs, action, reward, terminated) =  zip(*self.buffer)
    
        return (
            torch.tensor(obs).float(),
            torch.tensor(next_obs).float(),
            torch.tensor(action).float(),
            torch.tensor(reward).float(),
            torch.tensor(terminated).float(),
        )

    def __len__(self):
        return len(self.buffer)
    
    def reset(self):
        self.buffer.clear()