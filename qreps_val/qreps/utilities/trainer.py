import numpy as np
from dm_env import Environment, truncation


class Trainer:
    """Trainer for running environments together with algorithms"""

    def __init__(self, seed):
        self.env = None
        self.algo = None
        self.iter = 0
        self.seed = seed

    def setup(self, algo, env: Environment):
        self.env = env
        self.algo = algo

    def _obtain_episode(self, max_steps):
        obs, info = self.env.reset(seed=self.seed)
        step = 0
        done = False
        rewards = []
        while not done and step < max_steps:
            action = self.algo.select_action(obs)
            new_observation, reward, done, truncated, info = self.env.step(action)
            self.algo.update(obs, new_observation, action, reward, done)
            rewards.append(reward)

            # Book-keeping.
            obs = new_observation
            step += 1
        return np.sum(rewards)

    def _validate_once(self, max_steps):
        timestep = self.env.reset()
        step = 0
        rewards = []
        while not timestep.last():
            # Generate an action from the agent's policy.
            action = self.algo.select_action(timestep)
            # Step the environment.
            new_timestep = self.env.step(action)
            if step == max_steps:
                new_timestep = truncation(new_timestep.reward, new_timestep.observation)

            # Book-keeping.
            timestep = new_timestep
            step += 1
            rewards.append(new_timestep.reward)
        return np.sum(rewards)

    def validate(self, num_validation, max_steps):
        return [self._validate_once(max_steps) for _ in range(num_validation)]

    def train(self, num_iterations, max_steps, number_rollouts=1, logging_callback=None):

        for iteration in range(num_iterations):
            for rollout in range(number_rollouts):
                reward = self._obtain_episode(max_steps)
                if logging_callback is not None:
                    logging_callback(reward)

            self.algo.update_policy(self.iter)
            self.iter += 1
