import dm_env
import numpy as np
from torch import Tensor

from qreps.algorithms import AbstractAlgorithm


class OrderedDictFlattenTransform(AbstractAlgorithm):
    def calc_weights(
        self, features: Tensor, features_next: Tensor, rewards: Tensor, actions: Tensor
    ) -> Tensor:
        self._agent.calc_weights(features, features_next, rewards, actions)

    def __init__(self, agent: AbstractAlgorithm, identifiers: list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._agent = agent
        self.identifiers = identifiers

    def obs_transform(self, observation):
        arrays = [observation[identifier] for identifier in self.identifiers]
        return np.concatenate(arrays)

    def select_action(self, timestep: dm_env.TimeStep):
        timestep_1 = dm_env.TimeStep(
            timestep.step_type,
            timestep.reward,
            timestep.discount,
            self.obs_transform(timestep.observation),
        )
        return self._agent.select_action(timestep_1)

    def update(self, timestep, action, new_timestep,) -> None:
        timestep_1 = dm_env.TimeStep(
            timestep.step_type,
            timestep.reward,
            timestep.discount,
            self.obs_transform(timestep.observation),
        )
        new_timestep_1 = dm_env.TimeStep(
            new_timestep.step_type,
            new_timestep.reward,
            new_timestep.discount,
            self.obs_transform(new_timestep.observation),
        )
        self._agent.update(timestep_1, action, new_timestep_1)

    def update_policy(self, iteration):
        self._agent.update_policy(iteration)
