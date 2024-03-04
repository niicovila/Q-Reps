import dm_env
import numpy as np
from bsuite.baselines import base
from bsuite.baselines.utils.sequence import Trajectory


class Buffer:
    """A simple buffer for accumulating trajectories."""

    _observations: []
    _actions: []
    _rewards: []
    _discounts: []

    _max_sequence_length: int
    _needs_reset: bool = True
    _t: int = 0

    def __init__(
        self, max_sequence_length: int,
    ):
        """Pre-allocates buffers of numpy arrays to hold the sequences."""
        self._observations = [None] * (max_sequence_length + 1)
        self._actions = [None] * max_sequence_length
        self._rewards = np.zeros(max_sequence_length, dtype=np.float32)
        self._discounts = np.zeros(max_sequence_length, dtype=np.float32)

        self._max_sequence_length = max_sequence_length

    def append(
        self,
        timestep: dm_env.TimeStep,
        action: base.Action,
        new_timestep: dm_env.TimeStep,
    ):
        """Appends an observation, action, reward, and discount to the buffer."""
        if self.full():
            raise ValueError("Cannot append; sequence buffer is full.")

        # Start a new sequence with an initial observation, if required.
        if self._needs_reset:
            self._t = 0
            self._observations[self._t] = timestep.observation
            self._needs_reset = False

        # Append (o, a, r, d) to the sequence buffer.
        self._observations[self._t + 1] = new_timestep.observation
        self._actions[self._t] = action
        self._rewards[self._t] = new_timestep.reward
        self._discounts[self._t] = new_timestep.discount
        self._t += 1

        # Don't accumulate sequences that cross episode boundaries.
        # It is up to the caller to drain the buffer in this case.
        if new_timestep.last():
            self._needs_reset = True

    def drain(self) -> Trajectory:
        """Empties the buffer and returns the (possibly partial) trajectory."""
        if self.empty():
            raise ValueError("Cannot drain; sequence buffer is empty.")
        trajectory = Trajectory(
            self._observations[: self._t + 1],
            self._actions[: self._t],
            self._rewards[: self._t],
            self._discounts[: self._t],
        )
        self._t = 0  # Mark sequences as consumed.
        self._needs_reset = True
        return trajectory

    def empty(self) -> bool:
        """Returns whether or not the trajectory buffer is empty."""
        return self._t == 0

    def full(self) -> bool:
        """Returns whether or not the trajectory buffer is full."""
        return self._t == self._max_sequence_length
