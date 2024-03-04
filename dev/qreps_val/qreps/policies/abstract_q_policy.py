"""Interface for Q-Function derived policies."""

from abc import ABCMeta

from qreps.policies import StochasticPolicy
from qreps.valuefunctions import AbstractQFunction


class AbstractQFunctionPolicy(StochasticPolicy, metaclass=ABCMeta):
    """Interface for policies to control an environment.

    Parameters
    ----------
    q_function: q_function to derive policy from.
    param: policy parameter.

    """

    def __init__(self, q_function: AbstractQFunction):
        super().__init__()
        self.q_function = q_function
