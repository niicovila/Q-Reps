"""Python Script Template."""
import torch

from qreps.policies.abstract_q_policy import AbstractQFunctionPolicy
from qreps.valuefunctions import AbstractQFunction


class QREPSPolicy(AbstractQFunctionPolicy):
    """Implementation of a softmax policy with some small off-set for stability."""

    def __init__(self, q_function: AbstractQFunction, temp):
        super().__init__(q_function)
        self.counter = 0
        self.temperature = temp

    def reset(self):
        """Reset parameters and update counter."""
        self.counter += 1

    def distribution(self, x) -> torch.distributions.Distribution:
        q_values = self.q_function.forward_state(x)
        return torch.distributions.Categorical(logits=q_values)

    def log_likelihood(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return self.distribution(features).log_prob(actions)

    def sample(self, observation: torch.Tensor):
        return self.distribution(observation).sample().item()
