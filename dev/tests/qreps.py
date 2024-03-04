"""Implementation of REPS Algorithm."""
from dataclasses import dataclass, field
import torch
import torch.distributions

from abstract_algorithm import AbstractAlgorithm
from valuefunction import SoftValueFunction
from policy import QREPSPolicy

@dataclass
class Loss:
    """Basic Loss class.

    Other Parameters
    ----------------
    losses: Tensor.
        Combined loss to optimize.
    td_error: Tensor.
        TD-Error of critic.
    policy_loss: Tensor.
        Loss of policy optimization.
    reg_loss: Tensor.
        Either KL-divergence or entropy bonus.
    dual_loss: Tensor.
        Loss of dual minimization problem.
    """

    combined_loss: torch.Tensor = field(init=False)
    td_error: torch.Tensor = torch.tensor(0.0)
    policy_loss: torch.Tensor = torch.tensor(0.0)
    critic_loss: torch.Tensor = torch.tensor(0.0)
    reg_loss: torch.Tensor = torch.tensor(0.0)
    dual_loss: torch.Tensor = torch.tensor(0.0)

    def __post_init__(self):
        """Post-initialize Loss dataclass.

        Fill in the attribute `loss' by adding all other losses.
        """
        self.combined_loss = (
            self.policy_loss + self.critic_loss + self.reg_loss + self.dual_loss
        )

    def __add__(self, other):
        """Add two losses."""
        return Loss(*map(lambda x: x[0] + x[1], zip(self, other)))

    def __sub__(self, other):
        """Add two losses."""
        return Loss(*map(lambda x: x[0] - x[1], zip(self, other)))

    def __neg__(self):
        """Substract two losses."""
        return Loss(*map(lambda x: -x, self))

    def __mul__(self, other):
        """Multiply losses by a scalar."""
        return Loss(*map(lambda x: x * other, self))

    def __rmul__(self, other):
        """Multiply losses by a scalar."""
        return self * other

    def __truediv__(self, other):
        """Divide losses by a scalar."""
        return Loss(*map(lambda x: x / other, self))

    def __iter__(self):
        """Iterate through the losses and yield all the separated losses.

        Notes
        -----
        It does not return the loss entry.
        It is useful to create new losses.
        """
        for key, value in self.__dict__.items():
            if key == "combined_loss":
                continue
            else:
                yield value

    def reduce(self, kind):
        """Reduce losses."""
        if kind == "sum":
            return Loss(*map(lambda x: x.sum(), self))
        elif kind == "mean":
            return Loss(*map(lambda x: x.mean(), self))
        elif kind == "none":
            return self
        else:
            raise NotImplementedError

def atleast_nd(input_tensor, n=1):
    """Make an input tensor at least `n`-dimensional.

    Parameters
    ----------
    input_tensor: Tensor
        Tensor to expand.
    n: int
        Integer of minimum number of dimensions to have.

    Returns
    -------
    output_tensor: Tensor.
        Expanded Tensor.
    """
    if input_tensor.ndim > n:
        raise ValueError(
            f"The size of input tensor ({input_tensor.ndim}) is larger than n ({n})"
        )
    while input_tensor.ndim < n:
        input_tensor = input_tensor.unsqueeze(-1)
    return input_tensor

def broadcast_to_tensor(input_tensor, target_tensor):
    """Broadcast an input tensor to a target tensor shape.

    Parameters
    ----------
    input_tensor: Tensor
    target_tensor: Tensor

    Returns
    -------
    output_tensor: Tensor.

    """
    if input_tensor.shape == target_tensor.shape:
        return input_tensor

    # First expand index to target_tensor ndim.
    input_tensor = atleast_nd(input_tensor, n=target_tensor.ndim)

    # Then repeat along index where dimensions do not match.
    for idx, size in enumerate(target_tensor.shape):
        if input_tensor.shape[idx] != size:
            input_tensor = input_tensor.repeat_interleave(size, dim=idx)
    assert input_tensor.shape == target_tensor.shape
    return input_tensor

class QREPS(Ab):
    r"""Q-Relative Entropy Policy Search Algorithm.

    Q-REPS optimizes the following regularized LP over the vectors
    \mu(X, A) and d(X, A).

    ..math::  \max \mu r - eta R(\mu, d)
    ..math::  s.t. \sum_a d(x, a) = \sum_{x', a'} = \mu(x', a') P(x|x', a'),
    ..math::  s.t.  d(x, a) = \mu(x, a)
    ..math::  s.t.  1/2 (\mu + d) is a distribution.

    The LP dual is:
    ..math::  G(V) = \eta \log \sum_{x, a} d(x, a) 0.5 (
                \exp^{\delta(x, a) / \eta} + \exp^{(Q(x, a) - V(x)) / \eta}).


    where \delta(x,a) = r + \sum_{x'} P(x'|x, a) V(x') - V(x) is the TD-error and V(x)
    are the dual variables associated with the stationary constraints in the primal,
    and Q(x, a) are the dual variables associated with the equality of distributions
    constraint.
    V(x) is usually referred to as the value function and Q(x, a) as the q-function.

    Using d(x,a) as the empirical distribution, G(V) can be approximated by samples.

    The optimal policy is given by:
    ..math::  \pi(a|x) \propto d(x, a) \exp^{Q(x, a) / \eta}.

    By default, the policy is a soft-max policy.
    However, if Q-REPS is initialized with a parametric policy it can be fit by
    minimizing the negative log-likelihood at the sampled elements.


    Calling REPS() returns a sampled based estimate of G(V) and the NLL of the policy.
    Both G(V) and NLL lend are differentiable and lend themselves to gradient based
    optimization.


    References
    ----------
    Peters, J., Mulling, K., & Altun, Y. (2010, July).
    Relative entropy policy search. AAAI.

    Deisenroth, M. P., Neumann, G., & Peters, J. (2013).
    A survey on policy search for robotics. Foundations and Trends® in Robotics.
    """

    def __init__(
        self, q_function, eta, alpha=None, learn_policy=False, *args, **kwargs
    ):
        kwargs.pop("critic", None)
        kwargs.pop("policy", None)
        if alpha is None:
            alpha = eta
        critic = SoftValueFunction(q_function, param=alpha)
        policy = QREPSPolicy(q_function, param=alpha)
        super().__init__(
            reps_eta=eta,
            critic=critic,
            policy=policy,
            learn_policy=learn_policy,
            entropy_regularization=True,
            *args,
            **kwargs
        )
        self.q_function = q_function

    def get_value_target(self, observation):
        """Get value-function target."""
        next_v = self.critic(observation.next_state)
        not_done = broadcast_to_tensor(1.0 - observation.done, target_tensor=next_v)
        return self.get_reward(observation) + self.gamma * next_v * not_done
    
    def actor_loss(self, observation):
        """Return primal and dual loss terms from Q-REPS."""
        state, action, reward, next_state, done, *r = observation

        # Calculate dual variables
        value = self.critic(state)
        target = self.get_value_target(observation)
        q_value = self.q_function(state, action)

        td = target - q_value
        self._info.update(td=td)

        # Calculate weights.
        weights_td = self.eta() * td  # type: torch.Tensor
        if weights_td.ndim == 1:
            weights_td = weights_td.unsqueeze(-1)
        dual = 1 / self.eta() * torch.logsumexp(weights_td, dim=-1)
        dual += (1 - self.gamma) * value.squeeze(-1)
        return Loss(dual_loss=dual.mean(), td_error=td)
