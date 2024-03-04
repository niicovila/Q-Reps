import math

import torch


def logmeanexp(values: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """
    Logmeanexp using the logsumexp trick and the mean as part of the exponent.

    @param values: values that are in the exponent
    @param args: args to give to logsumexp (dim, ...)
    @param kwargs: kwargs to give to logsumexp
    @return: the logmeanexp of the values
    """
    n = values.shape[0]
    return torch.logsumexp(values + math.log(1 / n), *args, **kwargs)
