import copy
import os

import torch


def deep_copy_module(module):
    """Deep copy a module."""
    if isinstance(module, torch.jit.ScriptModule):
        module.save(module.original_name)
        out = torch.jit.load(module.original_name)
        os.system(f"rm {module.original_name}")
        return out
    return copy.deepcopy(module)


def accumulate_parameters(target_module, new_module, count):
    """Accumulate the parameters of target_target_module with those of new_module.
    The parameters of target_nn are replaced by:
        target_params <- (count * target_params + new_params) / (count + 1)
    Parameters
    ----------
    target_module: nn.Module
    new_module: nn.Module
    count: int.
    Returns
    -------
    None.
    """
    with torch.no_grad():
        target_state_dict = target_module.state_dict()
        new_state_dict = new_module.state_dict()

        for name in target_state_dict.keys():
            if target_state_dict[name] is new_state_dict[name]:
                continue
            else:
                if target_state_dict[name].data.ndim == 0:
                    target_state_dict[name].data = new_state_dict[name].data
                else:
                    target_state_dict[name].data[:] = (
                        count * target_state_dict[name].data + new_state_dict[name].data
                    ) / (count + 1)


def freeze_parameters(module):
    """Freeze all module parameters.
    Can be used to exclude module parameters from the graph.
    Parameters
    ----------
    module : torch.nn.Module
    """
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_parameters(module):
    """Unfreeze all module parameters.
    Can be used to include excluded module parameters in the graph.
    Parameters
    ----------
    module : torch.nn.Module
    """
    for param in module.parameters():
        param.requires_grad = True
