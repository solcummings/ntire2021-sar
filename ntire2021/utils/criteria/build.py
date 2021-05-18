import torch
import torch.nn as nn


def build(loss_name, loss_args, **kwargs):
    if loss_args is not None and 'weight' in loss_args:
        _weight = loss_args['weight']
        loss_args['weight'] = torch.tensor(_weight) if not isinstance(_weight, torch.Tensor) else _weight

    loss_dict = {
            'ce': nn.CrossEntropyLoss,
            'mse': nn.MSELoss,
            'mae': nn.L1Loss,
    }
    return loss_dict[loss_name.lower()](**(loss_args or {}))

