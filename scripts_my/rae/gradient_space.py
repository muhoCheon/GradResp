"""Gradient parameter selection for standalone RAE."""

from __future__ import annotations
from typing import List, Tuple

import torch
import torch.nn as nn


def classifier_layer(net: nn.Module) -> nn.Module:
    if hasattr(net, 'get_fc_layer'):
        return net.get_fc_layer()
    if hasattr(net, 'fc'):
        return net.fc
    if hasattr(net, 'head'):
        return net.head
    if hasattr(net, 'heads'):
        heads = net.heads
        if isinstance(heads, nn.Sequential) and len(heads) > 0:
            return heads[0]
        return heads
    raise AttributeError(
        f'{net.__class__.__name__} does not expose a supported classifier '
        'layer (get_fc_layer/fc/head/heads).')


def classifier_layer_name(net: nn.Module) -> str:
    layer = classifier_layer(net)
    for name, module in net.named_modules():
        if module is layer:
            return name
    return ''


def trainable_named_parameters(module: nn.Module, prefix: str = ''):
    for name, param in module.named_parameters(recurse=True):
        if param.requires_grad:
            full = f'{prefix}.{name}' if prefix else name
            yield full, param


def select_gradient_parameters(net: nn.Module,
                               gradient_space: str,
                               ) -> List[Tuple[str, torch.nn.Parameter]]:
    if gradient_space == 'classifier':
        layer_name = classifier_layer_name(net)
        params = list(trainable_named_parameters(classifier_layer(net), layer_name))
    elif gradient_space == 'last_block':
        classifier_name = classifier_layer_name(net).split('.', 1)[0]
        candidates = []
        for name, module in net.named_children():
            if name == classifier_name:
                continue
            params = list(trainable_named_parameters(module, name))
            if params:
                candidates.append(params)
        if not candidates:
            raise ValueError('Unable to find a trainable non-classifier block')
        params = candidates[-1]
    elif gradient_space == 'all':
        params = [
            (name, param)
            for name, param in net.named_parameters()
            if param.requires_grad
        ]
    else:
        raise ValueError(f'Unknown gradient space: {gradient_space}')
    if not params:
        raise ValueError(
            f'No trainable parameters selected for gradient_space={gradient_space}')
    return params
