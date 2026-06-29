"""Acceptance-direction computation for standalone RAE."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap
from tqdm import tqdm

from .config import NUMERIC_EPS
from .gradient_space import classifier_layer, select_gradient_parameters


def forward_logits_features(net: torch.nn.Module, data: torch.Tensor):
    try:
        output = net(data, return_feature=True)
    except TypeError:
        output = net(data)
    if isinstance(output, tuple) and len(output) >= 2:
        logits, features = output[0], output[1]
        return logits, features
    logits = output
    raise TypeError(
        f'{net.__class__.__name__} does not support return_feature=True; '
        'use a dense gradient space or adapt the model wrapper.')


def classifier_has_bias(net: torch.nn.Module) -> bool:
    layer = classifier_layer(net)
    return bool(getattr(layer, 'bias', None) is not None)


def classifier_residuals(probs: torch.Tensor,
                         labels: torch.Tensor) -> torch.Tensor:
    residuals = probs.clone()
    residuals[torch.arange(probs.shape[0], device=probs.device), labels] -= 1.0
    return residuals


def classifier_direction_dense(features: torch.Tensor,
                               probs: torch.Tensor,
                               labels: torch.Tensor,
                               *,
                               include_bias: bool = True,
                               eps: float = NUMERIC_EPS) -> torch.Tensor:
    residuals = classifier_residuals(probs, labels)
    weight_grad = residuals[:, :, None] * features[:, None, :]
    flat = weight_grad.flatten(start_dim=1)
    if include_bias:
        flat = torch.cat([flat, residuals], dim=1)
    directions = -flat
    return F.normalize(directions, p=2, dim=1, eps=eps)


def classifier_candidate_directions_dense(features: torch.Tensor,
                                          probs: torch.Tensor,
                                          candidate_classes: torch.Tensor,
                                          *,
                                          include_bias: bool = True,
                                          eps: float = NUMERIC_EPS
                                          ) -> torch.Tensor:
    rows = []
    for row_idx in range(features.shape[0]):
        labels = candidate_classes[row_idx].to(probs.device)
        row_probs = probs[row_idx].expand(labels.shape[0], -1)
        row_features = features[row_idx].expand(labels.shape[0], -1)
        rows.append(
            classifier_direction_dense(
                row_features,
                row_probs,
                labels,
                include_bias=include_bias,
                eps=eps,
            ))
    return torch.stack(rows, dim=0)


def normalized_grad_vector(loss: torch.Tensor,
                           params: Sequence[torch.nn.Parameter],
                           *,
                           eps: float = NUMERIC_EPS) -> torch.Tensor:
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )
    flat = torch.cat([grad.reshape(-1) for grad in grads])
    return -F.normalize(flat, p=2, dim=0, eps=eps)


def _parameter_names_for(net: torch.nn.Module,
                         params: Sequence[torch.nn.Parameter]) -> list[str]:
    names_by_id = {id(param): name for name, param in net.named_parameters()}
    names = []
    for param in params:
        name = names_by_id.get(id(param))
        if name is None:
            raise ValueError('Selected gradient parameter is not owned by the model')
        names.append(name)
    return names


def _dense_directions_vmap(net: torch.nn.Module,
                           data: torch.Tensor,
                           labels: torch.Tensor,
                           params: Sequence[torch.nn.Parameter],
                           *,
                           eps: float = NUMERIC_EPS) -> torch.Tensor:
    directions, _ = _dense_directions_and_norms_vmap(
        net, data, labels, params, eps=eps)
    return directions


def _dense_directions_and_norms_vmap(net: torch.nn.Module,
                                     data: torch.Tensor,
                                     labels: torch.Tensor,
                                     params: Sequence[torch.nn.Parameter],
                                     *,
                                     eps: float = NUMERIC_EPS
                                     ) -> tuple[torch.Tensor, torch.Tensor]:
    selected_names = _parameter_names_for(net, params)
    selected_name_set = set(selected_names)
    base_state = {
        name: param.detach()
        for name, param in net.named_parameters()
        if name not in selected_name_set
    }
    base_state.update({
        name: buffer.detach()
        for name, buffer in net.named_buffers()
    })
    selected_values = tuple(params)
    labels = labels.to(device=data.device).long()

    def loss_for_one(selected, sample, label):
        state = dict(base_state)
        state.update({
            name: value
            for name, value in zip(selected_names, selected)
        })
        logits = functional_call(net, state, (sample.unsqueeze(0),))
        return F.cross_entropy(logits, label.view(1))

    grads = vmap(grad(loss_for_one), in_dims=(None, 0, 0))(
        selected_values, data, labels)
    flat = torch.cat([item.reshape(item.shape[0], -1) for item in grads], dim=1)
    norms = torch.linalg.norm(flat, dim=1).clamp_min(eps)
    return -flat / norms[:, None], norms


def apply_direction_step(params: Sequence[torch.nn.Parameter],
                         direction: torch.Tensor,
                         step_size: float):
    """Apply a flat direction to params and return cloned originals."""
    originals = [param.detach().clone() for param in params]
    offset = 0
    with torch.no_grad():
        for param in params:
            width = param.numel()
            param.add_(step_size * direction[offset:offset + width].view_as(param))
            offset += width
    return originals


def restore_parameters(params: Sequence[torch.nn.Parameter], originals) -> None:
    with torch.no_grad():
        for param, original in zip(params, originals):
            param.copy_(original)


def dense_reference_directions(net: torch.nn.Module,
                               data: torch.Tensor,
                               labels: torch.Tensor,
                               params: Sequence[torch.nn.Parameter],
                               *,
                               eps: float = NUMERIC_EPS) -> torch.Tensor:
    return _dense_directions_vmap(net, data, labels, params, eps=eps)


def dense_candidate_directions(net: torch.nn.Module,
                               data: torch.Tensor,
                               candidate_classes: torch.Tensor,
                               params: Sequence[torch.nn.Parameter],
                               *,
                               eps: float = NUMERIC_EPS) -> torch.Tensor:
    batch_size, candidate_count = candidate_classes.shape
    flat_data = data[:, None].expand(
        batch_size, candidate_count, *data.shape[1:]).reshape(
            batch_size * candidate_count, *data.shape[1:])
    flat_labels = candidate_classes.reshape(batch_size * candidate_count)
    directions = _dense_directions_vmap(
        net,
        flat_data,
        flat_labels,
        params,
        eps=eps,
    )
    return directions.reshape(batch_size, candidate_count, -1)


def dense_candidate_directions_and_norms(
        net: torch.nn.Module,
        data: torch.Tensor,
        candidate_classes: torch.Tensor,
        params: Sequence[torch.nn.Parameter],
        *,
        eps: float = NUMERIC_EPS) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, candidate_count = candidate_classes.shape
    flat_data = data[:, None].expand(
        batch_size, candidate_count, *data.shape[1:]).reshape(
            batch_size * candidate_count, *data.shape[1:])
    flat_labels = candidate_classes.reshape(batch_size * candidate_count)
    directions, norms = _dense_directions_and_norms_vmap(
        net,
        flat_data,
        flat_labels,
        params,
        eps=eps,
    )
    return (
        directions.reshape(batch_size, candidate_count, -1),
        norms.reshape(batch_size, candidate_count),
    )


def compute_ref_grad_bank(net: torch.nn.Module,
                          loader,
                          *,
                          gradient_space: str,
                          device: torch.device,
                          eps: float = NUMERIC_EPS):
    net.eval()
    labels_all = []
    pred_all = []
    conf_all = []
    index_all = []
    image_names = []
    if gradient_space == 'classifier':
        features_all = []
        probs_all = []
        has_bias = classifier_has_bias(net)
        with torch.no_grad():
            for batch in tqdm(loader, desc='RAE reference classifier bank'):
                data = batch['data'].to(device)
                labels = batch['label'].to(device).long()
                logits, features = forward_logits_features(net, data)
                probs = torch.softmax(logits, dim=1)
                conf, pred = probs.max(dim=1)
                labels_all.append(labels.cpu().numpy())
                pred_all.append(pred.cpu().numpy())
                conf_all.append(conf.cpu().numpy())
                index_all.append(np.asarray(batch['index'], dtype=np.int64))
                image_names.extend([str(v) for v in batch.get('image_name', [])])
                features_all.append(features.detach().cpu().numpy().astype(np.float32))
                probs_all.append(probs.detach().cpu().numpy().astype(np.float32))
        return {
            'bank_type': 'classifier_compact',
            'labels': np.concatenate(labels_all).astype(np.int64),
            'pred': np.concatenate(pred_all).astype(np.int64),
            'conf': np.concatenate(conf_all).astype(np.float64),
            'sample_index': np.concatenate(index_all).astype(np.int64),
            'image_name': np.asarray(image_names, dtype=str),
            'features': np.concatenate(features_all).astype(np.float32),
            'probs': np.concatenate(probs_all).astype(np.float32),
            'classifier_has_bias': np.asarray(has_bias),
        }

    selected = select_gradient_parameters(net, gradient_space)
    params = [param for _, param in selected]
    directions_all = []
    for batch in tqdm(loader, desc=f'RAE reference {gradient_space} bank'):
        data = batch['data'].to(device)
        labels = batch['label'].to(device).long()
        with torch.no_grad():
            logits = net(data)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
        directions = dense_reference_directions(
            net, data, labels, params, eps=eps)
        labels_all.append(labels.cpu().numpy())
        pred_all.append(pred.cpu().numpy())
        conf_all.append(conf.cpu().numpy())
        index_all.append(np.asarray(batch['index'], dtype=np.int64))
        image_names.extend([str(v) for v in batch.get('image_name', [])])
        directions_all.append(directions.detach().cpu().numpy().astype(np.float32))
    return {
        'bank_type': 'dense',
        'labels': np.concatenate(labels_all).astype(np.int64),
        'pred': np.concatenate(pred_all).astype(np.int64),
        'conf': np.concatenate(conf_all).astype(np.float64),
        'sample_index': np.concatenate(index_all).astype(np.int64),
        'image_name': np.asarray(image_names, dtype=str),
        'directions': np.concatenate(directions_all).astype(np.float32),
        'parameter_names': np.asarray([name for name, _ in selected], dtype=str),
    }
