#!/usr/bin/env python
"""Evaluate TARR scores with an independent script.

TARR intentionally stays outside the OpenOOD postprocessor registry here. The
method mutates model parameters per target sample, so this script owns the
adapt/restore loop and writes OpenOOD-compatible score files.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from openood.datasets.imglist_dataset import ImglistDataset
import openood.evaluation_api.evaluator as evaluator_module
from openood.evaluation_api import Evaluator
from openood.evaluation_api.datasets import DATA_INFO
from openood.evaluation_api.preprocessor import get_default_preprocessor
from openood.evaluators.metrics import compute_all_metrics
from openood.networks import ResNet18_224x224, ResNet18_32x32, ResNet50
from openood.postprocessors import BasePostprocessor

from scripts_my.tarr.adaptation import (
    OBJECTIVES,
    PERTURBATION_CACHE_POLICIES,
    PERTURBATION_KINDS,
    PERTURBATION_RESPONSES,
    RUNTIME_IMPL_VERSION,
    RUNTIME_MODES,
    UPDATE_SCOPES,
    perturbation_config_dict,
    perturbation_config_id,
    resolve_runtime_mode as resolve_runtime_mode_value,
    resolve_tta_update_impl,
    tta_config_dict,
    tta_config_id,
)
from scripts_my.tarr.reference import (
    REFERENCE_FILTERS,
    parse_reference_configs,
    selected_reference_hash,
)
import scripts_my.tarr.reference as reference_helpers
from scripts_my.tarr.protocol import (
    expected_csid_datasets,
    far_dataset_names,
    near_dataset_names,
    supported_dataset_names,
    uses_evaluator_csid_loaders,
)
from scripts_my.tarr.scoring import (
    CACHE_SCHEMA_VERSION,
    DELTA_DEFINITION,
    PERTURBATION_DEFINITION,
    PERTURBATION_SCORE_DIRECTION,
    PROBE_SCORE_RULES,
    SCORE_DIRECTION,
    SCORE_RULE_CHOICES,
    ood_score_from_cache,
    score_from_delta,
    selected_perturbation_score_rules,
    selected_probe_score_rules,
    selected_score_rules,
)

SUPPORTED_DATASETS = supported_dataset_names()
NUM_CLASSES = {
    'cifar10': 10,
    'cifar100': 100,
    'imagenet': 1000,
    'imagenet200': 200,
}
MODEL_ARCH = {
    'cifar10': ResNet18_32x32,
    'cifar100': ResNet18_32x32,
    'imagenet': ResNet50,
    'imagenet200': ResNet18_224x224,
}
DEFAULT_CHECKPOINT = {
    'cifar10':
    'results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt',
    'cifar100':
    'results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt',
    'imagenet':
    'results/pretrained_weights/resnet50_imagenet1k_v1.pth',
    'imagenet200':
    'results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s0/best.ckpt',
}

SCHEMES = ['ood', 'fsood', 'both']
BASELINE_PROTOCOLS = ['main_py', 'eval_api']
TRAIN_CANDIDATE_METADATA_SCHEMA_VERSION = 2
PERTURBATION_RESPONSE_CODES = {
    'none': 0,
    'pixel': 1,
    'feature': 2,
}
PERTURBATION_KIND_CODES = {
    'gaussian': 0,
    'sign_ce': 1,
}
PERTURBATION_CACHE_POLICY_CODES = {
    'auto': 0,
    'error_on_feature_cache': 1,
}
ANCHOR_LOSS_TYPES = ['none', 'ce', 'distill', 'param_reg']
ACCEPT_PROBE_TYPES = [
    'predicted_label_ce',
    'entropy_min',
    'view_consistency',
]
REJECT_PROBE_TYPES = [
    'entropy_max',
    'uniform',
    'logit_suppression',
]
TTA_MODES = ['normal', 'ar_bank']
SOFT_VIEW_OBJECTIVES = {
    'view_consistency_kl',
    'view_consistency_js',
    'entropy_consistency',
}
VIEW_PERTURBATION_OBJECTIVES = SOFT_VIEW_OBJECTIVES | {'memo_marginal_entropy'}
DEFAULT_PREFETCH_FACTOR = 2


def parse_response_steps(value, max_steps):
    if value is None or str(value).strip() == '':
        return [int(max_steps)]
    steps = []
    for raw in str(value).split(','):
        token = raw.strip()
        if not token:
            continue
        try:
            step = int(token)
        except ValueError as exc:
            raise ValueError(
                f'--save-steps entries must be integers: {token}') from exc
        if step < 1:
            raise ValueError('--save-steps entries must be positive.')
        if step > int(max_steps):
            raise ValueError(
                f'--save-steps entry {step} exceeds --steps {max_steps}.')
        steps.append(step)
    if not steps:
        raise ValueError('--save-steps must contain at least one step.')
    return sorted(set(steps))


def parse_probe_type_list(value, choices, option_name):
    if value is None or str(value).strip() == '':
        return None
    items = []
    seen = set()
    for raw in str(value).split(','):
        token = raw.strip()
        if not token:
            continue
        if token not in choices:
            raise ValueError(
                f'{option_name} entry {token!r} is invalid; choices are '
                f'{", ".join(choices)}.')
        if token in seen:
            raise ValueError(
                f'{option_name} contains duplicate probe type {token!r}.')
        seen.add(token)
        items.append(token)
    if not items:
        raise ValueError(f'{option_name} must contain at least one probe type.')
    return items


def normalize_probe_banks(args, parser):
    try:
        accept_bank = parse_probe_type_list(
            args.accept_probe_types, ACCEPT_PROBE_TYPES, '--accept-probe-types')
        reject_bank = parse_probe_type_list(
            args.reject_probe_types, REJECT_PROBE_TYPES, '--reject-probe-types')
    except ValueError as exc:
        parser.error(str(exc))

    if args.tta_mode == 'normal':
        if args.objective is None:
            parser.error('--tta-mode normal requires --objective.')
        if accept_bank is not None or reject_bank is not None:
            parser.error(
                '--accept-probe-types/--reject-probe-types are valid only '
                'with --tta-mode ar_bank.')
        args.use_accept_reject_probe = False
        args.use_response_bank = False
        args.accept_probe_type_bank = []
        args.reject_probe_type_bank = []
        args.accept_probe_type = ''
        args.reject_probe_type = ''
        args.primary_accept_branch_id = ''
        args.primary_reject_branch_id = ''
        args.accept_branch_ids = []
        args.reject_branch_ids = []
        return

    if args.objective is not None:
        parser.error('--objective is valid only with --tta-mode normal.')
    if accept_bank is None:
        parser.error('--tta-mode ar_bank requires --accept-probe-types.')
    if reject_bank is None:
        parser.error('--tta-mode ar_bank requires --reject-probe-types.')

    args.use_accept_reject_probe = True
    args.accept_probe_type_bank = accept_bank
    args.reject_probe_type_bank = reject_bank
    args.use_response_bank = True

    # Primary fields keep the first branch available for existing cache readers.
    args.accept_probe_type = accept_bank[0]
    args.reject_probe_type = reject_bank[0]
    args.primary_accept_branch_id = accept_bank[0]
    args.primary_reject_branch_id = reject_bank[0]
    args.accept_branch_ids = list(accept_bank)
    args.reject_branch_ids = list(reject_bank)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    command = 'run-all'
    if argv and argv[0] in {'run-all', 'run-response'}:
        command = argv[0]
        argv = argv[1:]
    parser.add_argument('--dataset', required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument('--checkpoint')
    parser.add_argument('--output-root', default='results_test/tarr')
    parser.add_argument(
        '--baseline-protocol',
        default='main_py',
        choices=BASELINE_PROTOCOLS,
        help=('Baseline protocol to match for comparison. For CIFAR-10 FSOOD, '
              'main_py uses cinic10 as csID and eval_api uses cifar10c.'),
    )
    parser.add_argument('--run-id')
    parser.add_argument(
        '--experiment-tag',
        default='',
        help='Free-form label such as strict_transfer or c100_tuned.',
    )
    parser.add_argument(
        '--ablation-type',
        default='auto',
        help=('Experiment ablation label, e.g. baseline, anchor_only, '
              'accept_reject_only, accept_reject_anchor, anchor_sweep, '
              'probe_sweep, or score_sweep. Logged in manifests only.'),
    )
    parser.add_argument('--overwrite',
                        action='store_true',
                        help='Allow writing into an existing run directory.')
    parser.add_argument('--reference-per-class', type=int, default=4)
    parser.add_argument('--reference-filter',
                        default='all',
                        choices=REFERENCE_FILTERS)
    parser.add_argument('--reference-min-confidence', type=float, default=0.9)
    parser.add_argument(
        '--train-candidate-metadata-root',
        help=('Directory for train split metadata artifacts. Defaults to '
              '<output-root>/train_candidate_metadata.'),
    )
    parser.add_argument(
        '--rebuild-train-candidate-metadata',
        action='store_true',
        help='Rebuild train candidate metadata even if identity matches.',
    )
    parser.add_argument(
        '--reference-config',
        action='append',
        default=[],
        help=('Repeatable multi-reference spec: '
              '<id>:per_class=<int>,filter=<name>,min_confidence=<float>,seed=<int>. '
              'If omitted, the single-reference options are used.'),
    )
    parser.add_argument('--tta-mode',
                        required=True,
                        choices=TTA_MODES,
                        help=('normal runs one target TTA objective; ar_bank '
                              'runs independent acceptance/rejection response '
                              'banks.'))
    parser.add_argument('--objective',
                        default=None,
                        choices=OBJECTIVES,
                        help='Normal TARR update objective. Valid only with --tta-mode normal.')
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument(
        '--save-steps',
        default='',
        help=('Comma-separated update steps at which to save target/reference '
              'responses. Defaults to --steps. Example: 5,10,30.'),
    )
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--update-scope',
                        default='classifier',
                        choices=UPDATE_SCOPES)
    parser.add_argument(
        '--runtime-mode',
        default='auto',
        choices=RUNTIME_MODES,
        help=('Runtime implementation. auto uses full classifier feature '
              'caching for classifier updates and full forward for all-parameter '
              'updates.'),
    )
    parser.add_argument('--perturbation-response',
                        default='none',
                        choices=PERTURBATION_RESPONSES)
    parser.add_argument('--perturbation-kind',
                        default='gaussian',
                        choices=PERTURBATION_KINDS)
    parser.add_argument('--perturbation-eps', type=float, default=0.0)
    parser.add_argument('--perturbation-repeats', type=int, default=1)
    parser.add_argument('--perturbation-seed', type=int, default=0)
    parser.add_argument('--perturbation-cache-policy',
                        default='auto',
                        choices=PERTURBATION_CACHE_POLICIES)
    parser.add_argument('--score-rule',
                        default='predicted_class_loss_increase',
                        choices=SCORE_RULE_CHOICES)
    parser.add_argument('--use-anchor-reference',
                        action='store_true',
                        help='Use prebuilt anchor_set artifacts in the update loss.')
    parser.add_argument(
        '--anchor-set-root',
        help='Directory for reusable anchor_set artifacts. Defaults to <output-root>/anchor_sets.',
    )
    parser.add_argument('--anchor-loss-type',
                        default='none',
                        choices=ANCHOR_LOSS_TYPES)
    parser.add_argument(
        '--accept-probe-types',
        default='',
        help=('Comma-separated acceptance probe bank. Required with '
              '--tta-mode ar_bank. The first branch is the primary cache view.'),
    )
    parser.add_argument(
        '--reject-probe-types',
        default='',
        help=('Comma-separated rejection probe bank. Required with '
              '--tta-mode ar_bank. The first branch is the primary cache view.'),
    )
    parser.add_argument('--anchor-weight', type=float, default=0.0)
    parser.add_argument('--freeze-bn-stats',
                        dest='freeze_bn_stats',
                        action='store_true',
                        default=True)
    parser.add_argument('--no-freeze-bn-stats',
                        dest='freeze_bn_stats',
                        action='store_false')
    parser.add_argument(
        '--save-tta-response',
        action='store_true',
        help='Save per-sample TTA response tensors for scoring.',
    )
    parser.add_argument(
        '--tta-response-shard-size',
        type=int,
        default=0,
        help=('0 writes the single tta_response/<dataset>.npz artifact. A '
              'positive value streams TTA responses to '
              'tta_response/<dataset>/part_*.npz shards of this many '
              'samples. Required for full ImageNet/ImageNet-200 cache writes.'),
    )
    parser.add_argument(
        '--debug-output-mode',
        default='full',
        choices=['full', 'none'],
        help=('full writes debug_samples*.csv as before. none skips debug row '
              'accumulation and CSV writing for large full runs.'),
    )
    parser.add_argument('--scheme', default='both', choices=SCHEMES)
    parser.add_argument(
        '--near-datasets',
        default='all',
        help='Comma-separated near-OOD dataset names, or all.',
    )
    parser.add_argument(
        '--far-datasets',
        default='all',
        help='Comma-separated far-OOD dataset names, or all.',
    )
    parser.add_argument('--max-samples',
                        type=int,
                        default=0,
                        help='Limit every inference loader. 0 means all.')
    parser.add_argument('--max-id-samples',
                        type=int,
                        default=0,
                        help='Limit ID and csID loaders. Overrides --max-samples.')
    parser.add_argument('--max-ood-samples',
                        type=int,
                        default=0,
                        help='Limit OOD loaders. Overrides --max-samples.')
    parser.add_argument(
        '--target-shard-count',
        type=int,
        default=1,
        help='Number of modulo shards to split target inference samples into.')
    parser.add_argument(
        '--target-shard-index',
        type=int,
        default=0,
        help='Process target samples whose global index modulo count matches this index.')
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--reference-set-batch-size', type=int, default=512)
    parser.add_argument(
        '--train-candidate-batch-size',
        type=int,
        default=0,
        help=('Batch size for building train candidate metadata. '
              '0 reuses --batch-size.'),
    )
    parser.add_argument(
        '--reference-set-root',
        help=('Directory for reusable reference_set artifacts. Defaults '
              'to <output-root>/reference_sets.'),
    )
    parser.add_argument(
        '--use-prebuilt-reference-set',
        action='store_true',
        help=('Stage 3 strict mode: load prebuilt reference_set artifacts and '
              'fail if any requested reference_set is missing.'),
    )
    parser.add_argument(
        '--rebuild-reference-set',
        action='store_true',
        help='Rebuild reference_set artifacts even if identity matches.',
    )
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-progress', action='store_true')
    args = parser.parse_args(argv)
    args.command = command
    if args.steps < 1:
        parser.error('--steps must be positive.')
    try:
        args.response_steps = parse_response_steps(args.save_steps, args.steps)
    except ValueError as exc:
        parser.error(str(exc))
    normalize_probe_banks(args, parser)
    if args.perturbation_eps < 0:
        parser.error('--perturbation-eps must be non-negative.')
    if args.perturbation_repeats < 1:
        parser.error('--perturbation-repeats must be at least 1.')
    if args.train_candidate_batch_size < 0:
        parser.error('--train-candidate-batch-size must be non-negative.')
    if args.tta_response_shard_size < 0:
        parser.error('--tta-response-shard-size must be non-negative.')
    if args.target_shard_count < 1:
        parser.error('--target-shard-count must be at least 1.')
    if args.target_shard_index < 0:
        parser.error('--target-shard-index must be non-negative.')
    if args.target_shard_index >= args.target_shard_count:
        parser.error('--target-shard-index must be less than --target-shard-count.')
    if args.anchor_weight < 0:
        parser.error('--anchor-weight must be non-negative.')
    if args.use_anchor_reference:
        if args.anchor_loss_type == 'none':
            parser.error('--use-anchor-reference requires --anchor-loss-type != none.')
        if args.anchor_weight <= 0:
            parser.error('--use-anchor-reference requires --anchor-weight > 0.')
        if args.anchor_loss_type in {'ce', 'distill'}:
            if args.update_scope != 'classifier':
                parser.error(
                    'anchor_loss_type ce/distill requires --update-scope classifier.')
            if args.runtime_mode not in {'auto', 'classifier_feature_cache'}:
                parser.error(
                    'anchor_loss_type ce/distill requires classifier feature cache runtime.')
    if not args.use_anchor_reference and args.anchor_loss_type != 'none':
        parser.error('--anchor-loss-type != none requires --use-anchor-reference.')
    if args.score_rule == 'probe_all' and args.tta_mode != 'ar_bank':
        parser.error('--score-rule probe_all requires --tta-mode ar_bank.')
    if args.score_rule in PROBE_SCORE_RULES and args.tta_mode != 'ar_bank':
        parser.error(
            f'--score-rule {args.score_rule} requires --tta-mode ar_bank.')
    if (args.tta_mode == 'ar_bank'
            and 'view_consistency' in args.accept_probe_type_bank):
        if args.perturbation_response not in {'pixel', 'feature'}:
            parser.error(
                'accept_probe_type=view_consistency requires '
                '--perturbation-response pixel or feature.')
        if args.perturbation_kind != 'gaussian':
            parser.error(
                'accept_probe_type=view_consistency currently supports only '
                '--perturbation-kind gaussian.')
        if args.perturbation_eps <= 0:
            parser.error('accept_probe_type=view_consistency requires --perturbation-eps > 0.')
        if args.perturbation_repeats < 2:
            parser.error(
                'accept_probe_type=view_consistency requires '
                '--perturbation-repeats >= 2.')
    if args.objective in VIEW_PERTURBATION_OBJECTIVES:
        if args.perturbation_response not in {'pixel', 'feature'}:
            parser.error(
                f'{args.objective} requires --perturbation-response pixel '
                'or feature.')
        if args.perturbation_kind != 'gaussian':
            parser.error(
                f'{args.objective} currently supports only '
                '--perturbation-kind gaussian.')
        if args.perturbation_eps <= 0:
            parser.error(f'{args.objective} requires --perturbation-eps > 0.')
        if args.perturbation_repeats < 2:
            parser.error(
                f'{args.objective} requires --perturbation-repeats >= 2.')
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def dataloader_runtime_kwargs(num_workers):
    num_workers = int(num_workers or 0)
    kwargs = {'pin_memory': torch.cuda.is_available()}
    if num_workers > 0:
        kwargs.update(
            persistent_workers=True,
            prefetch_factor=DEFAULT_PREFETCH_FACTOR,
        )
    return kwargs


def with_runtime_dataloader_kwargs(base_factory):
    def factory(id_name, data_root, preprocessor, **loader_kwargs):
        loader_kwargs = dict(loader_kwargs)
        loader_kwargs.update(
            dataloader_runtime_kwargs(loader_kwargs.get('num_workers', 0)))
        return base_factory(id_name, data_root, preprocessor, **loader_kwargs)

    return factory


def load_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


def build_model(dataset):
    return MODEL_ARCH[dataset](num_classes=NUM_CLASSES[dataset])


def classifier_layer(net):
    if hasattr(net, 'get_fc_layer'):
        return net.get_fc_layer()
    if hasattr(net, 'fc'):
        return net.fc
    raise AttributeError(
        f'{net.__class__.__name__} does not expose a supported classifier '
        'layer. ResNet-style models should provide get_fc_layer() or fc.')


def classifier_layer_name(net):
    layer = classifier_layer(net)
    for name, module in net.named_modules():
        if module is layer:
            return name or '<root>'
    return '<unknown>'


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def tensor_stats(value):
    stats = torch.stack((
        value.mean(),
        value.std(unbiased=False),
        value.min(),
        value.max(),
    )).detach().cpu()
    return {
        'mean': float(stats[0].item()),
        'std': float(stats[1].item()),
        'min': float(stats[2].item()),
        'max': float(stats[3].item()),
    }


def classwise_tensor_stats(values, labels, num_classes):
    stats = {}
    for class_id in range(num_classes):
        class_values = values[labels == class_id]
        if class_values.numel() == 0:
            stats[str(class_id)] = {
                'count': 0,
                'mean': None,
                'std': None,
                'min': None,
                'max': None,
            }
        else:
            row = tensor_stats(class_values)
            row['count'] = int(class_values.numel())
            stats[str(class_id)] = row
    return stats


def vector_diagnostics(values):
    stats = torch.stack((
        values.mean(),
        values.std(unbiased=False),
        values.min(),
        values.max(),
    )).detach().cpu()
    return {
        'mean': float(stats[0].item()),
        'std': float(stats[1].item()),
        'min': float(stats[2].item()),
        'max': float(stats[3].item()),
    }


def logit_diagnostics(logits):
    probs = torch.softmax(logits, dim=1)
    conf, pred = torch.max(probs, dim=1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
    if probs.shape[1] > 1:
        top2 = torch.topk(probs, k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
    else:
        margin = conf
    energy = -torch.logsumexp(logits, dim=1)
    return {
        'probs': probs,
        'conf': conf,
        'pred': pred,
        'entropy': entropy,
        'margin': margin,
        'energy': energy,
    }


def save_npz(path, pred, conf, label):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        pred=as_numpy(pred).astype(np.int64),
        conf=as_numpy(conf).astype(np.float64),
        label=as_numpy(label).astype(np.int64),
    )


def save_tta_response(path, scores):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'pred': scores['pred'].astype(np.int64),
        'label': scores['label'].astype(np.int64),
        'score_rules': np.asarray(selected_score_rules(scores['args_score_rule'])),
        'args_score_rule': np.asarray(scores['args_score_rule']),
        'reference_config_id': np.asarray(scores['reference_config_id']),
        'cache_schema_version': np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int64),
        'score_direction': np.asarray(SCORE_DIRECTION),
        'delta_definition': np.asarray(DELTA_DEFINITION),
        'perturbation_score_rules': np.asarray(
            selected_perturbation_score_rules('all')),
        'perturbation_score_direction': np.asarray(PERTURBATION_SCORE_DIRECTION),
        'perturbation_definition': np.asarray(PERTURBATION_DEFINITION),
    }
    int_keys = {
        'y_hat',
        'target_global_index',
        'perturbation_response_code',
        'perturbation_kind_code',
        'perturbation_repeats',
        'perturbation_seed',
        'perturbation_cache_policy_code',
        'post_tta_pred',
        'adapted_target_pred',
        'target_pred_changed',
    }
    float32_keys = {
        'target_probs',
        'post_tta_target_probs',
        'adapted_target_probs',
        'base_reference_loss',
        'adapted_reference_loss',
        'delta',
        'reference_conf_delta_by_class',
        'reference_entropy_delta_by_class',
        'reference_margin_delta_by_class',
        'reference_energy_delta_by_class',
        'reference_pred_changed_rate_by_class',
        'reference_correct_rate_before_by_class',
        'reference_correct_rate_after_by_class',
        'accept_ref_loss_delta',
        'reject_ref_loss_delta',
        'accept_ref_loss_delta_bank',
        'reject_ref_loss_delta_bank',
    }
    string_scalar_keys = {
        'tta_mode',
        'perturbation_config_id',
        'perturbation_response',
        'perturbation_kind',
        'perturbation_cache_policy',
        'probe_config_id',
        'accept_probe_type',
        'reject_probe_type',
        'primary_accept_branch_id',
        'primary_reject_branch_id',
        'anchor_loss_type',
    }
    for key in RESPONSE_CACHE_SCALAR_KEYS + RESPONSE_CACHE_ARRAY_KEYS:
        value = scores[key]
        if key in int_keys:
            payload[key] = value.astype(np.int64)
        elif key in float32_keys:
            payload[key] = value.astype(np.float32)
        elif key in string_scalar_keys:
            payload[key] = np.asarray(value)
        else:
            payload[key] = value.astype(np.float64)
    for key in PROBE_RESPONSE_SCALAR_KEYS + PROBE_RESPONSE_ARRAY_KEYS:
        if key not in scores:
            continue
        value = scores[key]
        if key in int_keys:
            payload[key] = value.astype(np.int64)
        elif key in float32_keys:
            payload[key] = value.astype(np.float32)
        elif key in string_scalar_keys:
            payload[key] = np.asarray(value)
        else:
            payload[key] = value.astype(np.float64)
    for key in RESPONSE_CACHE_CONFIG_KEYS + PROBE_RESPONSE_CONFIG_KEYS:
        if key in scores:
            payload[key] = np.asarray(scores[key])
    np.savez_compressed(path, **payload)


RESPONSE_CACHE_SCALAR_KEYS = [
    'target_global_index',
    'y_hat',
    'target_conf',
    'target_entropy',
    'target_margin',
    'target_energy',
    'perturbation_logit_l2',
    'perturbation_prob_l1',
    'perturbation_conf_delta',
    'perturbation_entropy_delta',
    'perturbation_response_code',
    'perturbation_kind_code',
    'perturbation_eps',
    'perturbation_repeats',
    'perturbation_seed',
    'perturbation_cache_policy_code',
    'target_tta_loss_before',
    'target_tta_loss_after',
    'post_tta_pred',
    'post_tta_target_conf',
    'post_tta_target_entropy',
    'post_tta_pseudo_label_prob',
    'adapted_target_pred',
    'adapted_target_conf',
    'adapted_target_entropy',
    'adapted_target_margin',
    'adapted_target_energy',
    'target_conf_delta',
    'target_entropy_delta',
    'target_margin_delta',
    'target_energy_delta',
    'target_pred_changed',
    'base_reference_loss_mean',
    'base_reference_loss_std',
    'base_reference_loss_min',
    'base_reference_loss_max',
    'adapted_reference_loss_mean',
    'adapted_reference_loss_std',
    'adapted_reference_loss_min',
    'adapted_reference_loss_max',
    'reference_delta_mean',
    'reference_delta_std',
    'reference_delta_min',
    'reference_delta_max',
    'reference_delta_positive_mean',
    'runtime_per_sample',
]

RESPONSE_CACHE_ARRAY_KEYS = [
    'target_probs',
    'post_tta_target_probs',
    'adapted_target_probs',
    'base_reference_loss',
    'adapted_reference_loss',
    'delta',
    'reference_conf_delta_by_class',
    'reference_entropy_delta_by_class',
    'reference_margin_delta_by_class',
    'reference_energy_delta_by_class',
    'reference_pred_changed_rate_by_class',
    'reference_correct_rate_before_by_class',
    'reference_correct_rate_after_by_class',
]

RESPONSE_CACHE_CONFIG_KEYS = [
    'tta_mode',
    'response_steps',
    'target_shard_count',
    'target_shard_index',
    'perturbation_config_id',
    'perturbation_response',
    'perturbation_kind',
    'perturbation_cache_policy',
    'perturbation_eps_config',
    'perturbation_repeats_config',
    'perturbation_seed_config',
]

PROBE_RESPONSE_SCALAR_KEYS = [
    'use_accept_reject_probe',
    'use_anchor_reference',
    'anchor_weight',
    'probe_schema_version',
    'accept_target_objective_delta',
    'reject_target_objective_delta',
    'reject_target_entropy_delta',
    'accept_target_objective_delta_bank',
    'reject_target_objective_delta_bank',
    'reject_target_entropy_delta_bank',
]

PROBE_RESPONSE_ARRAY_KEYS = [
    'accept_ref_loss_delta',
    'reject_ref_loss_delta',
    'accept_ref_loss_delta_bank',
    'reject_ref_loss_delta_bank',
]

PROBE_RESPONSE_CONFIG_KEYS = [
    'probe_config_id',
    'accept_probe_type',
    'reject_probe_type',
    'accept_branch_ids',
    'accept_branch_probe_types',
    'reject_branch_ids',
    'reject_branch_probe_types',
    'primary_accept_branch_id',
    'primary_reject_branch_id',
    'response_bank_schema_version',
    'anchor_loss_type',
    'probe_score_rules',
    'probe_score_rule_arg',
]


def probe_response_config_metadata(args):
    metadata = {
        'probe_config_id': probe_config_id(args),
        'accept_probe_type': args.accept_probe_type,
        'reject_probe_type': args.reject_probe_type,
        'anchor_loss_type': args.anchor_loss_type,
    }
    if getattr(args, 'use_response_bank', False):
        metadata.update({
            'accept_branch_ids':
            np.asarray(args.accept_branch_ids, dtype='<U64'),
            'accept_branch_probe_types':
            np.asarray(args.accept_probe_type_bank, dtype='<U64'),
            'reject_branch_ids':
            np.asarray(args.reject_branch_ids, dtype='<U64'),
            'reject_branch_probe_types':
            np.asarray(args.reject_probe_type_bank, dtype='<U64'),
            'primary_accept_branch_id': args.primary_accept_branch_id,
            'primary_reject_branch_id': args.primary_reject_branch_id,
            'response_bank_schema_version': 1,
        })
    if args.use_accept_reject_probe:
        metadata['probe_score_rules'] = np.asarray(
            selected_probe_score_rules('probe_all'))
        metadata['probe_score_rule_arg'] = 'probe_all'
    return metadata


def empty_tta_response_lists():
    return {
        key: []
        for key in RESPONSE_CACHE_SCALAR_KEYS + RESPONSE_CACHE_ARRAY_KEYS
    }


def tta_response_scores_from_lists(cache_lists, pred_array, label_array,
                                    args, reference_config_id):
    scores = {
        'pred': np.asarray(pred_array, dtype=np.int64),
        'label': np.asarray(label_array, dtype=np.int64),
    }
    int_scalar_keys = {
        'y_hat',
        'target_global_index',
        'perturbation_response_code',
        'perturbation_kind_code',
        'perturbation_repeats',
        'perturbation_seed',
        'perturbation_cache_policy_code',
        'post_tta_pred',
        'adapted_target_pred',
        'target_pred_changed',
        'use_accept_reject_probe',
        'use_anchor_reference',
        'probe_schema_version',
    }
    for key in RESPONSE_CACHE_SCALAR_KEYS:
        dtype = np.int64 if key in int_scalar_keys else np.float64
        scores[key] = np.asarray(cache_lists[key], dtype=dtype)
    for key in RESPONSE_CACHE_ARRAY_KEYS:
        scores[key] = np.stack(cache_lists[key])
    for key in PROBE_RESPONSE_SCALAR_KEYS:
        if key in cache_lists and cache_lists[key]:
            dtype = np.int64 if key in int_scalar_keys else np.float64
            scores[key] = np.asarray(cache_lists[key], dtype=dtype)
    for key in PROBE_RESPONSE_ARRAY_KEYS:
        if key in cache_lists and cache_lists[key]:
            scores[key] = np.stack(cache_lists[key])
    scores.update({
        'args_score_rule': args.score_rule,
        'reference_config_id': reference_config_id,
        'tta_mode': args.tta_mode,
        'response_steps': np.asarray(args.response_steps, dtype=np.int64),
        'target_shard_count': int(args.target_shard_count),
        'target_shard_index': int(args.target_shard_index),
        'perturbation_config_id': perturbation_config_id(args),
        'perturbation_response': args.perturbation_response,
        'perturbation_kind': args.perturbation_kind,
        'perturbation_cache_policy': args.perturbation_cache_policy,
        'perturbation_eps_config': float(args.perturbation_eps),
        'perturbation_repeats_config': int(args.perturbation_repeats),
        'perturbation_seed_config': int(args.perturbation_seed),
    })
    scores.update(probe_response_config_metadata(args))
    return scores


class TTAResponseShardWriter:
    """Streaming writer for ImageNet-scale TTA responses."""

    def __init__(self, response_dir, dataset_name, reference_config_id, args):
        self.response_dir = Path(response_dir)
        self.dataset_name = dataset_name
        self.reference_config_id = reference_config_id
        self.args = args
        self.shard_size = int(args.tta_response_shard_size)
        if self.shard_size <= 0:
            raise ValueError('tta_response shard size must be positive')
        self.dataset_dir = self.response_dir / dataset_name
        if self.dataset_dir.exists():
            if args.overwrite:
                shutil.rmtree(self.dataset_dir)
            else:
                raise FileExistsError(
                    f'Sharded tta_response already exists: {self.dataset_dir}. '
                    'Use --overwrite to replace it.')
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.dataset_dir / 'manifest.json'
        self.buffer = empty_tta_response_lists()
        self.pred_buffer = []
        self.label_buffer = []
        self.shards = []
        self.num_samples = 0
        self.shard_index = 0
        write_json(
            self.manifest_path,
            {
                'storage': 'sharded_npz',
                'complete': False,
                'cache_schema_version': CACHE_SCHEMA_VERSION,
                'dataset_name': self.dataset_name,
                'reference_config_id': self.reference_config_id,
                'score_rule_arg': args.score_rule,
                'expanded_score_rules': selected_score_rules(args.score_rule),
                'probe_config': probe_config_dict(args),
                'response_steps': list(args.response_steps),
                'target_shard_count': int(args.target_shard_count),
                'target_shard_index': int(args.target_shard_index),
                'shard_size': self.shard_size,
                'num_samples': 0,
                'num_shards': 0,
                'shards': [],
            },
        )

    def add(self, pred, label, cache):
        self.pred_buffer.append(int(pred))
        self.label_buffer.append(int(label))
        for key in RESPONSE_CACHE_SCALAR_KEYS + RESPONSE_CACHE_ARRAY_KEYS:
            self.buffer[key].append(cache[key])
        for key in PROBE_RESPONSE_SCALAR_KEYS + PROBE_RESPONSE_ARRAY_KEYS:
            if key in cache:
                self.buffer.setdefault(key, []).append(cache[key])
        if len(self.pred_buffer) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self.pred_buffer:
            return
        start = self.num_samples
        count = len(self.pred_buffer)
        final_path = self.dataset_dir / f'part_{self.shard_index:06d}.npz'
        temp_path = self.dataset_dir / f'part_{self.shard_index:06d}.tmp.npz'
        scores = tta_response_scores_from_lists(
            self.buffer,
            np.asarray(self.pred_buffer, dtype=np.int64),
            np.asarray(self.label_buffer, dtype=np.int64),
            self.args,
            self.reference_config_id,
        )
        save_tta_response(temp_path, scores)
        temp_path.replace(final_path)
        self.shards.append({
            'path': final_path.name,
            'start': int(start),
            'end': int(start + count),
            'num_samples': int(count),
        })
        self.num_samples += count
        self.shard_index += 1
        self.buffer = empty_tta_response_lists()
        self.pred_buffer = []
        self.label_buffer = []

    def close(self):
        self.flush()
        write_json(
            self.manifest_path,
            {
                'storage': 'sharded_npz',
                'complete': True,
                'cache_schema_version': CACHE_SCHEMA_VERSION,
                'dataset_name': self.dataset_name,
                'reference_config_id': self.reference_config_id,
                'score_rule_arg': self.args.score_rule,
                'expanded_score_rules': selected_score_rules(self.args.score_rule),
                'probe_config': probe_config_dict(self.args),
                'response_steps': list(self.args.response_steps),
                'target_shard_count': int(self.args.target_shard_count),
                'target_shard_index': int(self.args.target_shard_index),
                'shard_size': self.shard_size,
                'num_samples': int(self.num_samples),
                'num_shards': len(self.shards),
                'shards': self.shards,
            },
        )
        return {
            'storage': 'sharded_npz',
            'manifest': str(self.manifest_path),
            'num_samples': int(self.num_samples),
            'num_shards': len(self.shards),
            'target_shard_count': int(self.args.target_shard_count),
            'target_shard_index': int(self.args.target_shard_index),
        }


def write_metrics_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['dataset', 'FPR@95', 'AUROC', 'AUPR_IN', 'AUPR_OUT', 'ACC']
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


RUNTIME_BENCHMARK_FIELDS = [
    'run_id',
    'dataset',
    'baseline_protocol',
    'scheme',
    'reference_config_id',
    'runtime_mode',
    'train_candidate_metadata_schema_version',
    'train_candidate_metadata_reused',
    'train_candidate_metadata_sec',
    'reference_set_reused',
    'reference_set_sec',
    'setup_total_sec',
    'inference_total_sec',
    'scoring_total_sec',
    'processed_targets',
    'runtime_per_target_sec',
    'batch_size',
    'train_candidate_batch_size',
    'reference_set_batch_size',
    'num_workers',
    'cuda_device',
]


def append_runtime_benchmark_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=RUNTIME_BENCHMARK_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def file_sha256(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolved_checkpoint(args):
    return args.checkpoint or DEFAULT_CHECKPOINT[args.dataset]


def train_candidate_metadata_root(args):
    root = args.train_candidate_metadata_root
    if root:
        return Path(root)
    return Path(args.output_root) / 'train_candidate_metadata'


def reference_set_root(args):
    root = args.reference_set_root
    if root:
        return Path(root)
    return Path(args.output_root) / 'reference_sets'


def anchor_set_root(args):
    root = args.anchor_set_root
    if root:
        return Path(root)
    return Path(args.output_root) / 'anchor_sets'


def train_candidate_batch_size(args):
    return int(args.train_candidate_batch_size or args.batch_size)


def rebatched_loader(data_loader, batch_size, num_workers):
    if int(batch_size) == int(getattr(data_loader, 'batch_size', 0) or 0):
        return data_loader
    return DataLoader(
        data_loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        **dataloader_runtime_kwargs(num_workers),
    )


def train_candidate_metadata_identity(args, data_loader, checkpoint):
    train_spec = dataset_spec(args.dataset, 'id', 'train')
    imglist_path = (ROOT_DIR / 'data' / train_spec['imglist_path']).resolve()
    preprocessor = getattr(data_loader.dataset, 'preprocessor', None)
    preprocessor_identity = (
        f'{preprocessor.__class__.__module__}.{preprocessor.__class__.__name__}'
        if preprocessor is not None else 'unknown')
    helper = getattr(reference_helpers, 'train_candidate_metadata_identity',
                     None)
    if helper is not None:
        try:
            return helper(
                args.dataset,
                imglist_path,
                checkpoint,
                MODEL_ARCH[args.dataset].__name__,
                NUM_CLASSES[args.dataset],
                preprocessor_identity,
            )
        except TypeError:
            pass
    return {
        'dataset': args.dataset,
        'source': 'train',
        'train_imglist_path': str(imglist_path),
        'train_imglist_sha256': file_sha256(imglist_path),
        'checkpoint_resolved': str(Path(checkpoint).resolve()),
        'checkpoint_sha256': file_sha256(checkpoint),
        'model_arch': MODEL_ARCH[args.dataset].__name__,
        'num_classes': NUM_CLASSES[args.dataset],
        'preprocessor_identity': preprocessor_identity,
        'train_candidate_metadata_schema_version':
        TRAIN_CANDIDATE_METADATA_SCHEMA_VERSION,
    }


def train_candidate_metadata_id(identity):
    helper = getattr(reference_helpers, 'train_candidate_metadata_id', None)
    if helper is not None:
        try:
            return helper(identity)
        except TypeError:
            pass
    payload = json.dumps(identity, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def reference_set_identity(args, train_metadata, config, selected_hash,
                            classifier_name, runtime_mode):
    candidate_info = {}
    if train_metadata:
        candidate_info = {
            'candidate_id': train_metadata.get('candidate_id'),
            'identity': train_metadata.get('identity'),
        }
    return {
        'schema_version': 1,
        'dataset': args.dataset,
        'checkpoint_resolved': str(Path(resolved_checkpoint(args)).resolve()),
        'model_arch': MODEL_ARCH[args.dataset].__name__,
        'num_classes': NUM_CLASSES[args.dataset],
        'classifier_layer': classifier_name,
        'runtime_mode': runtime_mode,
        'reference_config': config.to_dict(),
        'selected_reference_hash': selected_hash,
        'train_candidate_metadata': candidate_info,
    }


def reference_set_id(identity):
    payload = json.dumps(identity, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _load_train_candidate_metadata_manifest(path):
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _train_candidate_metadata_matches(response_dir, identity):
    manifest = _load_train_candidate_metadata_manifest(response_dir / 'manifest.json')
    if not manifest:
        return False
    if manifest.get('identity') != identity:
        return False
    return (response_dir / 'candidates.npz').exists()


def normalize_train_candidate_metadata_record(cache, identity, candidate_id,
                                               reused):
    if not isinstance(cache, dict):
        cache = getattr(cache, '__dict__', {})
    cache = dict(cache)
    response_dir = cache.get('metadata_dir')
    candidate_path = cache.get('metadata_path')
    manifest_path = cache.get('manifest_path')
    normalized = dict(cache)
    if response_dir is not None:
        normalized['metadata_dir'] = Path(response_dir)
    if candidate_path is not None:
        normalized['metadata_path'] = Path(candidate_path)
    if manifest_path is not None:
        normalized['manifest_path'] = Path(manifest_path)
    normalized.setdefault('identity', identity)
    normalized.setdefault('candidate_id', cache.get('candidate_id', candidate_id))
    normalized['reused'] = bool(reused)
    return normalized


def _load_reference_set_manifest(path):
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


REFERENCE_SET_FILE = 'reference_set.npz'
ANCHOR_SET_FILE = 'anchor_set.npz'


def _reference_set_matches(response_dir, identity):
    manifest = _load_reference_set_manifest(response_dir / 'manifest.json')
    if not manifest:
        return False
    if manifest.get('identity') != identity:
        return False
    return (response_dir / REFERENCE_SET_FILE).exists()


def load_reference_set(response_dir, identity):
    if not _reference_set_matches(response_dir, identity):
        return None
    with np.load(response_dir / REFERENCE_SET_FILE, allow_pickle=False) as data:
        bank = {key: data[key] for key in data.files}
    return {
        'metadata_dir': response_dir,
        'reference_set_path': response_dir / REFERENCE_SET_FILE,
        'manifest_path': response_dir / 'manifest.json',
        'manifest': _load_reference_set_manifest(response_dir / 'manifest.json'),
        'identity': identity,
        'bank': bank,
        'reused': True,
    }


def _anchor_set_matches(response_dir, identity):
    manifest = _load_reference_set_manifest(response_dir / 'manifest.json')
    if not manifest:
        return False
    if manifest.get('identity') != identity:
        return False
    return (response_dir / ANCHOR_SET_FILE).exists()


def load_anchor_set(response_dir, identity):
    if not _anchor_set_matches(response_dir, identity):
        return None
    with np.load(response_dir / ANCHOR_SET_FILE, allow_pickle=False) as data:
        bank = {key: data[key] for key in data.files}
    return {
        'metadata_dir': response_dir,
        'anchor_set_path': response_dir / ANCHOR_SET_FILE,
        'manifest_path': response_dir / 'manifest.json',
        'manifest': _load_reference_set_manifest(response_dir / 'manifest.json'),
        'identity': identity,
        'bank': bank,
        'reused': True,
    }


def write_selected_samples_csv(path, metadata, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels.detach().cpu().numpy()
                        if isinstance(labels, torch.Tensor) else labels)
    fields = [
        'row',
        'label',
        'scan_index',
        'dataset_index',
        'image_name',
        'pred',
        'confidence',
        'entropy',
        'margin',
        'energy',
        'ce_loss',
        'correct',
    ]
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        count = len(labels)
        for row in range(count):
            item = {'row': row, 'label': int(labels[row])}
            for key in fields[2:]:
                source_key = 'dataset_index' if key == 'dataset_index' else key
                if source_key not in metadata:
                    item[key] = ''
                    continue
                value = np.asarray(metadata[source_key])[row]
                if isinstance(value, np.generic):
                    value = value.item()
                item[key] = value
            writer.writerow(item)


def save_reference_set(response_dir, identity, bank):
    response_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'features':
        bank['features'].detach().cpu().numpy(),
        'labels':
        bank['label'].detach().cpu().numpy().astype(np.int64),
        'selected_reference_hash':
        np.asarray(bank['selected_reference_hash']),
        'base_reference_loss':
        bank['base_reference_sample_diag']['loss'].detach().cpu().numpy(),
        'base_reference_confidence':
        bank['base_reference_sample_diag']
        ['confidence'].detach().cpu().numpy(),
        'base_reference_entropy':
        bank['base_reference_sample_diag']['entropy'].detach().cpu().numpy(),
        'base_reference_margin':
        bank['base_reference_sample_diag']['margin'].detach().cpu().numpy(),
        'base_reference_energy':
        bank['base_reference_sample_diag']['energy'].detach().cpu().numpy(),
        'base_reference_prediction':
        bank['base_reference_sample_diag']
        ['prediction'].detach().cpu().numpy(),
        'base_reference_correct':
        bank['base_reference_sample_diag']['correct'].detach().cpu().numpy(),
    }
    metadata = bank.get('selected_metadata') or {}
    for source, target in (
        ('scan_index', 'selected_scan_index'),
        ('dataset_index', 'selected_dataset_index'),
        ('index', 'selected_dataset_index'),
    ):
        if source in metadata and target not in payload:
            payload[target] = np.asarray(metadata[source])
    np.savez_compressed(response_dir / REFERENCE_SET_FILE, **payload)
    write_selected_samples_csv(
        response_dir / 'selected_samples.csv',
        metadata,
        bank['label'],
    )
    manifest = {
        'schema_version': 1,
        'identity': identity,
        'reference_set_id': reference_set_id(identity),
        'num_reference': int(bank['label'].numel()),
        'created_at_unix': time.time(),
        'fields': sorted(payload.keys()),
    }
    write_json(response_dir / 'manifest.json', manifest)
    return {
        'reference_set_dir': response_dir,
        'reference_set_path': response_dir / REFERENCE_SET_FILE,
        'manifest_path': response_dir / 'manifest.json',
        'manifest': manifest,
        'identity': identity,
        'reused': False,
    }


def build_train_candidate_metadata(net, data_loader, response_dir, identity, args):
    response_dir.mkdir(parents=True, exist_ok=True)
    indices = []
    labels = []
    preds = []
    confidences = []
    entropies = []
    margins = []
    energies = []
    ce_losses = []
    correct = []
    image_names = []
    scan_index = 0
    with torch.no_grad():
        iterator = tqdm(
            data_loader,
            desc='Build TARR train_candidate_metadata',
            disable=args.no_progress,
        )
        for batch in iterator:
            data = batch['data'].cuda(non_blocking=True)
            label = batch['label'].cpu()
            label_gpu = batch['label'].cuda(non_blocking=True)
            logits = net(data)
            diag = logit_diagnostics(logits)
            ce_loss = F.cross_entropy(logits, label_gpu, reduction='none')
            diag_cpu = {
                key: diag[key].detach().cpu()
                for key in ('pred', 'conf', 'entropy', 'margin', 'energy')
            }
            ce_loss_cpu = ce_loss.detach().cpu()
            batch_indices = batch.get('index')
            if batch_indices is None:
                batch_indices = torch.arange(scan_index,
                                             scan_index + label.numel())
            elif isinstance(batch_indices, torch.Tensor):
                batch_indices = batch_indices.detach().cpu()
            batch_image_names = batch.get('image_name')
            for row in range(label.numel()):
                indices.append(int(batch_indices[row]))
                labels.append(int(label[row]))
                preds.append(int(diag_cpu['pred'][row]))
                confidences.append(float(diag_cpu['conf'][row]))
                entropies.append(float(diag_cpu['entropy'][row]))
                margins.append(float(diag_cpu['margin'][row]))
                energies.append(float(diag_cpu['energy'][row]))
                ce_losses.append(float(ce_loss_cpu[row]))
                correct.append(bool(preds[-1] == labels[-1]))
                if batch_image_names is not None:
                    image_names.append(str(batch_image_names[row]))
                scan_index += 1

    np.savez_compressed(
        response_dir / 'candidates.npz',
        scan_index=np.arange(len(labels), dtype=np.int64),
        index=np.asarray(indices, dtype=np.int64),
        label=np.asarray(labels, dtype=np.int64),
        pred=np.asarray(preds, dtype=np.int64),
        confidence=np.asarray(confidences, dtype=np.float32),
        entropy=np.asarray(entropies, dtype=np.float32),
        margin=np.asarray(margins, dtype=np.float32),
        energy=np.asarray(energies, dtype=np.float32),
        ce_loss=np.asarray(ce_losses, dtype=np.float32),
        correct=np.asarray(correct, dtype=np.bool_),
        image_name=np.asarray(image_names, dtype=str),
    )
    manifest = {
        'schema_version': TRAIN_CANDIDATE_METADATA_SCHEMA_VERSION,
        'identity': identity,
        'candidate_id': train_candidate_metadata_id(identity),
        'metadata_path': str(response_dir),
        'num_candidates': len(labels),
        'created_at_unix': time.time(),
    }
    write_json(response_dir / 'manifest.json', manifest)
    return manifest


def load_or_build_train_candidate_metadata(net, data_loader, args):
    checkpoint = resolved_checkpoint(args)
    identity = train_candidate_metadata_identity(args, data_loader, checkpoint)
    metadata_id = train_candidate_metadata_id(identity)
    metadata_dir = train_candidate_metadata_root(args) / args.dataset / metadata_id

    if (args.rebuild_train_candidate_metadata
            or not _train_candidate_metadata_matches(metadata_dir, identity)):
        manifest = build_train_candidate_metadata(
            net, data_loader, metadata_dir, identity, args)
        reused = False
    else:
        manifest = _load_train_candidate_metadata_manifest(metadata_dir / 'manifest.json')
        reused = True
    return {
        'metadata_dir': metadata_dir,
        'metadata_path': metadata_dir / 'candidates.npz',
        'manifest_path': metadata_dir / 'manifest.json',
        'identity': identity,
        'candidate_id': metadata_id,
        'manifest': manifest,
        'reused': reused,
    }


def _passes_candidate_filter(candidates, idx, config):
    if config.filter == 'all':
        return True
    if config.filter == 'correct':
        return bool(candidates['correct'][idx])
    if config.filter == 'high_confidence':
        return float(candidates['confidence'][idx]) >= config.min_confidence
    if config.filter == 'correct_high_confidence':
        return (bool(candidates['correct'][idx])
                and float(candidates['confidence'][idx]) >=
                config.min_confidence)
    raise ValueError(f'Unknown reference filter: {config.filter}')


def select_train_candidate_indices(candidates, config, num_classes):
    if config.filter == 'correct_confidence_stratified':
        selected = []
        labels = candidates['label']
        sample_indices = candidates.get('index', candidates.get('dataset_index'))
        confidence = candidates['confidence']
        correct = candidates['correct']
        for class_id in range(num_classes):
            eligible = np.where((labels == class_id) & correct)[0]
            if eligible.shape[0] < config.per_class:
                selected.append([])
                continue
            order = eligible[np.argsort(confidence[eligible], kind='mergesort')]
            rng = random.Random(
                f'{config.seed}:{class_id}:correct_confidence_stratified')
            if config.per_class == 1:
                strata = [order[order.shape[0] // 3:
                                max(order.shape[0] // 3,
                                    2 * order.shape[0] // 3)]]
                if len(strata[0]) == 0:
                    strata = [order]
                allocation = [1]
            elif config.per_class == 2:
                strata = np.array_split(order, 3)
                allocation = [0, 1, 1]
            else:
                strata = np.array_split(order, 3)
                base = config.per_class // 3
                allocation = [base, base, base]
                for stratum_id in [1, 2, 0]:
                    if sum(allocation) >= config.per_class:
                        break
                    allocation[stratum_id] += 1
            class_selected = []
            for stratum, quota in zip(strata, allocation):
                if quota <= 0:
                    continue
                pool = [int(sample_indices[idx]) for idx in stratum]
                if len(pool) < quota:
                    class_selected.extend(pool)
                else:
                    class_selected.extend(rng.sample(pool, quota))
            if len(class_selected) < config.per_class:
                selected_set = set(class_selected)
                remaining = [
                    int(sample_indices[idx]) for idx in order
                    if int(sample_indices[idx]) not in selected_set
                ]
                class_selected.extend(
                    rng.sample(remaining,
                               config.per_class - len(class_selected)))
            selected.append(class_selected)
        missing = [
            str(class_id) for class_id, samples in enumerate(selected)
            if len(samples) < config.per_class
        ]
        if missing:
            raise RuntimeError(
                'Not enough reference samples for classes: ' + ', '.join(missing))
        return selected

    rng = random.Random(config.seed)
    seen = [0] * num_classes
    selected = [[] for _ in range(num_classes)]
    k = config.per_class
    labels = candidates['label']
    sample_indices = candidates.get('index', candidates.get('dataset_index'))
    for idx in range(labels.shape[0]):
        class_id = int(labels[idx])
        if class_id < 0 or class_id >= num_classes:
            continue
        if not _passes_candidate_filter(candidates, idx, config):
            continue
        seen[class_id] += 1
        item = int(sample_indices[idx])
        if len(selected[class_id]) < k:
            selected[class_id].append(item)
        else:
            replace_idx = rng.randrange(seen[class_id])
            if replace_idx < k:
                selected[class_id][replace_idx] = item

    missing = [
        str(class_id) for class_id, samples in enumerate(selected)
        if len(samples) < k
    ]
    if missing:
        raise RuntimeError(
            'Not enough reference samples for classes: ' + ', '.join(missing))
    return selected


def load_reference_samples_by_index(data_loader, selected_by_class):
    dataset = data_loader.dataset
    ref_data = []
    ref_label = []
    for class_id, sample_indices in enumerate(selected_by_class):
        for sample_index in sample_indices:
            sample = dataset[int(sample_index)]
            ref_data.append(sample['data'].cpu())
            ref_label.append(class_id)
    return torch.stack(ref_data), torch.tensor(ref_label, dtype=torch.long)


def select_reference_from_train_candidate_metadata(train_metadata, data_loader, config,
                                          num_classes):
    def normalize_selection(result):
        if result is None:
            return None
        if hasattr(result, 'data') and hasattr(result, 'label'):
            return (
                result.data,
                result.label,
                getattr(result, 'selected_metadata', None),
            )
        if isinstance(result, tuple):
            if len(result) >= 3:
                return result[0], result[1], result[2]
            if len(result) >= 2:
                return result[0], result[1], None
        return None

    tensor_helper = getattr(reference_helpers,
                            'select_reference_tensors_from_metadata', None)
    if tensor_helper is not None:
        try:
            candidates = (train_metadata.get('candidates')
                          if isinstance(train_metadata, dict) else None)
            if candidates is not None:
                result = tensor_helper(data_loader, candidates, config,
                                       num_classes)
                normalized = normalize_selection(result)
                if normalized is None:
                    raise TypeError('unsupported reference selection result')
                if normalized[2] is not None:
                    return normalized
                selected_rows = reference_helpers.select_train_candidate_indices(
                    candidates, config, num_classes)[0]
                metadata = selected_reference_metadata(candidates, selected_rows)
                return normalized[0], normalized[1], metadata
        except TypeError:
            pass
    helper = getattr(reference_helpers, 'select_reference_from_train_candidate_metadata',
                     None)
    if helper is not None:
        try:
            result = helper(
                train_metadata=train_metadata,
                data_loader=data_loader,
                config=config,
                num_classes=num_classes,
            )
            normalized = normalize_selection(result)
            if normalized is None:
                raise TypeError('unsupported reference selection result')
            return normalized
        except TypeError:
            pass

    if not isinstance(train_metadata, dict):
        train_metadata = getattr(train_metadata, '__dict__', {})
    candidate_path = train_metadata.get('metadata_path')
    if candidate_path is None:
        raise ValueError('train_candidate_metadata record is missing metadata_path')
    with np.load(candidate_path, allow_pickle=False) as data:
        candidates = {key: data[key] for key in data.files}
    selected = select_train_candidate_indices(candidates, config,
                                                  num_classes)
    selected_rows = selected_reference_rows(candidates, selected)
    reference_data, reference_label = load_reference_samples_by_index(
        data_loader, selected)
    return reference_data, reference_label, selected_reference_metadata(
        candidates, selected_rows)


def selected_reference_rows(candidates, selected_by_class):
    sample_indices = candidates.get('index', candidates.get('dataset_index'))
    if sample_indices is None:
        return None
    row_by_sample = {int(sample): row for row, sample in enumerate(sample_indices)}
    rows = []
    for sample_indices_for_class in selected_by_class:
        for sample_index in sample_indices_for_class:
            row = row_by_sample.get(int(sample_index))
            if row is None:
                return None
            rows.append(row)
    return np.asarray(rows, dtype=np.int64)


def selected_reference_metadata(candidates, selected_rows):
    if selected_rows is None:
        return None
    metadata = {}
    for key in [
            'scan_index',
            'index',
            'dataset_index',
            'label',
            'pred',
            'confidence',
            'entropy',
            'margin',
            'energy',
            'ce_loss',
            'correct',
            'image_name',
    ]:
        if key in candidates:
            metadata[key] = np.asarray(candidates[key])[selected_rows]
    return metadata


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def train_candidate_metadata_info(postprocessor):
    cache = getattr(postprocessor, 'train_candidate_metadata', None)
    if not cache:
        return {}
    if not isinstance(cache, dict):
        cache = getattr(cache, '__dict__', {})
    info = {}
    for key in ['metadata_dir', 'metadata_path', 'manifest_path']:
        if key in cache and cache[key] is not None:
            info[key] = str(cache[key])
    for key in ['candidate_id', 'identity', 'manifest', 'reused']:
        if key in cache:
            info[key] = json_safe(cache[key])
    return info


def reference_set_info(postprocessor):
    return json_safe(getattr(postprocessor, 'reference_set_records', {}))


def anchor_set_info(postprocessor):
    return json_safe(getattr(postprocessor, 'anchor_set_records', {}))


def timing_info(postprocessor):
    timing = dict(getattr(postprocessor, 'timing', {}))
    processed = int(timing.get('processed_count') or 0)
    runtime_sum = float(timing.get('runtime_per_target_sum_sec') or 0.0)
    timing['processed_count'] = processed
    timing['runtime_per_target_mean_sec'] = (
        runtime_sum / processed if processed else None)
    return json_safe(timing)


def runtime_benchmark_rows(args, run_id, postprocessor):
    timing = timing_info(postprocessor)
    train_metadata = train_candidate_metadata_info(postprocessor)
    candidate_manifest = train_metadata.get('manifest') or {}
    candidate_identity = train_metadata.get('identity') or {}
    candidate_schema = (
        candidate_manifest.get('schema_version')
        or candidate_identity.get('schema_version')
        or candidate_identity.get('train_candidate_metadata_schema_version'))
    reference_set_records = reference_set_info(postprocessor)
    processed = int(timing.get('processed_count') or 0)
    inference_sec = float(timing.get('inference_total_sec') or 0.0)
    cuda_name = 'unavailable'
    if torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(torch.cuda.current_device())
    rows = []
    for config in postprocessor.reference_configs:
        set_info = reference_set_records.get(config.id, {}) if reference_set_records else {}
        rows.append({
            'run_id':
            run_id,
            'dataset':
            args.dataset,
            'baseline_protocol':
            args.baseline_protocol,
            'scheme':
            args.scheme,
            'reference_config_id':
            config.id,
            'runtime_mode':
            postprocessor.runtime_mode,
            'train_candidate_metadata_schema_version':
            candidate_schema,
            'train_candidate_metadata_reused':
            train_metadata.get('reused'),
            'train_candidate_metadata_sec':
            timing.get('train_candidate_metadata_sec'),
            'reference_set_reused':
            set_info.get('reused'),
            'reference_set_sec':
            set_info.get('elapsed_sec', timing.get('reference_set_sec')),
            'setup_total_sec':
            timing.get('setup_total_sec'),
            'inference_total_sec':
            inference_sec,
            'scoring_total_sec':
            '',
            'processed_targets':
            processed,
            'runtime_per_target_sec':
            inference_sec / processed if processed else '',
            'batch_size':
            args.batch_size,
            'train_candidate_batch_size':
            train_candidate_batch_size(args),
            'reference_set_batch_size':
            args.reference_set_batch_size,
            'num_workers':
            args.num_workers,
            'cuda_device':
            cuda_name,
        })
    return rows


def format_metric_row(dataset_name, metrics):
    fpr, auroc, aupr_in, aupr_out, acc = metrics
    return {
        'dataset': dataset_name,
        'FPR@95': f'{100 * fpr:.2f}',
        'AUROC': f'{100 * auroc:.2f}',
        'AUPR_IN': f'{100 * aupr_in:.2f}',
        'AUPR_OUT': f'{100 * aupr_out:.2f}',
        'ACC': f'{100 * acc:.2f}',
    }


def limit_for_split(args, split):
    if split in {'id', 'csid'}:
        return args.max_id_samples or args.max_samples
    return args.max_ood_samples or args.max_samples


def target_shard_config_dict(args):
    return {
        'count': int(args.target_shard_count),
        'index': int(args.target_shard_index),
        'rule': 'global_sample_index % count == index',
    }


def protocol_dataset_items(dataset, split, dataset_map, choice):
    if split == 'near':
        expected_names = near_dataset_names(dataset)
    elif split == 'far':
        expected_names = far_dataset_names(dataset)
    else:
        raise ValueError(f'Unknown protocol OOD split: {split}')
    if choice != 'all':
        expected_names = [item.strip() for item in choice.split(',')
                          if item.strip()]
    missing = [name for name in expected_names if name not in dataset_map]
    if missing:
        raise ValueError('Unknown OOD dataset(s): ' + ', '.join(missing))
    return [(name, dataset_map[name]) for name in expected_names]


def dataset_names_for_scheme(args, evaluator, scheme):
    near_names = [name for name, _ in protocol_dataset_items(
        args.dataset, 'near', evaluator.dataloader_dict['ood']['near'],
        args.near_datasets)]
    far_names = [name for name, _ in protocol_dataset_items(
        args.dataset, 'far', evaluator.dataloader_dict['ood']['far'],
        args.far_datasets)]
    csid_names = []
    if scheme == 'fsood':
        csid_names = list(evaluator.dataloader_dict['csid'].keys())
    return {
        'id': [args.dataset],
        'csid': csid_names,
        'near': near_names,
        'far': far_names,
    }


def dataset_manifest(args, data_root, names):
    payload = {'id': {}, 'csid': {}, 'ood': {'near': {}, 'far': {}}}
    payload['id'][args.dataset] = {
        'test': manifest_path(data_root, dataset_spec(args.dataset, 'id', 'test')),
        'train': manifest_path(data_root, dataset_spec(args.dataset, 'id', 'train')),
    }
    for name in names.get('csid', []):
        payload['csid'][name] = manifest_path(
            data_root, dataset_spec(args.dataset, 'csid', name))
    for split in ['near', 'far']:
        for name in names.get(split, []):
            payload['ood'][split][name] = manifest_path(
                data_root, dataset_spec(args.dataset, split, name))
    return payload


def default_run_id(args):
    ref_group = 'refs' + str(len(parse_reference_configs(args)))
    return (
        f'{args.dataset}_{args.baseline_protocol}_train'
        f'_{tta_config_id(args, resolve_runtime_mode(args))}_{ref_group}'
        f'_{perturbation_config_id(args)}_{args.score_rule}_seed{args.seed}'
    )


def resolve_protocol_csid(dataset, baseline_protocol):
    return expected_csid_datasets(dataset, baseline_protocol)


def dataset_spec(dataset, split, name=None):
    info = DATA_INFO[dataset]
    if split == 'id':
        return info['id'][name]
    if split == 'csid':
        return info['csid'][name]
    if split in {'near', 'far'}:
        return info['ood'][split][name]
    if split == 'val':
        return info['ood']['val']
    raise ValueError(f'Unknown split: {split}')


def manifest_path(data_root, spec):
    imglist_path = (Path(data_root) / spec['imglist_path']).resolve()
    return {
        'data_dir': str((Path(data_root) / spec['data_dir']).resolve()),
        'imglist_path': str(imglist_path),
        'imglist_sha256': file_sha256(imglist_path),
    }


def build_imglist_loader(dataset, data_root, split, name, batch_size, shuffle,
                         num_workers):
    spec = dataset_spec(dataset, split, name)
    preprocessor = get_default_preprocessor(dataset)
    standard_preprocessor = get_default_preprocessor(dataset)
    ds = ImglistDataset(
        name='_'.join((dataset, split, name)),
        imglist_pth=str(Path(data_root) / spec['imglist_path']),
        data_dir=str(Path(data_root) / spec['data_dir']),
        num_classes=NUM_CLASSES[dataset],
        preprocessor=preprocessor,
        data_aux_preprocessor=standard_preprocessor,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **dataloader_runtime_kwargs(num_workers),
    )


def make_runtime_train_loader(dataset, batch_size, num_workers):
    return build_imglist_loader(
        dataset,
        ROOT_DIR / 'data',
        'id',
        'train',
        int(batch_size),
        False,
        int(num_workers),
    )


def apply_protocol_dataloaders(args, evaluator, data_root):
    csid_names = resolve_protocol_csid(args.dataset, args.baseline_protocol)
    if not uses_evaluator_csid_loaders(args.dataset):
        evaluator.dataloader_dict['csid'] = {
            name: build_imglist_loader(args.dataset, data_root, 'csid', name,
                                       args.batch_size, False, args.num_workers)
            for name in csid_names
        }
    return csid_names


def resolve_runtime_mode(args):
    return resolve_runtime_mode_value(args.runtime_mode, args.update_scope)


def score_tuple(scores, score_rule, label_override=None):
    label = scores['label'] if label_override is None else label_override
    return scores['pred'], -scores['ood_score'][score_rule], label


def concat_score_parts(parts, score_rule):
    return (
        np.concatenate([part['pred'] for part in parts]),
        -np.concatenate([part['ood_score'][score_rule] for part in parts]),
        np.concatenate([part['label'] for part in parts]),
    )


def reference_dir(scheme_dir, reference_config_id):
    return scheme_dir / 'references' / reference_config_id


def output_score_dir(reference_dir_path, args, score_rule):
    return reference_dir_path / 'score_results' / score_rule / 'scores'


def output_metrics_path(reference_dir_path, args, score_rule):
    return reference_dir_path / 'score_results' / score_rule / 'ood.csv'


def output_debug_path(reference_dir_path, args):
    if args.score_rule in {'all', 'probe_all'}:
        return reference_dir_path / 'score_results' / 'debug_samples_all_scores.csv'
    return reference_dir_path / 'score_results' / 'debug_samples.csv'


class TARRPostprocessor(BasePostprocessor):
    def __init__(self, args, num_classes):
        super().__init__(config=None)
        self.args = args
        self.num_classes = num_classes
        self.APS_mode = False
        self.hyperparam_search_done = True

        self.reference_configs = parse_reference_configs(args)
        self.reference_sets = {}
        self.reference_set_records = {}
        self.anchor_sets = {}
        self.anchor_set_records = {}
        self.base_state = None
        self.fc_state = None
        self.base_restore_pairs = None
        self.fc_restore_pairs = None
        self.classifier = None
        self.classifier_name = None
        self.runtime_mode = resolve_runtime_mode(args)
        self.tta_update_impl = resolve_tta_update_impl(args.update_scope)
        self.reference_feature_cache_enabled = self.runtime_mode in {
            'classifier_feature_cache',
        }
        self.target_feature_cache_enabled = (
            self.runtime_mode == 'classifier_feature_cache')
        self.trainable_params = None
        self.optimizer = None
        self.reference_stats = {}
        self.anchor_stats = {}
        self.train_candidate_metadata = None
        self.timing = {
            'train_candidate_metadata_sec': 0.0,
            'reference_set_sec': 0.0,
            'reference_set_build_sec': 0.0,
            'reference_set_reuse_sec': 0.0,
            'anchor_set_sec': 0.0,
            'anchor_set_reuse_sec': 0.0,
            'setup_total_sec': 0.0,
            'inference_total_sec': 0.0,
            'processed_count': 0,
            'seen_count': 0,
            'skipped_by_shard_count': 0,
            'target_shard_count': int(args.target_shard_count),
            'target_shard_index': int(args.target_shard_index),
            'runtime_per_target_sum_sec': 0.0,
            'runtime_per_target_min_sec': None,
            'runtime_per_target_max_sec': None,
            'inference_calls': [],
        }
        self.sample_debug = []
        self.debug_output_mode = args.debug_output_mode
        self.response_steps = list(args.response_steps)
        self.perturbation_generators = {}
        self.score_rules = selected_score_rules(args.score_rule)
        self.skip_view_clean_logits = (
            args.freeze_bn_stats and args.objective in VIEW_PERTURBATION_OBJECTIVES
        )
        if args.objective == 'memo_marginal_entropy':
            if args.perturbation_response not in {'pixel', 'feature'}:
                raise ValueError(
                    'memo_marginal_entropy requires pixel or feature '
                    'perturbation response.')
            if args.perturbation_kind != 'gaussian':
                raise ValueError(
                    'memo_marginal_entropy currently supports only gaussian '
                    'perturbation.')
        if args.objective in SOFT_VIEW_OBJECTIVES:
            if args.perturbation_response not in {'pixel', 'feature'}:
                raise ValueError(
                    f'{args.objective} requires pixel or feature perturbation '
                    'response.')
            if args.perturbation_kind != 'gaussian':
                raise ValueError(
                    f'{args.objective} currently supports only gaussian '
                    'perturbation.')
            if args.perturbation_eps <= 0:
                raise ValueError(f'{args.objective} requires perturbation eps > 0.')
            if args.perturbation_repeats < 2:
                raise ValueError(
                    f'{args.objective} requires at least two perturbation '
                    'repeats.')
        if (args.perturbation_response == 'pixel'
                and args.perturbation_cache_policy == 'error_on_feature_cache'
                and self.runtime_mode == 'classifier_feature_cache'):
            raise ValueError(
                'Pixel perturbation response requires a full forward path, but '
                'classifier_feature_cache runtime is active and '
                '--perturbation-cache-policy error_on_feature_cache was set. '
                'Use --perturbation-cache-policy auto or a non-cache '
                '--runtime-mode.')

    def setup(self, net, id_loader_dict, ood_loader_dict):
        setup_start = time.perf_counter()
        del ood_loader_dict
        net.eval()
        self.classifier = classifier_layer(net)
        self.classifier_name = classifier_layer_name(net)
        self.base_state = {
            key: value.detach().clone()
            for key, value in net.state_dict().items()
        }
        self.fc_state = {
            key: value.detach().clone()
            for key, value in self.classifier.state_dict().items()
        }
        self.base_restore_pairs = self._state_restore_pairs(net, self.base_state)
        self.fc_restore_pairs = self._state_restore_pairs(
            self.classifier, self.fc_state)
        self.reference_sets = {}
        self.reference_set_records = {}
        self.anchor_sets = {}
        self.anchor_set_records = {}
        self.train_candidate_metadata = None
        if self.args.use_anchor_reference and not self.args.use_prebuilt_reference_set:
            raise ValueError(
                '--use-anchor-reference requires --use-prebuilt-reference-set '
                'so Probe P and Anchor A artifacts are explicit and disjoint.')
        if self.args.use_prebuilt_reference_set:
            if self.runtime_mode != 'classifier_feature_cache':
                raise ValueError(
                    '--use-prebuilt-reference-set currently requires '
                    'classifier_feature_cache runtime because canonical '
                    'reference_set artifacts store classifier features, not '
                    'reference images.')
            expected_candidate_identity = train_candidate_metadata_identity(
                self.args,
                id_loader_dict['train'],
                resolved_checkpoint(self.args),
            )
            for config in self.reference_configs:
                bank_start = time.perf_counter()
                bank, record = self._load_prebuilt_reference_set(config)
                self._validate_prebuilt_reference_set_identity(
                    config,
                    record,
                    expected_candidate_identity,
                )
                self._register_reference_bank(config, bank, True, record,
                                              bank_start)
                if self.args.use_anchor_reference:
                    anchor_start = time.perf_counter()
                    anchor_bank, anchor_record = self._load_prebuilt_anchor_set(
                        config, record)
                    self._validate_prebuilt_anchor_set_identity(
                        config,
                        record,
                        anchor_record,
                    )
                    self._register_anchor_bank(config, anchor_bank,
                                               anchor_record, anchor_start)
            if self.args.update_scope == 'classifier':
                self.trainable_params = self._set_trainable_parameters(net)
                if self.tta_update_impl == 'reused_torch_optimizer':
                    self.optimizer = torch.optim.SGD(self.trainable_params,
                                                     lr=self.args.lr)
            self.reference_stats = {
                config_id: bank['stats']
                for config_id, bank in self.reference_sets.items()
            }
            self.anchor_stats = {
                config_id: bank['stats']
                for config_id, bank in self.anchor_sets.items()
            }
            self.timing['setup_total_sec'] += time.perf_counter() - setup_start
            return
        candidate_start = time.perf_counter()
        candidate_loader = rebatched_loader(
            id_loader_dict['train'],
            train_candidate_batch_size(self.args),
            self.args.num_workers,
        )
        self.train_candidate_metadata = load_or_build_train_candidate_metadata(
            net, candidate_loader, self.args)
        self.timing['train_candidate_metadata_sec'] += (
            time.perf_counter() - candidate_start)
        for config in self.reference_configs:
            bank_start = time.perf_counter()
            reference_data, reference_label, reference_metadata = self._build_reference(
                net, id_loader_dict['train'], config)
            reference_data = reference_data.cuda(non_blocking=True)
            reference_label = reference_label.cuda(non_blocking=True)
            bank = {
                'config': config,
                'data': reference_data,
                'label': reference_label,
                'class_counts': torch.bincount(
                    reference_label,
                    minlength=self.num_classes,
                ).clamp_min(1),
                'class_indices': [
                    torch.where(reference_label == class_id)[0]
                    for class_id in range(self.num_classes)
                ],
                'features': None,
                'selected_metadata': reference_metadata,
                'selected_reference_hash': selected_reference_hash(
                    reference_label, reference_data),
            }
            reference_set_record = None
            reference_set_reused = False
            if self.reference_feature_cache_enabled:
                reference_set_reused = self._try_load_reference_set(bank)
                if not reference_set_reused:
                    bank['features'] = self._reference_feature_cache(net, bank)
            if reference_set_reused:
                base_diag = bank['base_reference_sample_diag']
            else:
                metadata_diag = self._reference_sample_diagnostics_from_metadata(
                    reference_metadata, reference_label.device)
                if metadata_diag is not None:
                    base_diag = metadata_diag
                else:
                    base_diag = self._reference_sample_diagnostics(net, bank)
                bank['base_reference_sample_diag'] = base_diag
                if self.reference_feature_cache_enabled:
                    reference_set_record = self._save_reference_set(bank)
            base_class_diag = self._reference_classwise_diagnostics(
                base_diag, bank)
            base_loss = base_class_diag['loss']
            bank['base_reference_loss'] = base_loss
            bank['base_reference_loss_cpu'] = base_loss.detach().cpu().numpy()
            bank['base_reference_diag'] = base_class_diag
            bank['base_reference_pred'] = base_diag['prediction'].detach()
            bank['base_reference_loss_diag'] = vector_diagnostics(base_loss)
            bank['base_reference_correct_rate_cpu'] = (
                base_class_diag['correct_rate'].detach().cpu().numpy())
            bank['base_reference_correct_rate_list'] = (
                base_class_diag['correct_rate'].detach().cpu().tolist())
            bank['stats'] = {
                'num_reference': int(reference_label.numel()),
                'per_class_counts': [
                    int((reference_label == i).sum().item())
                    for i in range(self.num_classes)
                ],
                'selected_reference_hash': bank['selected_reference_hash'],
                'base_reference_loss': tensor_stats(base_diag['loss']),
                'base_reference_loss_by_class': classwise_tensor_stats(
                    base_diag['loss'], reference_label, self.num_classes),
                'reference_confidence': tensor_stats(base_diag['confidence']),
                'reference_confidence_by_class': classwise_tensor_stats(
                    base_diag['confidence'], reference_label, self.num_classes),
                'reference_entropy_by_class': classwise_tensor_stats(
                    base_diag['entropy'], reference_label, self.num_classes),
                'reference_margin_by_class': classwise_tensor_stats(
                    base_diag['margin'], reference_label, self.num_classes),
                'reference_energy_by_class': classwise_tensor_stats(
                    base_diag['energy'], reference_label, self.num_classes),
                'reference_correct_by_class': classwise_tensor_stats(
                    base_diag['correct'].float(), reference_label,
                    self.num_classes),
            }
            if reference_set_reused:
                bank['stats']['reference_set_reused'] = True
            elif reference_set_record is not None:
                bank['stats']['reference_set_reused'] = False
            if self.reference_feature_cache_enabled:
                bank['data'] = None
            self._register_reference_bank(config, bank, reference_set_reused,
                                          reference_set_record, bank_start)
        if self.args.update_scope == 'classifier':
            self.trainable_params = self._set_trainable_parameters(net)
            if self.tta_update_impl == 'reused_torch_optimizer':
                self.optimizer = torch.optim.SGD(self.trainable_params,
                                                 lr=self.args.lr)
        self.reference_stats = {
            config_id: bank['stats']
            for config_id, bank in self.reference_sets.items()
        }
        self.timing['setup_total_sec'] += time.perf_counter() - setup_start

    def _register_reference_bank(self, config, bank, reference_set_reused,
                                 reference_set_record, bank_start):
        self.reference_sets[config.id] = bank
        bank_elapsed = time.perf_counter() - bank_start
        set_info = self.reference_set_records.setdefault(config.id, {})
        if reference_set_record:
            set_info.update(json_safe(reference_set_record))
        set_info['elapsed_sec'] = bank_elapsed
        set_info['reused'] = bool(reference_set_reused)
        self.timing['reference_set_sec'] += bank_elapsed
        if reference_set_reused:
            self.timing['reference_set_reuse_sec'] += bank_elapsed
        else:
            self.timing['reference_set_build_sec'] += bank_elapsed

    def _register_anchor_bank(self, config, bank, anchor_set_record, bank_start):
        self.anchor_sets[config.id] = bank
        bank_elapsed = time.perf_counter() - bank_start
        set_info = self.anchor_set_records.setdefault(config.id, {})
        if anchor_set_record:
            set_info.update(json_safe(anchor_set_record))
        set_info['elapsed_sec'] = bank_elapsed
        set_info['reused'] = True
        self.timing['anchor_set_sec'] += bank_elapsed
        self.timing['anchor_set_reuse_sec'] += bank_elapsed

    def _find_prebuilt_reference_set_dir(self, config):
        root = (reference_set_root(self.args) / self.args.dataset / config.id /
                f'seed{config.seed}')
        if not root.exists():
            raise FileNotFoundError(
                f'reference_set directory not found for {config.id}: {root}')
        matches = []
        for manifest_path in sorted(root.glob('*/manifest.json')):
            try:
                with manifest_path.open() as f:
                    manifest = json.load(f)
            except json.JSONDecodeError:
                continue
            identity = manifest.get('identity') or {}
            ref_config = identity.get('reference_config') or manifest.get(
                'reference_config') or {}
            if ref_config.get('id') != config.id:
                continue
            if int(ref_config.get('per_class', config.per_class)) != int(
                    config.per_class):
                continue
            if ref_config.get('filter', config.filter) != config.filter:
                continue
            if int(ref_config.get('seed', config.seed)) != int(config.seed):
                continue
            matches.append((manifest_path.parent, manifest))
        if not matches:
            raise FileNotFoundError(
                f'No matching prebuilt reference_set found for {config.id} '
                f'under {root}. Build it with reference.py build-reference-set.')
        if len(matches) > 1:
            candidates = '\n'.join(str(path) for path, _ in matches)
            raise RuntimeError(
                f'Ambiguous reference_set for {config.id}; pass a unique '
                f'config/seed or remove stale artifacts:\n{candidates}')
        return matches[0]

    def _find_prebuilt_anchor_set_dir(self, config, probe_reference_set_id):
        root = (anchor_set_root(self.args) / self.args.dataset / config.id /
                f'seed{config.seed}')
        if not root.exists():
            raise FileNotFoundError(
                f'anchor_set directory not found for {config.id}: {root}')
        matches = []
        for manifest_path in sorted(root.glob('*/manifest.json')):
            try:
                with manifest_path.open() as f:
                    manifest = json.load(f)
            except json.JSONDecodeError:
                continue
            identity = manifest.get('identity') or {}
            ref_config = identity.get('reference_config') or manifest.get(
                'reference_config') or {}
            if ref_config.get('id') != config.id:
                continue
            if int(ref_config.get('per_class', config.per_class)) != int(
                    config.per_class):
                continue
            if ref_config.get('filter', config.filter) != config.filter:
                continue
            if int(ref_config.get('seed', config.seed)) != int(config.seed):
                continue
            if manifest.get('probe_reference_set_id') != probe_reference_set_id:
                continue
            if not manifest.get('anchor_probe_disjoint', False):
                continue
            matches.append((manifest_path.parent, manifest))
        if not matches:
            raise FileNotFoundError(
                f'No matching prebuilt anchor_set found for {config.id} '
                f'and probe_reference_set_id={probe_reference_set_id} under '
                f'{root}. Build it with reference.py build-anchor-set.')
        if len(matches) > 1:
            candidates = '\n'.join(str(path) for path, _ in matches)
            raise RuntimeError(
                f'Ambiguous anchor_set for {config.id}; remove stale artifacts:\n'
                f'{candidates}')
        return matches[0]

    def _metadata_record_from_prebuilt_reference_set(
            self, reference_identity, expected_candidate_identity):
        candidate_info = reference_identity.get('train_candidate_metadata') or {}
        candidate_identity = candidate_info.get('identity')
        candidate_id = candidate_info.get('candidate_id')
        expected_candidate_id = train_candidate_metadata_id(
            expected_candidate_identity)
        if candidate_identity != expected_candidate_identity:
            raise ValueError(
                'prebuilt reference_set was built from a different '
                'train_candidate_metadata identity than the current run.')
        if candidate_id != expected_candidate_id:
            raise ValueError(
                'prebuilt reference_set train_candidate_metadata id mismatch: '
                f'expected {expected_candidate_id}, got {candidate_id}')
        metadata_dir = (train_candidate_metadata_root(self.args) /
                        self.args.dataset / expected_candidate_id)
        manifest_path = metadata_dir / 'manifest.json'
        metadata_path = metadata_dir / 'candidates.npz'
        if not manifest_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                'prebuilt reference_set points to missing '
                f'train_candidate_metadata artifact: {metadata_dir}')
        manifest = _load_train_candidate_metadata_manifest(manifest_path)
        if manifest is None or manifest.get('identity') != expected_candidate_identity:
            raise ValueError(
                'train_candidate_metadata manifest does not match the current '
                f'run identity: {manifest_path}')
        return {
            'metadata_dir': metadata_dir,
            'metadata_path': metadata_path,
            'manifest_path': manifest_path,
            'identity': expected_candidate_identity,
            'candidate_id': expected_candidate_id,
            'manifest': manifest,
            'reused': True,
            'source': 'prebuilt_reference_set',
        }

    def _validate_prebuilt_reference_set_identity(
            self, config, record, expected_candidate_identity):
        identity = record.get('identity') or {}
        if identity.get('dataset') != self.args.dataset:
            raise ValueError(
                f'prebuilt reference_set dataset mismatch for {config.id}: '
                f'{identity.get("dataset")} != {self.args.dataset}')
        expected_checkpoint = str(Path(resolved_checkpoint(self.args)).resolve())
        if identity.get('checkpoint_resolved') != expected_checkpoint:
            raise ValueError(
                f'prebuilt reference_set checkpoint mismatch for {config.id}: '
                f'{identity.get("checkpoint_resolved")} != {expected_checkpoint}')
        if identity.get('model_arch') != MODEL_ARCH[self.args.dataset].__name__:
            raise ValueError(
                f'prebuilt reference_set model_arch mismatch for {config.id}.')
        if int(identity.get('num_classes', -1)) != int(self.num_classes):
            raise ValueError(
                f'prebuilt reference_set num_classes mismatch for {config.id}.')
        if identity.get('classifier_layer') != self.classifier_name:
            raise ValueError(
                f'prebuilt reference_set classifier layer mismatch for '
                f'{config.id}: {identity.get("classifier_layer")} != '
                f'{self.classifier_name}')
        if identity.get('runtime_mode') != self.runtime_mode:
            raise ValueError(
                f'prebuilt reference_set runtime_mode mismatch for {config.id}: '
                f'{identity.get("runtime_mode")} != {self.runtime_mode}')
        metadata_record = self._metadata_record_from_prebuilt_reference_set(
            identity,
            expected_candidate_identity,
        )
        if self.train_candidate_metadata is None:
            self.train_candidate_metadata = metadata_record
        elif (self.train_candidate_metadata.get('candidate_id') !=
              metadata_record.get('candidate_id')):
            raise ValueError(
                'prebuilt reference_sets were built from multiple '
                'train_candidate_metadata artifacts in one run.')

    def _validate_prebuilt_anchor_set_identity(
            self, config, probe_record, anchor_record):
        probe_identity = probe_record.get('identity') or {}
        anchor_identity = anchor_record.get('identity') or {}
        anchor_manifest = anchor_record.get('manifest') or {}
        if anchor_identity.get('artifact_type') != 'anchor_set':
            raise ValueError(f'prebuilt anchor_set artifact_type mismatch for {config.id}.')
        if (anchor_identity.get('train_candidate_metadata') !=
                probe_identity.get('train_candidate_metadata')):
            raise ValueError(
                f'anchor_set train_candidate_metadata does not match Probe P '
                f'for {config.id}.')
        if anchor_manifest.get('probe_reference_set_id') != probe_record.get(
                'reference_set_id'):
            raise ValueError(
                f'anchor_set probe_reference_set_id mismatch for {config.id}.')
        if not bool(anchor_manifest.get('anchor_probe_disjoint', False)):
            raise ValueError(f'anchor_set is not marked disjoint for {config.id}.')

    def _load_prebuilt_reference_set(self, config):
        response_dir, manifest = self._find_prebuilt_reference_set_dir(config)
        identity = manifest.get('identity')
        if identity is None:
            raise ValueError(f'reference_set manifest missing identity: {response_dir}')
        cache = load_reference_set(response_dir, identity)
        if cache is None:
            raise ValueError(f'invalid reference_set artifact: {response_dir}')
        cached = cache['bank']
        required = [
            'features',
            'labels',
            'base_reference_loss',
            'base_reference_confidence',
            'base_reference_entropy',
            'base_reference_margin',
            'base_reference_energy',
            'base_reference_prediction',
            'base_reference_correct',
        ]
        missing = [key for key in required if key not in cached]
        if missing:
            raise ValueError(
                f'reference_set artifact is missing required field(s): '
                f'{", ".join(missing)} ({response_dir})')
        device = torch.device('cuda')
        labels = torch.as_tensor(cached['labels'], device=device, dtype=torch.long)
        bank = {
            'config': config,
            'data': None,
            'label': labels,
            'class_counts': torch.bincount(
                labels,
                minlength=self.num_classes,
            ).clamp_min(1),
            'class_indices': [
                torch.where(labels == class_id)[0]
                for class_id in range(self.num_classes)
            ],
            'features': torch.as_tensor(cached['features'], device=device),
            'selected_metadata': {},
            'selected_reference_hash': str(np.asarray(
                cached.get('selected_reference_hash', '')).item()),
            'base_reference_sample_diag': {
                'loss': torch.as_tensor(
                    cached['base_reference_loss'], device=device),
                'confidence': torch.as_tensor(
                    cached['base_reference_confidence'], device=device),
                'entropy': torch.as_tensor(
                    cached['base_reference_entropy'], device=device),
                'margin': torch.as_tensor(
                    cached['base_reference_margin'], device=device),
                'energy': torch.as_tensor(
                    cached['base_reference_energy'], device=device),
                'prediction': torch.as_tensor(
                    cached['base_reference_prediction'],
                    device=device,
                    dtype=torch.long,
                ),
                'correct': torch.as_tensor(
                    cached['base_reference_correct'],
                    device=device,
                    dtype=torch.bool,
                ),
            },
        }
        for key in cached:
            if key.startswith('selected_'):
                bank['selected_metadata'][key[len('selected_'):]] = cached[key]
        base_diag = bank['base_reference_sample_diag']
        base_class_diag = self._reference_classwise_diagnostics(base_diag, bank)
        base_loss = base_class_diag['loss']
        bank['base_reference_loss'] = base_loss
        bank['base_reference_loss_cpu'] = base_loss.detach().cpu().numpy()
        bank['base_reference_diag'] = base_class_diag
        bank['base_reference_pred'] = base_diag['prediction'].detach()
        bank['base_reference_loss_diag'] = vector_diagnostics(base_loss)
        bank['base_reference_correct_rate_cpu'] = (
            base_class_diag['correct_rate'].detach().cpu().numpy())
        bank['base_reference_correct_rate_list'] = (
            base_class_diag['correct_rate'].detach().cpu().tolist())
        bank['stats'] = {
            'num_reference': int(labels.numel()),
            'per_class_counts': [
                int((labels == i).sum().item()) for i in range(self.num_classes)
            ],
            'selected_reference_hash': bank['selected_reference_hash'],
            'base_reference_loss': tensor_stats(base_diag['loss']),
            'base_reference_loss_by_class': classwise_tensor_stats(
                base_diag['loss'], labels, self.num_classes),
            'reference_confidence': tensor_stats(base_diag['confidence']),
            'reference_confidence_by_class': classwise_tensor_stats(
                base_diag['confidence'], labels, self.num_classes),
            'reference_entropy_by_class': classwise_tensor_stats(
                base_diag['entropy'], labels, self.num_classes),
            'reference_margin_by_class': classwise_tensor_stats(
                base_diag['margin'], labels, self.num_classes),
            'reference_energy_by_class': classwise_tensor_stats(
                base_diag['energy'], labels, self.num_classes),
            'reference_correct_by_class': classwise_tensor_stats(
                base_diag['correct'].float(), labels, self.num_classes),
            'reference_set_reused': True,
        }
        cache['reference_set_id'] = reference_set_id(identity)
        cache['reused'] = True
        cache_info = {key: value for key, value in cache.items() if key != 'bank'}
        return bank, cache_info

    def _load_prebuilt_anchor_set(self, config, probe_record):
        probe_reference_set_id = probe_record.get('reference_set_id')
        response_dir, manifest = self._find_prebuilt_anchor_set_dir(
            config, probe_reference_set_id)
        identity = manifest.get('identity')
        if identity is None:
            raise ValueError(f'anchor_set manifest missing identity: {response_dir}')
        cache = load_anchor_set(response_dir, identity)
        if cache is None:
            raise ValueError(f'invalid anchor_set artifact: {response_dir}')
        cached = cache['bank']
        required = [
            'features',
            'labels',
            'base_reference_loss',
            'base_reference_confidence',
            'base_reference_entropy',
            'base_reference_margin',
            'base_reference_energy',
            'base_reference_prediction',
            'base_reference_correct',
        ]
        missing = [key for key in required if key not in cached]
        if missing:
            raise ValueError(
                f'anchor_set artifact is missing required field(s): '
                f'{", ".join(missing)} ({response_dir})')
        device = torch.device('cuda')
        labels = torch.as_tensor(cached['labels'], device=device, dtype=torch.long)
        features = torch.as_tensor(cached['features'], device=device)
        with torch.no_grad():
            if isinstance(self.classifier, torch.nn.Linear):
                teacher_logits = F.linear(
                    features, self.classifier.weight, self.classifier.bias)
            else:
                teacher_logits = self.classifier(features)
            teacher_probs = torch.softmax(teacher_logits, dim=1).detach()
        bank = {
            'config': config,
            'label': labels,
            'features': features,
            'teacher_probs': teacher_probs,
            'selected_reference_hash': str(np.asarray(
                cached.get('selected_reference_hash', '')).item()),
            'stats': {
                'num_anchor': int(labels.numel()),
                'per_class_counts': [
                    int((labels == i).sum().item())
                    for i in range(self.num_classes)
                ],
                'selected_anchor_hash': str(np.asarray(
                    cached.get('selected_reference_hash', '')).item()),
                'probe_reference_set_id': probe_reference_set_id,
                'anchor_probe_disjoint': bool(
                    manifest.get('anchor_probe_disjoint', False)),
            },
        }
        cache['anchor_set_id'] = manifest.get(
            'anchor_set_id', reference_set_id(identity))
        cache['reused'] = True
        cache_info = {key: value for key, value in cache.items() if key != 'bank'}
        return bank, cache_info

    def _reference_set_identity(self, bank):
        return reference_set_identity(
            self.args,
            self.train_candidate_metadata,
            bank['config'],
            bank['selected_reference_hash'],
            self.classifier_name,
            self.runtime_mode,
        )

    def _reference_set_dir(self, identity):
        set_id = reference_set_id(identity)
        config = identity.get('reference_config', {})
        config_id = config.get('id', 'unknown_reference_config')
        seed = config.get('seed', self.args.seed)
        return (reference_set_root(self.args) / self.args.dataset / config_id /
                f'seed{seed}' / set_id)

    def _try_load_reference_set(self, bank):
        if self.args.rebuild_reference_set:
            return False
        identity = self._reference_set_identity(bank)
        response_dir = self._reference_set_dir(identity)
        cache = load_reference_set(response_dir, identity)
        if cache is None:
            return False
        cached = cache['bank']
        device = bank['label'].device
        bank['features'] = torch.as_tensor(cached['features'], device=device)
        bank['base_reference_sample_diag'] = {
            'loss': torch.as_tensor(
                cached['base_reference_loss'], device=device),
            'confidence': torch.as_tensor(
                cached['base_reference_confidence'], device=device),
            'entropy': torch.as_tensor(
                cached['base_reference_entropy'], device=device),
            'margin': torch.as_tensor(
                cached['base_reference_margin'], device=device),
            'energy': torch.as_tensor(
                cached['base_reference_energy'], device=device),
            'prediction': torch.as_tensor(
                cached['base_reference_prediction'],
                device=device,
                dtype=torch.long,
            ),
            'correct': torch.as_tensor(
                cached['base_reference_correct'],
                device=device,
                dtype=torch.bool,
            ),
        }
        cache['reference_set_id'] = reference_set_id(identity)
        cache['reused'] = True
        cache_info = {key: value for key, value in cache.items() if key != 'bank'}
        self.reference_set_records[bank['config'].id] = json_safe(cache_info)
        return True

    def _save_reference_set(self, bank):
        identity = self._reference_set_identity(bank)
        response_dir = self._reference_set_dir(identity)
        cache = save_reference_set(response_dir, identity, bank)
        cache['reference_set_id'] = reference_set_id(identity)
        self.reference_set_records[bank['config'].id] = json_safe(cache)
        return cache

    def _reference_sample_diagnostics_from_metadata(self, metadata, device):
        if not metadata:
            return None
        key_map = {
            'confidence': 'confidence',
            'entropy': 'entropy',
            'margin': 'margin',
            'energy': 'energy',
            'pred': 'prediction',
            'ce_loss': 'loss',
            'correct': 'correct',
        }
        if any(source not in metadata for source in key_map):
            return None
        ce_loss = np.asarray(metadata['ce_loss'])
        if not np.all(np.isfinite(ce_loss)):
            return None
        return {
            target: torch.as_tensor(
                metadata[source],
                device=device,
                dtype=(torch.bool if target == 'correct' else
                       torch.long if target == 'prediction' else torch.float32),
            )
            for source, target in key_map.items()
        }

    def _passes_reference_filter(self, net, data, label, config):
        if config.filter == 'all':
            return torch.ones_like(label, dtype=torch.bool)

        with torch.no_grad():
            logits = net(data.cuda(non_blocking=True))
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

        label = label.cuda(non_blocking=True)
        if config.filter == 'correct':
            mask = pred == label
        elif config.filter == 'high_confidence':
            mask = conf >= config.min_confidence
        elif config.filter == 'correct_high_confidence':
            mask = ((pred == label) & (conf >= config.min_confidence))
        else:
            raise ValueError(f'Unknown reference filter: {config.filter}')
        return mask.cpu()

    def _build_reference(self, net, data_loader, config):
        if self.train_candidate_metadata is not None:
            result = select_reference_from_train_candidate_metadata(
                self.train_candidate_metadata, data_loader, config,
                self.num_classes)
            if isinstance(result, tuple) and len(result) >= 3:
                return result[0], result[1], result[2]
            return result[0], result[1], None

        rng = random.Random(config.seed)
        seen = [0] * self.num_classes
        selected = [[] for _ in range(self.num_classes)]
        k = config.per_class

        for batch in tqdm(data_loader,
                          desc=f'Build TARR reference {config.id}',
                          disable=self.args.no_progress):
            data = batch['data'].cpu()
            label = batch['label'].cpu()
            keep = self._passes_reference_filter(net, data, label, config)
            for idx in range(label.numel()):
                if not bool(keep[idx].item()):
                    continue
                class_id = int(label[idx].item())
                if class_id < 0 or class_id >= self.num_classes:
                    continue
                seen[class_id] += 1
                item = data[idx].clone()
                if len(selected[class_id]) < k:
                    selected[class_id].append(item)
                else:
                    replace_idx = rng.randrange(seen[class_id])
                    if replace_idx < k:
                        selected[class_id][replace_idx] = item

        missing = [
            str(class_id) for class_id, samples in enumerate(selected)
            if len(samples) < k
        ]
        if missing:
            raise RuntimeError(
                'Not enough reference samples for classes: ' + ', '.join(missing)
            )

        ref_data = []
        ref_label = []
        for class_id, samples in enumerate(selected):
            ref_data.extend(samples)
            ref_label.extend([class_id] * len(samples))
        return torch.stack(ref_data), torch.tensor(ref_label, dtype=torch.long), None

    def _reference_feature_cache(self, net, bank):
        features = []
        with torch.no_grad():
            batch_size = self.args.reference_set_batch_size
            data = bank['data']
            labels = bank['label']
            for start in range(0, labels.numel(), batch_size):
                end = start + batch_size
                _, feature = net(data[start:end],
                                 return_feature=True)
                features.append(feature.detach())
        return torch.cat(features)

    def _reference_sample_diagnostics(self, net, bank):
        losses = []
        confidences = []
        predictions = []
        entropies = []
        margins = []
        energies = []
        with torch.no_grad():
            batch_size = self.args.reference_set_batch_size
            data = bank['data']
            labels = bank['label']
            for start in range(0, labels.numel(), batch_size):
                end = start + batch_size
                if self.reference_feature_cache_enabled:
                    if isinstance(self.classifier, torch.nn.Linear):
                        logits = F.linear(
                            bank['features'][start:end],
                            self.classifier.weight,
                            self.classifier.bias,
                        )
                    else:
                        logits = self.classifier(bank['features'][start:end])
                else:
                    logits = net(data[start:end])
                loss = F.cross_entropy(
                    logits,
                    labels[start:end],
                    reduction='none',
                )
                diag = logit_diagnostics(logits)
                losses.append(loss.detach())
                confidences.append(diag['conf'].detach())
                predictions.append(diag['pred'].detach())
                entropies.append(diag['entropy'].detach())
                margins.append(diag['margin'].detach())
                energies.append(diag['energy'].detach())
        prediction = torch.cat(predictions)
        labels = bank['label']
        return {
            'loss': torch.cat(losses),
            'confidence': torch.cat(confidences),
            'entropy': torch.cat(entropies),
            'margin': torch.cat(margins),
            'energy': torch.cat(energies),
            'prediction': prediction,
            'correct': prediction == labels,
        }

    def _classwise_mean(self, per_sample, bank):
        values = per_sample
        counts = bank['class_counts'].to(device=values.device,
                                         dtype=values.dtype)
        sums = values.new_zeros(self.num_classes)
        sums.scatter_add_(0, bank['label'].to(values.device), values)
        return sums / counts

    def _reference_classwise_diagnostics(self, sample_diag, bank):
        return {
            'loss': self._classwise_mean(sample_diag['loss'], bank),
            'conf': self._classwise_mean(sample_diag['confidence'], bank),
            'entropy': self._classwise_mean(sample_diag['entropy'], bank),
            'margin': self._classwise_mean(sample_diag['margin'], bank),
            'energy': self._classwise_mean(sample_diag['energy'], bank),
            'correct_rate': self._classwise_mean(
                sample_diag['correct'].float(), bank),
        }

    def _reference_pred_changed_rate(self, base_pred, adapted_pred, bank):
        changed = (base_pred != adapted_pred).float()
        return self._classwise_mean(changed, bank)

    def _reference_losses(self, net, bank):
        return self._classwise_mean(
            self._reference_sample_diagnostics(net, bank)['loss'], bank)

    def _state_restore_pairs(self, module, saved_state):
        current_state = module.state_dict()
        return [(current_state[key], saved_state[key]) for key in saved_state]

    def _copy_restore_state(self, restore_pairs):
        with torch.no_grad():
            for target, source in restore_pairs:
                target.copy_(source, non_blocking=True)

    def _restore_base(self, net):
        if self.args.update_scope == 'classifier':
            self._copy_restore_state(self.fc_restore_pairs)
        else:
            self._copy_restore_state(self.base_restore_pairs)
        if self.args.freeze_bn_stats:
            net.eval()
        else:
            net.train()

    def _set_trainable_parameters(self, net):
        if self.args.update_scope == 'classifier':
            if self.trainable_params is not None:
                return self.trainable_params
            for param in net.parameters():
                param.requires_grad_(False)
            for param in self.classifier.parameters():
                param.requires_grad_(True)
            return list(self.classifier.parameters())

        for param in net.parameters():
            param.requires_grad_(True)
        return list(net.parameters())

    def _optimizer_for_sample(self, trainable):
        if self.tta_update_impl == 'reused_torch_optimizer':
            return self.optimizer
        return torch.optim.SGD(trainable, lr=self.args.lr)

    def _entropy_loss(self, logits):
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log_softmax(logits, dim=1)
        return -(probs * log_probs).sum(dim=1).mean()

    def _gaussian_view_logits(self, net, data, target_feature):
        if self.args.perturbation_kind != 'gaussian':
            raise ValueError(
                f'{self.args.objective} supports only gaussian perturbation.')
        if self.args.perturbation_response == 'pixel':
            return self._batched_gaussian_pixel_logits(net, data)
        elif self.args.perturbation_response == 'feature':
            if target_feature is None or self.args.update_scope != 'classifier':
                _, feature = self._forward_with_feature(net, data)
            else:
                feature = target_feature
            return self._batched_gaussian_feature_logits(feature)
        else:
            raise ValueError(
                f'{self.args.objective} requires pixel or feature perturbation.')

    def _view_probabilities(self, view_logits):
        return torch.softmax(view_logits, dim=-1), torch.log_softmax(
            view_logits, dim=-1)

    def _view_entropy(self, probs, log_probs):
        return -(probs * log_probs).sum(dim=-1)

    def _memo_marginal_entropy_loss(self, net, data, target_feature):
        view_logits = self._gaussian_view_logits(net, data, target_feature)
        probs, _ = self._view_probabilities(view_logits)
        marginal_probs = probs.mean(dim=0)
        return -(
            marginal_probs * torch.log(marginal_probs.clamp_min(1e-12))
        ).sum(dim=1).mean()

    def _view_consistency_kl_loss(self, net, data, target_feature):
        view_logits = self._gaussian_view_logits(net, data, target_feature)
        probs, log_probs = self._view_probabilities(view_logits)
        mean_probs = probs.mean(dim=0).detach()
        kl = probs * (
            log_probs - torch.log(mean_probs.clamp_min(1e-12)).unsqueeze(0))
        return kl.sum(dim=-1).mean()

    def _view_consistency_js_loss(self, net, data, target_feature):
        view_logits = self._gaussian_view_logits(net, data, target_feature)
        probs, log_probs = self._view_probabilities(view_logits)
        view_entropy = self._view_entropy(probs, log_probs)
        mean_probs = probs.mean(dim=0)
        marginal_entropy = -(
            mean_probs * torch.log(mean_probs.clamp_min(1e-12))
        ).sum(dim=1)
        return (marginal_entropy - view_entropy.mean(dim=0)).mean()

    def _entropy_consistency_loss(self, net, data, target_feature):
        view_logits = self._gaussian_view_logits(net, data, target_feature)
        probs, log_probs = self._view_probabilities(view_logits)
        view_entropy = self._view_entropy(probs, log_probs)
        return view_entropy.var(dim=0, unbiased=False).mean()

    def _tta_loss(self, logits, pseudo_label):
        if self.args.objective == 'predicted_label_ce':
            return F.cross_entropy(logits, pseudo_label)
        if self.args.objective == 'entropy':
            return self._entropy_loss(logits)
        if (self.args.objective == 'memo_marginal_entropy'
                or self.args.objective in SOFT_VIEW_OBJECTIVES):
            raise ValueError(
                f'{self.args.objective} requires _tta_step_loss context.')
        raise ValueError(f'Unknown objective: {self.args.objective}')

    def _tta_step_loss(self, net, data, target_feature, logits, pseudo_label):
        if self.args.objective == 'memo_marginal_entropy':
            return self._memo_marginal_entropy_loss(net, data, target_feature)
        if self.args.objective == 'view_consistency_kl':
            return self._view_consistency_kl_loss(net, data, target_feature)
        if self.args.objective == 'view_consistency_js':
            return self._view_consistency_js_loss(net, data, target_feature)
        if self.args.objective == 'entropy_consistency':
            return self._entropy_consistency_loss(net, data, target_feature)
        return self._tta_loss(logits, pseudo_label)

    def _perturbation_generator(self, device):
        key = str(device)
        if key not in self.perturbation_generators:
            generator_device = device if device.type == 'cuda' else 'cpu'
            generator = torch.Generator(device=generator_device)
            generator.manual_seed(int(self.args.perturbation_seed))
            self.perturbation_generators[key] = generator
        return self.perturbation_generators[key]

    def _randn_like_for_perturbation(self, value):
        return torch.randn(
            value.shape,
            device=value.device,
            dtype=value.dtype,
            generator=self._perturbation_generator(value.device),
        )

    def _batched_gaussian_views(self, value):
        repeats = int(self.args.perturbation_repeats)
        expanded = value.unsqueeze(0).expand(repeats, *value.shape)
        expanded = expanded.reshape(repeats * value.shape[0], *value.shape[1:])
        return expanded + float(self.args.perturbation_eps) * (
            self._randn_like_for_perturbation(expanded))

    def _batched_gaussian_pixel_logits(self, net, data):
        repeats = int(self.args.perturbation_repeats)
        batch_size = data.shape[0]
        perturbed = self._batched_gaussian_views(data)
        logits = net(perturbed)
        return logits.reshape(repeats, batch_size, -1)

    def _batched_gaussian_feature_logits(self, feature):
        repeats = int(self.args.perturbation_repeats)
        batch_size = feature.shape[0]
        perturbed = self._batched_gaussian_views(feature)
        logits = self.classifier(perturbed)
        return logits.reshape(repeats, batch_size, -1)

    def _perturbation_metadata(self):
        return {
            'perturbation_response_code': PERTURBATION_RESPONSE_CODES[
                self.args.perturbation_response],
            'perturbation_kind_code': PERTURBATION_KIND_CODES[
                self.args.perturbation_kind],
            'perturbation_eps': float(self.args.perturbation_eps),
            'perturbation_repeats': int(self.args.perturbation_repeats),
            'perturbation_seed': int(self.args.perturbation_seed),
            'perturbation_cache_policy_code': PERTURBATION_CACHE_POLICY_CODES[
                self.args.perturbation_cache_policy],
        }

    def _zero_perturbation_diagnostics(self):
        output = {
            'perturbation_logit_l2': 0.0,
            'perturbation_prob_l1': 0.0,
            'perturbation_conf_delta': 0.0,
            'perturbation_entropy_delta': 0.0,
        }
        output.update(self._perturbation_metadata())
        return output

    def _forward_with_feature(self, net, data):
        try:
            output = net(data, return_feature=True)
        except TypeError as exc:
            raise RuntimeError(
                f'{net.__class__.__name__} does not support return_feature=True; '
                'feature perturbation response requires a network forward path '
                'that returns (logits, feature).') from exc
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            raise RuntimeError(
                f'{net.__class__.__name__} returned {type(output).__name__} '
                'for return_feature=True; feature perturbation response '
                'requires (logits, feature).')
        return output[0], output[1]

    def _target_feature_for_perturbation(self, net, data, target_feature):
        if target_feature is not None:
            return target_feature
        with torch.no_grad():
            _, feature = self._forward_with_feature(net, data)
        return feature.detach()

    def _pixel_perturbation_logits(self, net, data, pseudo_label):
        eps = float(self.args.perturbation_eps)
        if self.args.perturbation_kind == 'gaussian':
            with torch.no_grad():
                return self._batched_gaussian_pixel_logits(net, data).detach()

        perturbed_data = data.detach().clone().requires_grad_(True)
        net.zero_grad(set_to_none=True)
        logits_for_grad = net(perturbed_data)
        loss = F.cross_entropy(logits_for_grad, pseudo_label)
        loss.backward()
        if perturbed_data.grad is None:
            raise RuntimeError(
                'Pixel sign_ce perturbation could not compute an input gradient.')
        direction = perturbed_data.grad.detach().sign()
        net.zero_grad(set_to_none=True)
        with torch.no_grad():
            perturbed = data + eps * direction
            if self.args.freeze_bn_stats:
                logits = net(perturbed).detach()
                return logits.unsqueeze(0).expand(
                    int(self.args.perturbation_repeats),
                    *logits.shape,
                )
            return torch.stack([
                net(perturbed).detach()
                for _ in range(self.args.perturbation_repeats)
            ])

    def _feature_perturbation_logits(self, net, data, target_feature,
                                     pseudo_label):
        eps = float(self.args.perturbation_eps)
        feature = self._target_feature_for_perturbation(net, data,
                                                        target_feature)
        if self.args.perturbation_kind == 'gaussian':
            with torch.no_grad():
                return self._batched_gaussian_feature_logits(feature).detach()

        feature_for_grad = feature.detach().clone().requires_grad_(True)
        self.classifier.zero_grad(set_to_none=True)
        logits_for_grad = self.classifier(feature_for_grad)
        loss = F.cross_entropy(logits_for_grad, pseudo_label)
        loss.backward()
        if feature_for_grad.grad is None:
            raise RuntimeError(
                'Feature sign_ce perturbation could not compute a feature '
                'gradient.')
        direction = feature_for_grad.grad.detach().sign()
        self.classifier.zero_grad(set_to_none=True)
        with torch.no_grad():
            perturbed = feature + eps * direction
            if self.args.freeze_bn_stats:
                logits = self.classifier(perturbed).detach()
                return logits.unsqueeze(0).expand(
                    int(self.args.perturbation_repeats),
                    *logits.shape,
                )
            return torch.stack([
                self.classifier(perturbed).detach()
                for _ in range(self.args.perturbation_repeats)
            ])

    def _perturbation_diagnostics(self, net, data, logits0, target_diag0,
                                  target_feature):
        if (self.args.perturbation_response == 'none'
                or self.args.perturbation_eps == 0):
            return self._zero_perturbation_diagnostics()

        pseudo_label = target_diag0['pred'].detach()
        if self.args.perturbation_response == 'pixel':
            perturbed_logits = self._pixel_perturbation_logits(
                net, data, pseudo_label)
        elif self.args.perturbation_response == 'feature':
            perturbed_logits = self._feature_perturbation_logits(
                net, data, target_feature, pseudo_label)
        else:
            raise ValueError(
                f'Unknown perturbation response: '
                f'{self.args.perturbation_response}')

        clean_probs = target_diag0['probs']
        clean_conf = target_diag0['conf']
        clean_entropy = target_diag0['entropy']
        with torch.no_grad():
            logits = perturbed_logits
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            conf = probs.max(dim=-1).values
            entropy = -(probs * log_probs).sum(dim=-1)

            logit_l2_value = torch.linalg.vector_norm(
                logits - logits0.detach().unsqueeze(0),
                ord=2,
                dim=-1,
            ).mean()
            prob_l1_value = (
                probs - clean_probs.unsqueeze(0)).abs().sum(dim=-1).mean()
            conf_delta_value = (conf - clean_conf.unsqueeze(0)).mean()
            entropy_delta_value = (
                entropy - clean_entropy.unsqueeze(0)).mean()

        perturbation_values = torch.stack((
            logit_l2_value,
            prob_l1_value,
            conf_delta_value,
            entropy_delta_value,
        )).detach().cpu()
        output = {
            'perturbation_logit_l2': float(perturbation_values[0].item()),
            'perturbation_prob_l1': float(perturbation_values[1].item()),
            'perturbation_conf_delta': float(perturbation_values[2].item()),
            'perturbation_entropy_delta': float(perturbation_values[3].item()),
        }
        output.update(self._perturbation_metadata())
        return output

    def _score_from_delta(self, delta, probs, y_hat, score_rule):
        return score_from_delta(delta, probs, y_hat, score_rule)

    def _scores_from_delta(self, delta, probs, y_hat):
        return {
            score_rule: self._score_from_delta(delta, probs, y_hat, score_rule)
            for score_rule in self.score_rules
        }

    def _classifier_logits_from_features(self, features):
        if isinstance(self.classifier, torch.nn.Linear):
            return F.linear(features, self.classifier.weight, self.classifier.bias)
        return self.classifier(features)

    def _target_logits(self, net, data, target_feature):
        if self.target_feature_cache_enabled and target_feature is not None:
            return self._classifier_logits_from_features(target_feature)
        return net(data)

    def _param_regularization_loss(self):
        if self.args.update_scope == 'classifier':
            saved = self.fc_state
            module = self.classifier
        else:
            saved = self.base_state
            module = self.net_for_param_reg
        loss = None
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            target = saved[name].to(device=param.device, dtype=param.dtype)
            value = (param - target).pow(2).mean()
            loss = value if loss is None else loss + value
        if loss is None:
            return torch.zeros((), device=next(self.classifier.parameters()).device)
        return loss

    def _anchor_loss(self, reference_config_id):
        if not self.args.use_anchor_reference:
            return 0.0
        loss_type = self.args.anchor_loss_type
        if loss_type == 'none':
            return 0.0
        if loss_type == 'param_reg':
            return self._param_regularization_loss()
        bank = self.anchor_sets[reference_config_id]
        features = bank['features']
        labels = bank['label']
        batch_size = int(self.args.reference_set_batch_size)
        losses = []
        for start in range(0, labels.numel(), batch_size):
            end = start + batch_size
            logits = self._classifier_logits_from_features(features[start:end])
            if loss_type == 'ce':
                losses.append(F.cross_entropy(logits, labels[start:end]))
            elif loss_type == 'distill':
                teacher = bank['teacher_probs'][start:end]
                losses.append(F.kl_div(
                    F.log_softmax(logits, dim=1),
                    teacher,
                    reduction='batchmean',
                ))
            else:
                raise ValueError(f'Unknown anchor loss type: {loss_type}')
        return torch.stack(losses).mean()

    def _target_probe_loss(self, net, data, target_feature, logits,
                           pseudo_label, mode, hypothesis_label=None,
                           probe_type=None):
        if mode == 'existing':
            return self._tta_step_loss(net, data, target_feature, logits,
                                       pseudo_label)
        if mode == 'accept':
            probe_type = probe_type or self.args.accept_probe_type
            if probe_type == 'predicted_label_ce':
                return F.cross_entropy(logits, pseudo_label)
            if probe_type == 'entropy_min':
                return self._entropy_loss(logits)
            if probe_type == 'view_consistency':
                return self._view_consistency_js_loss(net, data, target_feature)
            raise ValueError(f'Unknown accept probe type: {probe_type}')
        if mode == 'reject':
            probe_type = probe_type or self.args.reject_probe_type
            if probe_type == 'entropy_max':
                return -self._entropy_loss(logits)
            if probe_type == 'uniform':
                uniform = torch.full_like(logits, 1.0 / logits.shape[1])
                return F.kl_div(
                    F.log_softmax(logits, dim=1),
                    uniform,
                    reduction='batchmean',
                )
            if probe_type == 'logit_suppression':
                return 0.5 * logits.pow(2).mean()
            raise ValueError(f'Unknown reject probe type: {probe_type}')
        raise ValueError(f'Unknown probe mode: {mode}')

    def _run_update(self, net, data, target_feature, logits0, pseudo_label,
                    reference_config_id, mode, hypothesis_label=None,
                    measure_bank=None, probe_type=None, branch_id=None):
        self.net_for_param_reg = net
        trainable = self._set_trainable_parameters(net)
        optimizer = torch.optim.SGD(trainable, lr=self.args.lr)
        save_steps = set(self.response_steps)
        snapshots = []
        with torch.no_grad():
            before = float(self._target_probe_loss(
                net, data, target_feature, logits0, pseudo_label, mode,
                hypothesis_label, probe_type).detach().cpu().item())
        for step in range(1, self.args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            if (mode == 'existing' and self.skip_view_clean_logits):
                logits = None
            else:
                logits = self._target_logits(net, data, target_feature)
            target_loss = self._target_probe_loss(
                net, data, target_feature, logits, pseudo_label, mode,
                hypothesis_label, probe_type)
            anchor_loss = self._anchor_loss(reference_config_id)
            loss = target_loss + float(self.args.anchor_weight) * anchor_loss
            loss.backward()
            optimizer.step()
            if step in save_steps:
                net.eval()
                with torch.no_grad():
                    post_logits = self._target_logits(net, data, target_feature)
                    after = float(self._target_probe_loss(
                        net, data, target_feature, post_logits, pseudo_label, mode,
                        hypothesis_label, probe_type).detach().cpu().item())
                snapshots.append({
                    'step': int(step),
                    'branch_id': branch_id or '',
                    'probe_type': probe_type or '',
                    'before': before,
                    'after': after,
                    'post_logits': post_logits.detach(),
                    'measure': (
                        self._measure_reference_response(net, measure_bank)
                        if measure_bank is not None else None),
                })
                if not self.args.freeze_bn_stats:
                    net.train()
        net.eval()
        if len(snapshots) != len(self.response_steps):
            observed = [item['step'] for item in snapshots]
            raise RuntimeError(
                f'Missing response step snapshots: expected '
                f'{self.response_steps}, got {observed}')
        return snapshots

    def _run_existing_update_with_reference_snapshots(
            self, net, data, target_feature, logits0, pseudo_label):
        self.net_for_param_reg = net
        trainable = self._set_trainable_parameters(net)
        optimizer = self._optimizer_for_sample(trainable)
        save_steps = set(self.response_steps)
        snapshots = []
        measurements_by_reference = {
            reference_config_id: []
            for reference_config_id in self.reference_sets
        }
        with torch.no_grad():
            before = float(self._tta_step_loss(
                net,
                data,
                target_feature,
                logits0,
                pseudo_label,
            ).detach().cpu().item())
        for step in range(1, self.args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            if self.skip_view_clean_logits:
                logits = None
            elif self.target_feature_cache_enabled:
                logits = self._classifier_logits_from_features(target_feature)
            else:
                logits = net(data)
            loss = self._tta_step_loss(
                net,
                data,
                target_feature,
                logits,
                pseudo_label,
            )
            loss.backward()
            optimizer.step()
            if step not in save_steps:
                continue
            net.eval()
            with torch.no_grad():
                if self.target_feature_cache_enabled:
                    post_logits = self._classifier_logits_from_features(
                        target_feature)
                else:
                    post_logits = net(data)
                after = float(self._tta_step_loss(
                    net,
                    data,
                    target_feature,
                    post_logits,
                    pseudo_label,
                ).detach().cpu().item())
            snapshots.append({
                'step': int(step),
                'before': before,
                'after': after,
                'post_logits': post_logits.detach(),
                'measure': None,
            })
            for reference_config_id, bank in self.reference_sets.items():
                measurements_by_reference[reference_config_id].append(
                    self._measure_reference_response(net, bank))
            if not self.args.freeze_bn_stats:
                net.train()
        net.eval()
        if len(snapshots) != len(self.response_steps):
            observed = [item['step'] for item in snapshots]
            raise RuntimeError(
                f'Missing response step snapshots: expected '
                f'{self.response_steps}, got {observed}')
        return snapshots, measurements_by_reference

    def _measure_reference_response(self, net, bank):
        adapted_sample_diag = self._reference_sample_diagnostics(net, bank)
        adapted_class_diag = self._reference_classwise_diagnostics(
            adapted_sample_diag, bank)
        adapted_reference_loss = adapted_class_diag['loss']
        delta = adapted_reference_loss - bank['base_reference_loss']
        base_class_diag = bank['base_reference_diag']
        return {
            'sample_diag': adapted_sample_diag,
            'class_diag': adapted_class_diag,
            'adapted_reference_loss': adapted_reference_loss,
            'delta': delta,
            'reference_conf_delta':
            adapted_class_diag['conf'] - base_class_diag['conf'],
            'reference_entropy_delta':
            adapted_class_diag['entropy'] - base_class_diag['entropy'],
            'reference_margin_delta':
            adapted_class_diag['margin'] - base_class_diag['margin'],
            'reference_energy_delta':
            adapted_class_diag['energy'] - base_class_diag['energy'],
            'reference_pred_changed_rate': self._reference_pred_changed_rate(
                bank['base_reference_pred'], adapted_sample_diag['prediction'], bank),
        }

    def _probe_config_cache(self):
        cache = {
            'use_accept_reject_probe': int(bool(self.args.use_accept_reject_probe)),
            'use_anchor_reference': int(bool(self.args.use_anchor_reference)),
            'anchor_weight': float(self.args.anchor_weight),
            'probe_schema_version': (
                2 if getattr(self.args, 'use_response_bank', False)
                else 1 if self.args.use_accept_reject_probe else 0),
            'accept_probe_type': self.args.accept_probe_type,
            'reject_probe_type': self.args.reject_probe_type,
            'anchor_loss_type': self.args.anchor_loss_type,
            'probe_config_id': probe_config_id(self.args),
        }
        if getattr(self.args, 'use_response_bank', False):
            cache.update({
                'accept_branch_ids':
                np.asarray(self.args.accept_branch_ids, dtype='<U64'),
                'accept_branch_probe_types':
                np.asarray(self.args.accept_probe_type_bank, dtype='<U64'),
                'reject_branch_ids':
                np.asarray(self.args.reject_branch_ids, dtype='<U64'),
                'reject_branch_probe_types':
                np.asarray(self.args.reject_probe_type_bank, dtype='<U64'),
                'primary_accept_branch_id': self.args.primary_accept_branch_id,
                'primary_reject_branch_id': self.args.primary_reject_branch_id,
                'response_bank_schema_version': 1,
            })
        return cache

    def _common_target_values(self, net, data):
        with torch.no_grad():
            target_feature = None
            if self.target_feature_cache_enabled:
                logits0, target_feature = net(data, return_feature=True)
                target_feature = target_feature.detach()
            else:
                logits0 = net(data)
            target_diag0 = logit_diagnostics(logits0)
        return logits0, target_feature, target_diag0

    def _target_cache(self, target_diag0, post_logits, pred, probs0,
                      target_tta_loss_before, target_tta_loss_after,
                      perturbation_diag):
        target_conf = target_diag0['conf']
        target_values = torch.stack((
            pred.to(dtype=target_conf.dtype)[0],
            target_conf[0],
            target_diag0['entropy'][0],
            target_diag0['margin'][0],
            target_diag0['energy'][0],
        )).detach().cpu()
        pred_int = int(target_values[0].item())
        target_conf_float = float(target_values[1].item())
        target_entropy_float = float(target_values[2].item())
        target_margin_float = float(target_values[3].item())
        target_energy_float = float(target_values[4].item())
        with torch.no_grad():
            post_tta_diag = logit_diagnostics(post_logits)
            post_tta_probs = post_tta_diag['probs']
            post_tta_conf = post_tta_diag['conf']
            post_tta_pred = post_tta_diag['pred']
            post_tta_probs_cpu = post_tta_probs.detach().cpu()[0]
        post_tta_values = torch.stack((
            post_tta_pred.to(dtype=post_tta_conf.dtype)[0],
            post_tta_conf[0],
            post_tta_diag['entropy'][0],
            post_tta_diag['margin'][0],
            post_tta_diag['energy'][0],
        )).detach().cpu()
        post_tta_pred_int = int(post_tta_values[0].item())
        post_tta_conf_float = float(post_tta_values[1].item())
        post_tta_entropy_float = float(post_tta_values[2].item())
        post_tta_margin_float = float(post_tta_values[3].item())
        post_tta_energy_float = float(post_tta_values[4].item())
        post_tta_pseudo_label_prob = float(post_tta_probs_cpu[pred_int].item())
        return pred_int, {
            'y_hat': pred_int,
            'target_conf': target_conf_float,
            'target_entropy': target_entropy_float,
            'target_probs': probs0.detach().cpu()[0].numpy(),
            'target_margin': target_margin_float,
            'target_energy': target_energy_float,
            'perturbation_logit_l2': perturbation_diag['perturbation_logit_l2'],
            'perturbation_prob_l1': perturbation_diag['perturbation_prob_l1'],
            'perturbation_conf_delta': perturbation_diag[
                'perturbation_conf_delta'],
            'perturbation_entropy_delta': perturbation_diag[
                'perturbation_entropy_delta'],
            'perturbation_response_code': perturbation_diag[
                'perturbation_response_code'],
            'perturbation_kind_code': perturbation_diag['perturbation_kind_code'],
            'perturbation_eps': perturbation_diag['perturbation_eps'],
            'perturbation_repeats': perturbation_diag['perturbation_repeats'],
            'perturbation_seed': perturbation_diag['perturbation_seed'],
            'perturbation_cache_policy_code': perturbation_diag[
                'perturbation_cache_policy_code'],
            'target_tta_loss_before': target_tta_loss_before,
            'target_tta_loss_after': target_tta_loss_after,
            'post_tta_pred': post_tta_pred_int,
            'post_tta_target_conf': post_tta_conf_float,
            'post_tta_target_entropy': post_tta_entropy_float,
            'post_tta_target_probs': post_tta_probs_cpu.numpy(),
            'post_tta_pseudo_label_prob': post_tta_pseudo_label_prob,
            'adapted_target_pred': post_tta_pred_int,
            'adapted_target_conf': post_tta_conf_float,
            'adapted_target_entropy': post_tta_entropy_float,
            'adapted_target_margin': post_tta_margin_float,
            'adapted_target_energy': post_tta_energy_float,
            'adapted_target_probs': post_tta_probs_cpu.numpy(),
            'target_conf_delta': post_tta_conf_float - target_conf_float,
            'target_entropy_delta':
            post_tta_entropy_float - target_entropy_float,
            'target_margin_delta': post_tta_margin_float - target_margin_float,
            'target_energy_delta': post_tta_energy_float - target_energy_float,
            'target_pred_changed': int(post_tta_pred_int != pred_int),
        }

    def _target_cache_from_snapshots(self, target_diag0, snapshots, pred, probs0,
                                     perturbation_diag):
        target_conf = target_diag0['conf']
        target_values = torch.stack((
            pred.to(dtype=target_conf.dtype)[0],
            target_conf[0],
            target_diag0['entropy'][0],
            target_diag0['margin'][0],
            target_diag0['energy'][0],
        )).detach().cpu()
        pred_int = int(target_values[0].item())
        target_conf_float = float(target_values[1].item())
        target_entropy_float = float(target_values[2].item())
        target_margin_float = float(target_values[3].item())
        target_energy_float = float(target_values[4].item())

        post_pred = []
        post_conf = []
        post_entropy = []
        post_margin = []
        post_energy = []
        post_probs = []
        post_pseudo_prob = []
        loss_after = []
        pred_changed = []
        for snapshot in snapshots:
            post_logits = snapshot['post_logits']
            with torch.no_grad():
                post_diag = logit_diagnostics(post_logits)
                probs = post_diag['probs'].detach().cpu()[0]
            post_pred_int = int(post_diag['pred'].detach().cpu()[0].item())
            post_pred.append(post_pred_int)
            post_conf.append(float(post_diag['conf'].detach().cpu()[0].item()))
            post_entropy.append(float(post_diag['entropy'].detach().cpu()[0].item()))
            post_margin.append(float(post_diag['margin'].detach().cpu()[0].item()))
            post_energy.append(float(post_diag['energy'].detach().cpu()[0].item()))
            post_probs.append(probs.numpy())
            post_pseudo_prob.append(float(probs[pred_int].item()))
            loss_after.append(float(snapshot['after']))
            pred_changed.append(int(post_pred_int != pred_int))

        post_pred = np.asarray(post_pred, dtype=np.int64)
        post_conf = np.asarray(post_conf, dtype=np.float64)
        post_entropy = np.asarray(post_entropy, dtype=np.float64)
        post_margin = np.asarray(post_margin, dtype=np.float64)
        post_energy = np.asarray(post_energy, dtype=np.float64)
        loss_after = np.asarray(loss_after, dtype=np.float64)
        return pred_int, {
            'y_hat': pred_int,
            'target_conf': target_conf_float,
            'target_entropy': target_entropy_float,
            'target_probs': probs0.detach().cpu()[0].numpy(),
            'target_margin': target_margin_float,
            'target_energy': target_energy_float,
            'perturbation_logit_l2': perturbation_diag['perturbation_logit_l2'],
            'perturbation_prob_l1': perturbation_diag['perturbation_prob_l1'],
            'perturbation_conf_delta': perturbation_diag[
                'perturbation_conf_delta'],
            'perturbation_entropy_delta': perturbation_diag[
                'perturbation_entropy_delta'],
            'perturbation_response_code': perturbation_diag[
                'perturbation_response_code'],
            'perturbation_kind_code': perturbation_diag['perturbation_kind_code'],
            'perturbation_eps': perturbation_diag['perturbation_eps'],
            'perturbation_repeats': perturbation_diag['perturbation_repeats'],
            'perturbation_seed': perturbation_diag['perturbation_seed'],
            'perturbation_cache_policy_code': perturbation_diag[
                'perturbation_cache_policy_code'],
            'target_tta_loss_before': float(snapshots[0]['before']),
            'target_tta_loss_after': loss_after,
            'post_tta_pred': post_pred,
            'post_tta_target_conf': post_conf,
            'post_tta_target_entropy': post_entropy,
            'post_tta_target_probs': np.stack(post_probs),
            'post_tta_pseudo_label_prob': np.asarray(
                post_pseudo_prob, dtype=np.float64),
            'adapted_target_pred': post_pred,
            'adapted_target_conf': post_conf,
            'adapted_target_entropy': post_entropy,
            'adapted_target_margin': post_margin,
            'adapted_target_energy': post_energy,
            'adapted_target_probs': np.stack(post_probs),
            'target_conf_delta': post_conf - target_conf_float,
            'target_entropy_delta': post_entropy - target_entropy_float,
            'target_margin_delta': post_margin - target_margin_float,
            'target_energy_delta': post_energy - target_energy_float,
            'target_pred_changed': np.asarray(pred_changed, dtype=np.int64),
        }

    def _step_cache_from_measurements(self, measurements):
        return {
            'adapted_reference_loss': torch.stack([
                item['adapted_reference_loss'] for item in measurements
            ]).detach().cpu().numpy(),
            'delta': torch.stack([
                item['delta'] for item in measurements
            ]).detach().cpu().numpy(),
            'reference_conf_delta_by_class': torch.stack([
                item['reference_conf_delta'] for item in measurements
            ]).detach().cpu().numpy(),
            'reference_entropy_delta_by_class': torch.stack([
                item['reference_entropy_delta'] for item in measurements
            ]).detach().cpu().numpy(),
            'reference_margin_delta_by_class': torch.stack([
                item['reference_margin_delta'] for item in measurements
            ]).detach().cpu().numpy(),
            'reference_energy_delta_by_class': torch.stack([
                item['reference_energy_delta'] for item in measurements
            ]).detach().cpu().numpy(),
            'reference_pred_changed_rate_by_class': torch.stack([
                item['reference_pred_changed_rate'] for item in measurements
            ]).detach().cpu().numpy(),
            'reference_correct_rate_after_by_class': torch.stack([
                item['class_diag']['correct_rate'] for item in measurements
            ]).detach().cpu().numpy(),
            'adapted_reference_loss_mean': np.asarray([
                vector_diagnostics(item['adapted_reference_loss'])['mean']
                for item in measurements
            ], dtype=np.float64),
            'adapted_reference_loss_std': np.asarray([
                vector_diagnostics(item['adapted_reference_loss'])['std']
                for item in measurements
            ], dtype=np.float64),
            'adapted_reference_loss_min': np.asarray([
                vector_diagnostics(item['adapted_reference_loss'])['min']
                for item in measurements
            ], dtype=np.float64),
            'adapted_reference_loss_max': np.asarray([
                vector_diagnostics(item['adapted_reference_loss'])['max']
                for item in measurements
            ], dtype=np.float64),
            'reference_delta_mean': np.asarray([
                vector_diagnostics(item['delta'])['mean']
                for item in measurements
            ], dtype=np.float64),
            'reference_delta_std': np.asarray([
                vector_diagnostics(item['delta'])['std']
                for item in measurements
            ], dtype=np.float64),
            'reference_delta_min': np.asarray([
                vector_diagnostics(item['delta'])['min']
                for item in measurements
            ], dtype=np.float64),
            'reference_delta_max': np.asarray([
                vector_diagnostics(item['delta'])['max']
                for item in measurements
            ], dtype=np.float64),
            'reference_delta_positive_mean': np.asarray([
                float(item['delta'].clamp(min=0.0).mean().detach().cpu().item())
                for item in measurements
            ], dtype=np.float64),
        }

    def _scores_from_cache_row(self, cache):
        scores = {}
        for score_rule in self.score_rules:
            row_cache = {
                'response_steps': np.asarray(self.response_steps, dtype=np.int64)
            }
            for key, value in cache.items():
                array = np.asarray(value)
                if array.ndim == 0:
                    row_cache[key] = array.reshape(1)
                else:
                    row_cache[key] = array.reshape(1, *array.shape)
            scores[score_rule] = torch.as_tensor(
                ood_score_from_cache(row_cache, score_rule)[0],
                device=cache['delta'].device
                if isinstance(cache.get('delta'), torch.Tensor) else 'cpu',
                dtype=torch.float32,
            )
        return scores

    def _reference_output_from_measurement(self, measure, bank, probs0, pred_int,
                                           primary_score_rule, runtime,
                                           target_cache, probe_extra=None,
                                           step_cache=None):
        adapted_reference_loss = measure['adapted_reference_loss']
        delta = measure['delta']
        adapted_class_diag = measure['class_diag']
        adapted_sample_diag = measure['sample_diag']
        base_class_diag = bank['base_reference_diag']
        base_loss_diag = bank['base_reference_loss_diag']
        adapted_loss_diag = vector_diagnostics(adapted_reference_loss)
        delta_diag = vector_diagnostics(delta)
        positive_delta_mean = float(delta.clamp(min=0.0).mean().item())
        cache = dict(target_cache)
        cache.update({
            'base_reference_loss': bank['base_reference_loss_cpu'],
            'adapted_reference_loss': adapted_reference_loss.detach().cpu().numpy(),
            'delta': delta.detach().cpu().numpy(),
            'reference_conf_delta_by_class':
            measure['reference_conf_delta'].detach().cpu().numpy(),
            'reference_entropy_delta_by_class':
            measure['reference_entropy_delta'].detach().cpu().numpy(),
            'reference_margin_delta_by_class':
            measure['reference_margin_delta'].detach().cpu().numpy(),
            'reference_energy_delta_by_class':
            measure['reference_energy_delta'].detach().cpu().numpy(),
            'reference_pred_changed_rate_by_class':
            measure['reference_pred_changed_rate'].detach().cpu().numpy(),
            'reference_correct_rate_before_by_class':
            bank['base_reference_correct_rate_cpu'],
            'reference_correct_rate_after_by_class':
            adapted_class_diag['correct_rate'].detach().cpu().numpy(),
            'base_reference_loss_mean': base_loss_diag['mean'],
            'base_reference_loss_std': base_loss_diag['std'],
            'base_reference_loss_min': base_loss_diag['min'],
            'base_reference_loss_max': base_loss_diag['max'],
            'adapted_reference_loss_mean': adapted_loss_diag['mean'],
            'adapted_reference_loss_std': adapted_loss_diag['std'],
            'adapted_reference_loss_min': adapted_loss_diag['min'],
            'adapted_reference_loss_max': adapted_loss_diag['max'],
            'reference_delta_mean': delta_diag['mean'],
            'reference_delta_std': delta_diag['std'],
            'reference_delta_min': delta_diag['min'],
            'reference_delta_max': delta_diag['max'],
            'reference_delta_positive_mean': positive_delta_mean,
            'runtime_per_sample': runtime,
        })
        if step_cache:
            cache.update(step_cache)
        if probe_extra:
            cache.update(probe_extra)
        if any(score_rule in PROBE_SCORE_RULES for score_rule in self.score_rules):
            ood_scores = self._scores_from_cache_row(cache)
        else:
            ood_scores = self._scores_from_delta(delta, probs0.detach(), pred_int)
        primary_ood_score = float(
            ood_scores[primary_score_rule].detach().cpu().item())
        return {
            'primary_ood_score': primary_ood_score,
            'ood_scores': ood_scores,
            'cache': cache,
            'debug_core': {
                'ood_score': primary_ood_score,
                'id_confidence': -primary_ood_score,
                'base_reference_loss_mean': base_loss_diag['mean'],
                'base_reference_loss_std': base_loss_diag['std'],
                'base_reference_loss_min': base_loss_diag['min'],
                'base_reference_loss_max': base_loss_diag['max'],
                'adapted_reference_loss_mean': adapted_loss_diag['mean'],
                'adapted_reference_loss_std': adapted_loss_diag['std'],
                'adapted_reference_loss_min': adapted_loss_diag['min'],
                'adapted_reference_loss_max': adapted_loss_diag['max'],
                'reference_delta_mean': delta_diag['mean'],
                'reference_delta_std': delta_diag['std'],
                'reference_delta_min': delta_diag['min'],
                'reference_delta_max': delta_diag['max'],
                'reference_delta_positive_mean': positive_delta_mean,
                'reference_correct_rate_before_by_class':
                bank['base_reference_correct_rate_list'],
                'reference_correct_rate_after_by_class':
                adapted_class_diag['correct_rate'].detach().cpu().tolist(),
            },
        }

    def _score_one_anchor_or_probe(self, net, data):
        start_time = time.perf_counter()
        self._restore_base(net)
        logits0, target_feature, target_diag0 = self._common_target_values(net, data)
        probs0 = target_diag0['probs']
        pred = target_diag0['pred']
        pseudo_label = pred.detach()
        pred_int = int(pred.detach().cpu().view(-1)[0].item())
        perturbation_diag = self._perturbation_diagnostics(
            net, data, logits0, target_diag0, target_feature)
        primary_score_rule = self.score_rules[0]
        per_reference = {}

        for reference_config_id, bank in self.reference_sets.items():
            if self.args.use_accept_reject_probe:
                accept_results = []
                for branch_id, probe_type in zip(
                        self.args.accept_branch_ids,
                        self.args.accept_probe_type_bank):
                    self._restore_base(net)
                    snapshots = self._run_update(
                        net, data, target_feature, logits0, pseudo_label,
                        reference_config_id, 'accept', measure_bank=bank,
                        probe_type=probe_type, branch_id=branch_id)
                    measurements = [item['measure'] for item in snapshots]
                    step_cache = self._step_cache_from_measurements(measurements)
                    target_objective_delta = np.asarray([
                        float(item['after'] - item['before'])
                        for item in snapshots
                    ], dtype=np.float64)
                    accept_results.append({
                        'branch_id': branch_id,
                        'probe_type': probe_type,
                        'snapshots': snapshots,
                        'measurements': measurements,
                        'measure': measurements[-1],
                        'step_cache': step_cache,
                        'target_objective_delta': target_objective_delta,
                    })

                reject_results = []
                with torch.no_grad():
                    base_entropy = target_diag0['entropy'][0]
                for branch_id, probe_type in zip(
                        self.args.reject_branch_ids,
                        self.args.reject_probe_type_bank):
                    self._restore_base(net)
                    snapshots = self._run_update(
                        net, data, target_feature, logits0, pseudo_label,
                        reference_config_id, 'reject', measure_bank=bank,
                        probe_type=probe_type, branch_id=branch_id)
                    measurements = [item['measure'] for item in snapshots]
                    step_cache = self._step_cache_from_measurements(measurements)
                    target_objective_delta = np.asarray([
                        float(item['after'] - item['before'])
                        for item in snapshots
                    ], dtype=np.float64)
                    entropy_delta = []
                    with torch.no_grad():
                        for item in snapshots:
                            reject_entropy = logit_diagnostics(
                                item['post_logits'])['entropy'][0]
                            entropy_delta.append(float(
                                (reject_entropy - base_entropy
                                 ).detach().cpu().item()))
                    reject_results.append({
                        'branch_id': branch_id,
                        'probe_type': probe_type,
                        'snapshots': snapshots,
                        'measurements': measurements,
                        'measure': measurements[-1],
                        'step_cache': step_cache,
                        'target_objective_delta': target_objective_delta,
                        'target_entropy_delta':
                        np.asarray(entropy_delta, dtype=np.float64),
                    })
                self._restore_base(net)

                primary_accept = accept_results[0]
                primary_reject = reject_results[0]
                accept_snapshots = primary_accept['snapshots']
                accept_measure = primary_accept['measure']
                accept_step_cache = primary_accept['step_cache']
                accept_target_objective_delta = primary_accept['target_objective_delta']
                reject_step_cache = primary_reject['step_cache']
                reject_target_objective_delta = primary_reject['target_objective_delta']
                reject_target_entropy_delta = primary_reject[
                    'target_entropy_delta']

                runtime = time.perf_counter() - start_time
                _, target_cache = self._target_cache_from_snapshots(
                    target_diag0,
                    accept_snapshots,
                    pred,
                    probs0,
                    perturbation_diag,
                )
                probe_extra = self._probe_config_cache()
                probe_extra.update({
                    'accept_ref_loss_delta': accept_step_cache['delta'],
                    'reject_ref_loss_delta': reject_step_cache['delta'],
                    'accept_target_objective_delta': accept_target_objective_delta,
                    'reject_target_objective_delta': reject_target_objective_delta,
                    'reject_target_entropy_delta': reject_target_entropy_delta,
                })
                if getattr(self.args, 'use_response_bank', False):
                    probe_extra.update({
                        'accept_ref_loss_delta_bank': np.stack([
                            item['step_cache']['delta']
                            for item in accept_results
                        ], axis=1),
                        'reject_ref_loss_delta_bank': np.stack([
                            item['step_cache']['delta']
                            for item in reject_results
                        ], axis=1),
                        'accept_target_objective_delta_bank': np.stack([
                            item['target_objective_delta'] for item in accept_results
                        ], axis=1),
                        'reject_target_objective_delta_bank': np.stack([
                            item['target_objective_delta'] for item in reject_results
                        ], axis=1),
                        'reject_target_entropy_delta_bank': np.stack([
                            item['target_entropy_delta']
                            for item in reject_results
                        ], axis=1),
                    })
                output = self._reference_output_from_measurement(
                    accept_measure, bank, probs0, pred_int, primary_score_rule,
                    runtime, target_cache, probe_extra,
                    step_cache=accept_step_cache)
            else:
                self._restore_base(net)
                snapshots = self._run_update(
                    net, data, target_feature, logits0, pseudo_label,
                    reference_config_id, 'existing', measure_bank=bank)
                measurements = [item['measure'] for item in snapshots]
                measure = measurements[-1]
                step_cache = self._step_cache_from_measurements(measurements)
                runtime = time.perf_counter() - start_time
                _, target_cache = self._target_cache_from_snapshots(
                    target_diag0,
                    snapshots,
                    pred,
                    probs0,
                    perturbation_diag,
                )
                output = self._reference_output_from_measurement(
                    measure, bank, probs0, pred_int, primary_score_rule, runtime,
                    target_cache, self._probe_config_cache(),
                    step_cache=step_cache)
            debug_row = {
                'reference_config_id': reference_config_id,
                'score_rule': self.args.score_rule,
                'primary_score_rule': primary_score_rule,
                'pred': pred_int,
                'y_hat': pred_int,
                'target_conf': float(target_diag0['conf'][0].detach().cpu().item()),
                'target_entropy': float(
                    target_diag0['entropy'][0].detach().cpu().item()),
                'runtime_per_sample': output['cache']['runtime_per_sample'],
            }
            debug_row.update(output['debug_core'])
            if self.args.score_rule in {'all', 'probe_all'}:
                for score_rule, score in output['ood_scores'].items():
                    value = float(score.detach().cpu().item())
                    debug_row[f'ood_score_{score_rule}'] = value
                    debug_row[f'id_confidence_{score_rule}'] = -value
            if self.debug_output_mode == 'full':
                self.sample_debug.append(debug_row)
            per_reference[reference_config_id] = (
                {k: v.detach() for k, v in output['ood_scores'].items()},
                output['cache'],
            )
        self._restore_base(net)
        return pred.detach(), per_reference

    def score_one(self, net, data):
        if self.args.use_accept_reject_probe or self.args.use_anchor_reference:
            return self._score_one_anchor_or_probe(net, data)
        self._restore_base(net)
        start_time = time.perf_counter()

        with torch.no_grad():
            target_feature = None
            if self.target_feature_cache_enabled:
                logits0, target_feature = net(data, return_feature=True)
                target_feature = target_feature.detach()
            else:
                logits0 = net(data)
            target_diag0 = logit_diagnostics(logits0)
            probs0 = target_diag0['probs']
            pred = target_diag0['pred']

        perturbation_diag = self._perturbation_diagnostics(
            net, data, logits0, target_diag0, target_feature)

        pseudo_label = pred.detach()
        snapshots, measurements_by_reference = (
            self._run_existing_update_with_reference_snapshots(
                net, data, target_feature, logits0, pseudo_label))
        runtime = time.perf_counter() - start_time
        primary_score_rule = self.score_rules[0]
        pred_int, target_cache = self._target_cache_from_snapshots(
            target_diag0,
            snapshots,
            pred,
            probs0,
            perturbation_diag,
        )

        per_reference = {}
        for reference_config_id, bank in self.reference_sets.items():
            measurements = measurements_by_reference[reference_config_id]
            measure = measurements[-1]
            step_cache = self._step_cache_from_measurements(measurements)
            output = self._reference_output_from_measurement(
                measure, bank, probs0, pred_int, primary_score_rule, runtime,
                target_cache, step_cache=step_cache)
            debug_row = {
                'reference_config_id': reference_config_id,
                'score_rule': self.args.score_rule,
                'primary_score_rule': primary_score_rule,
                'pred': pred_int,
                'y_hat': pred_int,
                'target_conf': target_cache['target_conf'],
                'target_entropy': target_cache['target_entropy'],
                'runtime_per_sample': output['cache']['runtime_per_sample'],
            }
            debug_row.update(output['debug_core'])
            if self.args.score_rule == 'all':
                for score_rule, score in output['ood_scores'].items():
                    value = float(score.detach().cpu().item())
                    debug_row[f'ood_score_{score_rule}'] = value
                    debug_row[f'id_confidence_{score_rule}'] = -value
            if self.debug_output_mode == 'full':
                self.sample_debug.append(debug_row)
            per_reference[reference_config_id] = (
                {k: v.detach() for k, v in output['ood_scores'].items()},
                output['cache'],
            )
        return pred.detach(), per_reference

    def inference(self,
                  net,
                  data_loader,
                  progress=True,
                  max_samples=0,
                  cache_writers=None,
                  cache_label_override=None):
        inference_start = time.perf_counter()
        pred_list, label_list = [], []
        streaming_cache = cache_writers is not None
        per_reference_lists = {
            reference_config_id: {
                'ood_score': {
                    score_rule: []
                    for score_rule in self.score_rules
                },
                'cache': {
                    'target_global_index': [],
                    'y_hat': [],
                    'target_conf': [],
                    'target_entropy': [],
                    'target_probs': [],
                    'target_margin': [],
                    'target_energy': [],
                    'perturbation_logit_l2': [],
                    'perturbation_prob_l1': [],
                    'perturbation_conf_delta': [],
                    'perturbation_entropy_delta': [],
                    'perturbation_response_code': [],
                    'perturbation_kind_code': [],
                    'perturbation_eps': [],
                    'perturbation_repeats': [],
                    'perturbation_seed': [],
                    'perturbation_cache_policy_code': [],
                    'target_tta_loss_before': [],
                    'target_tta_loss_after': [],
                    'post_tta_pred': [],
                    'post_tta_target_conf': [],
                    'post_tta_target_entropy': [],
                    'post_tta_target_probs': [],
                    'post_tta_pseudo_label_prob': [],
                    'adapted_target_pred': [],
                    'adapted_target_conf': [],
                    'adapted_target_entropy': [],
                    'adapted_target_margin': [],
                    'adapted_target_energy': [],
                    'adapted_target_probs': [],
                    'target_conf_delta': [],
                    'target_entropy_delta': [],
                    'target_margin_delta': [],
                    'target_energy_delta': [],
                    'target_pred_changed': [],
                    'base_reference_loss': [],
                    'adapted_reference_loss': [],
                    'delta': [],
                    'reference_conf_delta_by_class': [],
                    'reference_entropy_delta_by_class': [],
                    'reference_margin_delta_by_class': [],
                    'reference_energy_delta_by_class': [],
                    'reference_pred_changed_rate_by_class': [],
                    'reference_correct_rate_before_by_class': [],
                    'reference_correct_rate_after_by_class': [],
                    'base_reference_loss_mean': [],
                    'base_reference_loss_std': [],
                    'base_reference_loss_min': [],
                    'base_reference_loss_max': [],
                    'adapted_reference_loss_mean': [],
                    'adapted_reference_loss_std': [],
                    'adapted_reference_loss_min': [],
                    'adapted_reference_loss_max': [],
                    'reference_delta_mean': [],
                    'reference_delta_std': [],
                    'reference_delta_min': [],
                    'reference_delta_max': [],
                    'reference_delta_positive_mean': [],
                    'runtime_per_sample': [],
                },
            }
            for reference_config_id in self.reference_sets
        }
        processed = 0
        seen = 0
        skipped_by_shard = 0
        shard_count = int(self.args.target_shard_count)
        shard_index = int(self.args.target_shard_index)
        try:
            iterator = tqdm(
                data_loader,
                disable=not progress,
                desc='TARR inference',
            )
            for batch in iterator:
                data = batch['data'].cuda(non_blocking=True)
                label = batch['label']
                batch_size = data.size(0)
                for idx in range(batch_size):
                    if max_samples > 0 and processed >= max_samples:
                        break
                    global_sample_index = seen
                    seen += 1
                    if global_sample_index % shard_count != shard_index:
                        skipped_by_shard += 1
                        continue
                    debug_start = len(self.sample_debug)
                    pred, per_reference = self.score_one(net, data[idx:idx + 1])
                    for row in self.sample_debug[debug_start:]:
                        row['target_global_index'] = int(global_sample_index)
                        row['target_shard_count'] = shard_count
                        row['target_shard_index'] = shard_index
                    pred_int = int(pred.detach().cpu().view(-1)[0].item())
                    label_int = int(label[idx].detach().cpu().item())
                    cache_label = (
                        int(cache_label_override)
                        if cache_label_override is not None else label_int)
                    pred_list.append(pred_int)
                    label_list.append(label_int)
                    for reference_config_id, (ood_scores, cache) in per_reference.items():
                        ref_lists = per_reference_lists[reference_config_id]
                        for score_rule, ood_score in ood_scores.items():
                            ref_lists['ood_score'][score_rule].append(
                                float(ood_score.detach().cpu().view(-1)[0].item()))
                        cache['target_global_index'] = int(global_sample_index)
                        if streaming_cache:
                            cache_writers[reference_config_id].add(
                                pred_int, cache_label, cache)
                        else:
                            for key, value in cache.items():
                                ref_lists['cache'].setdefault(key, []).append(value)
                    runtime = next(iter(per_reference.values()))[1][
                        'runtime_per_sample']
                    self._record_target_runtime(runtime)
                    processed += 1
                if max_samples > 0 and processed >= max_samples:
                    break
        finally:
            self._restore_base(net)

        if not pred_list:
            raise RuntimeError('No samples were processed.')
        inference_elapsed = time.perf_counter() - inference_start
        self.timing['inference_total_sec'] += inference_elapsed
        self.timing['processed_count'] += processed
        self.timing['seen_count'] += seen
        self.timing['skipped_by_shard_count'] += skipped_by_shard
        self.timing['inference_calls'].append({
            'processed_count': int(processed),
            'seen_count': int(seen),
            'skipped_by_shard_count': int(skipped_by_shard),
            'elapsed_sec': inference_elapsed,
            'max_samples': int(max_samples or 0),
            'target_shard_count': shard_count,
            'target_shard_index': shard_index,
        })

        pred_array = np.asarray(pred_list, dtype=np.int64)
        label_array = np.asarray(label_list, dtype=np.int64)
        outputs = {}
        for reference_config_id, ref_lists in per_reference_lists.items():
            cache_lists = ref_lists['cache']
            outputs[reference_config_id] = {
                'pred': pred_array,
                'ood_score': {
                    score_rule: np.asarray(values, dtype=np.float64)
                    for score_rule, values in ref_lists['ood_score'].items()
                },
                'label': label_array,
            }
            if streaming_cache:
                continue
            output_cache = {
                'target_global_index': np.asarray(
                    cache_lists['target_global_index'], dtype=np.int64),
                'y_hat': np.asarray(cache_lists['y_hat'], dtype=np.int64),
                'target_conf': np.asarray(cache_lists['target_conf'], dtype=np.float64),
                'target_entropy': np.asarray(cache_lists['target_entropy'], dtype=np.float64),
                'target_probs': np.stack(cache_lists['target_probs']),
                'target_margin': np.asarray(
                    cache_lists['target_margin'], dtype=np.float64),
                'target_energy': np.asarray(
                    cache_lists['target_energy'], dtype=np.float64),
                'perturbation_logit_l2': np.asarray(
                    cache_lists['perturbation_logit_l2'], dtype=np.float64),
                'perturbation_prob_l1': np.asarray(
                    cache_lists['perturbation_prob_l1'], dtype=np.float64),
                'perturbation_conf_delta': np.asarray(
                    cache_lists['perturbation_conf_delta'], dtype=np.float64),
                'perturbation_entropy_delta': np.asarray(
                    cache_lists['perturbation_entropy_delta'], dtype=np.float64),
                'perturbation_response_code': np.asarray(
                    cache_lists['perturbation_response_code'], dtype=np.int64),
                'perturbation_kind_code': np.asarray(
                    cache_lists['perturbation_kind_code'], dtype=np.int64),
                'perturbation_eps': np.asarray(
                    cache_lists['perturbation_eps'], dtype=np.float64),
                'perturbation_repeats': np.asarray(
                    cache_lists['perturbation_repeats'], dtype=np.int64),
                'perturbation_seed': np.asarray(
                    cache_lists['perturbation_seed'], dtype=np.int64),
                'perturbation_cache_policy_code': np.asarray(
                    cache_lists['perturbation_cache_policy_code'], dtype=np.int64),
                'target_tta_loss_before': np.asarray(
                    cache_lists['target_tta_loss_before'], dtype=np.float64),
                'target_tta_loss_after': np.asarray(
                    cache_lists['target_tta_loss_after'], dtype=np.float64),
                'post_tta_pred': np.asarray(
                    cache_lists['post_tta_pred'], dtype=np.int64),
                'post_tta_target_conf': np.asarray(
                    cache_lists['post_tta_target_conf'], dtype=np.float64),
                'post_tta_target_entropy': np.asarray(
                    cache_lists['post_tta_target_entropy'], dtype=np.float64),
                'post_tta_target_probs': np.stack(
                    cache_lists['post_tta_target_probs']),
                'post_tta_pseudo_label_prob': np.asarray(
                    cache_lists['post_tta_pseudo_label_prob'], dtype=np.float64),
                'adapted_target_pred': np.asarray(
                    cache_lists['adapted_target_pred'], dtype=np.int64),
                'adapted_target_conf': np.asarray(
                    cache_lists['adapted_target_conf'], dtype=np.float64),
                'adapted_target_entropy': np.asarray(
                    cache_lists['adapted_target_entropy'], dtype=np.float64),
                'adapted_target_margin': np.asarray(
                    cache_lists['adapted_target_margin'], dtype=np.float64),
                'adapted_target_energy': np.asarray(
                    cache_lists['adapted_target_energy'], dtype=np.float64),
                'adapted_target_probs': np.stack(
                    cache_lists['adapted_target_probs']),
                'target_conf_delta': np.asarray(
                    cache_lists['target_conf_delta'], dtype=np.float64),
                'target_entropy_delta': np.asarray(
                    cache_lists['target_entropy_delta'], dtype=np.float64),
                'target_margin_delta': np.asarray(
                    cache_lists['target_margin_delta'], dtype=np.float64),
                'target_energy_delta': np.asarray(
                    cache_lists['target_energy_delta'], dtype=np.float64),
                'target_pred_changed': np.asarray(
                    cache_lists['target_pred_changed'], dtype=np.int64),
                'base_reference_loss': np.stack(cache_lists['base_reference_loss']),
                'adapted_reference_loss': np.stack(cache_lists['adapted_reference_loss']),
                'delta': np.stack(cache_lists['delta']),
                'reference_conf_delta_by_class': np.stack(
                    cache_lists['reference_conf_delta_by_class']),
                'reference_entropy_delta_by_class': np.stack(
                    cache_lists['reference_entropy_delta_by_class']),
                'reference_margin_delta_by_class': np.stack(
                    cache_lists['reference_margin_delta_by_class']),
                'reference_energy_delta_by_class': np.stack(
                    cache_lists['reference_energy_delta_by_class']),
                'reference_pred_changed_rate_by_class': np.stack(
                    cache_lists['reference_pred_changed_rate_by_class']),
                'reference_correct_rate_before_by_class': np.stack(
                    cache_lists['reference_correct_rate_before_by_class']),
                'reference_correct_rate_after_by_class': np.stack(
                    cache_lists['reference_correct_rate_after_by_class']),
                'base_reference_loss_mean': np.asarray(
                    cache_lists['base_reference_loss_mean'], dtype=np.float64),
                'base_reference_loss_std': np.asarray(
                    cache_lists['base_reference_loss_std'], dtype=np.float64),
                'base_reference_loss_min': np.asarray(
                    cache_lists['base_reference_loss_min'], dtype=np.float64),
                'base_reference_loss_max': np.asarray(
                    cache_lists['base_reference_loss_max'], dtype=np.float64),
                'adapted_reference_loss_mean': np.asarray(
                    cache_lists['adapted_reference_loss_mean'], dtype=np.float64),
                'adapted_reference_loss_std': np.asarray(
                    cache_lists['adapted_reference_loss_std'], dtype=np.float64),
                'adapted_reference_loss_min': np.asarray(
                    cache_lists['adapted_reference_loss_min'], dtype=np.float64),
                'adapted_reference_loss_max': np.asarray(
                    cache_lists['adapted_reference_loss_max'], dtype=np.float64),
                'reference_delta_mean': np.asarray(
                    cache_lists['reference_delta_mean'], dtype=np.float64),
                'reference_delta_std': np.asarray(
                    cache_lists['reference_delta_std'], dtype=np.float64),
                'reference_delta_min': np.asarray(
                    cache_lists['reference_delta_min'], dtype=np.float64),
                'reference_delta_max': np.asarray(
                    cache_lists['reference_delta_max'], dtype=np.float64),
                'reference_delta_positive_mean': np.asarray(
                    cache_lists['reference_delta_positive_mean'], dtype=np.float64),
                'runtime_per_sample': np.asarray(
                    cache_lists['runtime_per_sample'], dtype=np.float64),
                'args_score_rule': self.args.score_rule,
                'reference_config_id': reference_config_id,
                'tta_mode': self.args.tta_mode,
                'response_steps': np.asarray(self.args.response_steps, dtype=np.int64),
                'target_shard_count': shard_count,
                'target_shard_index': shard_index,
                'perturbation_config_id': perturbation_config_id(self.args),
                'perturbation_response': self.args.perturbation_response,
                'perturbation_kind': self.args.perturbation_kind,
                'perturbation_cache_policy': self.args.perturbation_cache_policy,
                'perturbation_eps_config': float(self.args.perturbation_eps),
                'perturbation_repeats_config': int(
                    self.args.perturbation_repeats),
                'perturbation_seed_config': int(self.args.perturbation_seed),
            }
            int_optional = {
                'use_accept_reject_probe',
                'use_anchor_reference',
                'probe_schema_version',
            }
            for key in PROBE_RESPONSE_SCALAR_KEYS:
                if key in cache_lists and cache_lists[key]:
                    dtype = np.int64 if key in int_optional else np.float64
                    output_cache[key] = np.asarray(cache_lists[key], dtype=dtype)
            for key in PROBE_RESPONSE_ARRAY_KEYS:
                if key in cache_lists and cache_lists[key]:
                    output_cache[key] = np.stack(cache_lists[key])
            output_cache.update(probe_response_config_metadata(self.args))
            outputs[reference_config_id].update(output_cache)
        return outputs

    def _record_target_runtime(self, runtime):
        runtime = float(runtime)
        self.timing['runtime_per_target_sum_sec'] += runtime
        current_min = self.timing['runtime_per_target_min_sec']
        current_max = self.timing['runtime_per_target_max_sec']
        if current_min is None or runtime < current_min:
            self.timing['runtime_per_target_min_sec'] = runtime
        if current_max is None or runtime > current_max:
            self.timing['runtime_per_target_max_sec'] = runtime


def metric_summary(id_scores, split_scores):
    pred = np.concatenate([id_scores[0], split_scores[0]])
    conf = np.concatenate([id_scores[1], split_scores[1]])
    label = np.concatenate([
        id_scores[2],
        -1 * np.ones_like(split_scores[2], dtype=np.int64),
    ])
    return compute_all_metrics(conf, label, pred)


def run_inference(postprocessor,
                  net,
                  loader,
                  progress,
                  max_samples,
                  cache_writers=None,
                  cache_label_override=None):
    return postprocessor.inference(
        net,
        loader,
        progress=progress,
        max_samples=max_samples,
        cache_writers=cache_writers,
        cache_label_override=cache_label_override,
    )


def save_debug_csv(path, debug_rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not debug_rows:
        return
    base_fieldnames = [
        'index',
        'reference_config_id',
        'score_rule',
        'primary_score_rule',
        'ood_score',
        'id_confidence',
        'pred',
        'label',
        'split',
        'score_dataset',
        'y_hat',
        'target_conf',
        'target_entropy',
        'runtime_per_sample',
    ]
    extra_fieldnames = sorted({
        key
        for row in debug_rows
        for key in row.keys()
        if key not in base_fieldnames
    })
    fieldnames = base_fieldnames + extra_fieldnames
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(debug_rows):
            output = {'index': idx}
            output.update(row)
            writer.writerow(output)


def attach_metadata_to_debug(debug_rows, labels, split, dataset_name):
    labels = list(labels)
    if not labels:
        return
    refs_per_sample = max(1, len(debug_rows) // len(labels))
    for idx, row in enumerate(debug_rows):
        label = labels[min(idx // refs_per_sample, len(labels) - 1)]
        row['label'] = int(label)
        row['split'] = split
        row['score_dataset'] = dataset_name


def use_sharded_tta_response(args):
    return int(args.tta_response_shard_size) > 0


def make_tta_response_writers(args, scheme_dir, dataset_name):
    if not use_sharded_tta_response(args):
        return None
    writers = {}
    for config in parse_reference_configs(args):
        response_dir = reference_dir(scheme_dir, config.id) / 'tta_response'
        writers[config.id] = TTAResponseShardWriter(
            response_dir,
            dataset_name,
            config.id,
            args,
        )
    return writers


def close_tta_response_writers(writers):
    if not writers:
        return {}
    return {
        reference_config_id: writer.close()
        for reference_config_id, writer in writers.items()
    }


def is_full_run_args(args):
    return (args.max_samples == 0 and args.max_id_samples == 0
            and args.max_ood_samples == 0
            and args.target_shard_count == 1
            and args.target_shard_index == 0)


def require_sharded_tta_response_for_large_full_run(args):
    save_response = args.save_tta_response or args.score_rule in {'all', 'probe_all'}
    if (args.dataset in {'imagenet', 'imagenet200'}
            and is_full_run_args(args)
            and save_response
            and not use_sharded_tta_response(args)):
        raise ValueError(
            'Full ImageNet/ImageNet-200 runs that save tta_response artifacts '
            'must use --tta-response-shard-size > 0. This prevents holding the '
            'full split response in host memory. Example: '
            '--tta-response-shard-size 1024 --debug-output-mode none')


def scoring_config_dict(args):
    return {
        'score_rule_arg': args.score_rule,
        'expanded_score_rules': selected_score_rules(args.score_rule),
        'score_direction': SCORE_DIRECTION,
        'delta_definition': DELTA_DEFINITION,
        'cache_schema_version': CACHE_SCHEMA_VERSION,
        'conf_boundary_transform': 'conf = -ood_score',
        'probe_config': probe_config_dict(args),
    }


def probe_config_id(args):
    if args.tta_mode != 'ar_bank' and not args.use_anchor_reference:
        return 'probe-none_anchor-none'
    anchor = (
        f'anchor-{args.anchor_loss_type}_w{args.anchor_weight:g}'
        if args.use_anchor_reference else 'anchor-none')
    if args.tta_mode == 'ar_bank':
        bank_payload = {
            'accept': list(args.accept_probe_type_bank),
            'reject': list(args.reject_probe_type_bank),
            'primary_accept': args.primary_accept_branch_id,
            'primary_reject': args.primary_reject_branch_id,
        }
        digest = hashlib.sha1(
            json.dumps(bank_payload, sort_keys=True).encode('utf-8')
        ).hexdigest()[:10]
        probe = (
            f'ar-bank-a{len(args.accept_probe_type_bank)}'
            f'-r{len(args.reject_probe_type_bank)}-{digest}')
    else:
        probe = 'ar-none'
    return f'{probe}_{anchor}'.replace('.', 'p')


def probe_config_dict(args):
    config = {
        'ablation_type': args.ablation_type,
        'tta_mode': args.tta_mode,
        'use_accept_reject_probe': bool(args.use_accept_reject_probe),
        'use_anchor_reference': bool(args.use_anchor_reference),
        'anchor_set_root': str(anchor_set_root(args)),
        'anchor_loss_type': args.anchor_loss_type,
        'accept_probe_type': args.accept_probe_type,
        'reject_probe_type': args.reject_probe_type,
        'anchor_weight': float(args.anchor_weight),
        'probe_config_id': probe_config_id(args),
    }
    if args.tta_mode == 'ar_bank':
        config.update({
            'response_bank_schema_version': 1,
            'accept_branch_ids': list(args.accept_branch_ids),
            'accept_branch_probe_types': list(args.accept_probe_type_bank),
            'reject_branch_ids': list(args.reject_branch_ids),
            'reject_branch_probe_types': list(args.reject_probe_type_bank),
            'primary_accept_branch_id': args.primary_accept_branch_id,
            'primary_reject_branch_id': args.primary_reject_branch_id,
        })
    return config


def protocol_config_id(args):
    return f'{args.dataset}_{args.baseline_protocol}_{args.scheme}'


def protocol_config_dict(args, csid_names):
    return {
        'dataset': args.dataset,
        'baseline_protocol': args.baseline_protocol,
        'scheme': args.scheme,
        'reference_source': 'train',
        'resolved_csid_datasets': list(csid_names),
        'fsood_metric_id_side': 'both',
        'near_datasets': args.near_datasets,
        'far_datasets': args.far_datasets,
    }


def artifact_identity_dict(args, checkpoint, postprocessor):
    reference_hashes = {
        ref_id: stats.get('selected_reference_hash')
        for ref_id, stats in postprocessor.reference_stats.items()
    }
    train_metadata = train_candidate_metadata_info(postprocessor)
    reference_set_records = reference_set_info(postprocessor)
    anchor_set_records = anchor_set_info(postprocessor)
    return {
        'dataset': args.dataset,
        'baseline_protocol': args.baseline_protocol,
        'ablation_type': args.ablation_type,
        'checkpoint_resolved': str(Path(checkpoint).resolve()),
        'checkpoint_sha256': file_sha256(checkpoint),
        'model_arch': MODEL_ARCH[args.dataset].__name__,
        'num_classes': NUM_CLASSES[args.dataset],
        'classifier_layer': postprocessor.classifier_name,
        'tta_config_id': tta_config_id(args, postprocessor.runtime_mode),
        'perturbation_config_id': perturbation_config_id(args),
        'perturbation_config': perturbation_config_dict(args),
        'reference_config_ids': [config.id for config in postprocessor.reference_configs],
        'reference_hashes': reference_hashes,
        'train_candidate_metadata_id': train_metadata.get('candidate_id'),
        'train_candidate_metadata_identity': train_metadata.get('identity'),
        'reference_sets': reference_set_records,
        'anchor_sets': anchor_set_records,
        'anchor_stats': postprocessor.anchor_stats,
        'probe_config': probe_config_dict(args),
        'score_direction': SCORE_DIRECTION,
        'delta_definition': DELTA_DEFINITION,
        'cache_schema_version': CACHE_SCHEMA_VERSION,
        'target_shard': target_shard_config_dict(args),
        'is_full_run': is_full_run_args(args),
    }


def evaluate_scheme(args, evaluator, net, postprocessor, output_dir, scheme,
                    data_root):
    progress = not args.no_progress
    scheme_dir = output_dir / scheme
    score_rules = selected_score_rules(args.score_rule)
    scheme_debug_start = len(postprocessor.sample_debug)
    save_response = args.save_tta_response or args.score_rule in {'all', 'probe_all'}

    start_debug = len(postprocessor.sample_debug)
    id_response_writers = (
        make_tta_response_writers(args, scheme_dir, args.dataset)
        if save_response else None)
    id_scores_by_ref = run_inference(
        postprocessor,
        net,
        evaluator.dataloader_dict['id']['test'],
        progress,
        limit_for_split(args, 'id'),
        cache_writers=id_response_writers,
    )
    id_tta_response_files = close_tta_response_writers(id_response_writers)
    first_ref_id = next(iter(id_scores_by_ref))
    id_debug = postprocessor.sample_debug[start_debug:]
    attach_metadata_to_debug(id_debug, id_scores_by_ref[first_ref_id]['label'],
                             'id', args.dataset)

    reference_outputs = {
        ref_id: {
            'processed_counts': {
                'id': {args.dataset: int(id_scores['label'].size)},
                'ood': {'near': {}, 'far': {}},
            },
            'tta_response_files': {'id': {}, 'ood': {'near': {}, 'far': {}}},
            'metric_id_scores': {
                score_rule: score_tuple(id_scores, score_rule)
                for score_rule in score_rules
            },
            'metric_rows': {score_rule: [] for score_rule in score_rules},
        }
        for ref_id, id_scores in id_scores_by_ref.items()
    }
    for ref_id, id_scores in id_scores_by_ref.items():
        ref_dir = reference_dir(scheme_dir, ref_id)
        response_dir = ref_dir / 'tta_response'
        for score_rule in score_rules:
            save_npz(output_score_dir(ref_dir, args, score_rule) /
                     f'{args.dataset}.npz',
                     *score_tuple(id_scores, score_rule))
        if save_response:
            if use_sharded_tta_response(args):
                reference_outputs[ref_id]['tta_response_files']['id'][args.dataset] = (
                    id_tta_response_files[ref_id])
            else:
                save_tta_response(response_dir / f'{args.dataset}.npz', id_scores)
                reference_outputs[ref_id]['tta_response_files']['id'][args.dataset] = str(
                    response_dir / f'{args.dataset}.npz')

    if scheme == 'fsood':
        csid_parts_by_ref = {ref_id: [] for ref_id in id_scores_by_ref}
        for output in reference_outputs.values():
            output['processed_counts']['csid'] = {}
            output['tta_response_files']['csid'] = {}
        for name, loader in evaluator.dataloader_dict['csid'].items():
            start_debug = len(postprocessor.sample_debug)
            csid_response_writers = (
                make_tta_response_writers(args, scheme_dir, name)
                if save_response else None)
            csid_scores_by_ref = run_inference(
                postprocessor,
                net,
                loader,
                progress,
                limit_for_split(args, 'csid'),
                cache_writers=csid_response_writers,
            )
            csid_tta_response_files = close_tta_response_writers(csid_response_writers)
            csid_debug = postprocessor.sample_debug[start_debug:]
            attach_metadata_to_debug(csid_debug,
                                     csid_scores_by_ref[first_ref_id]['label'],
                                     'csid', name)
            for ref_id, csid_scores in csid_scores_by_ref.items():
                ref_dir = reference_dir(scheme_dir, ref_id)
                response_dir = ref_dir / 'tta_response'
                for score_rule in score_rules:
                    save_npz(output_score_dir(ref_dir, args, score_rule) /
                             f'{name}.npz',
                             *score_tuple(csid_scores, score_rule))
                if save_response:
                    if use_sharded_tta_response(args):
                        reference_outputs[ref_id]['tta_response_files']['csid'][name] = (
                            csid_tta_response_files[ref_id])
                    else:
                        save_tta_response(response_dir / f'{name}.npz', csid_scores)
                        reference_outputs[ref_id]['tta_response_files']['csid'][name] = str(
                            response_dir / f'{name}.npz')
                reference_outputs[ref_id]['processed_counts']['csid'][name] = int(
                    csid_scores['label'].size)
                csid_parts_by_ref[ref_id].append(csid_scores)

        for ref_id, csid_parts in csid_parts_by_ref.items():
            if csid_parts:
                reference_outputs[ref_id]['metric_id_scores'] = {
                    score_rule: concat_score_parts(
                        [id_scores_by_ref[ref_id]] + csid_parts, score_rule)
                    for score_rule in score_rules
                }

    split_choices = {
        'near': args.near_datasets,
        'far': args.far_datasets,
    }
    for split in ['near', 'far']:
        split_metrics_by_ref = {
            ref_id: {score_rule: [] for score_rule in score_rules}
            for ref_id in id_scores_by_ref
        }
        loaders = protocol_dataset_items(
            args.dataset,
            split,
            evaluator.dataloader_dict['ood'][split],
            split_choices[split],
        )
        for name, loader in loaders:
            start_debug = len(postprocessor.sample_debug)
            ood_label_override = -1
            ood_response_writers = (
                make_tta_response_writers(args, scheme_dir, name)
                if save_response else None)
            ood_scores_by_ref = run_inference(
                postprocessor,
                net,
                loader,
                progress,
                limit_for_split(args, 'ood'),
                cache_writers=ood_response_writers,
                cache_label_override=ood_label_override,
            )
            ood_tta_response_files = close_tta_response_writers(ood_response_writers)
            ood_debug = postprocessor.sample_debug[start_debug:]
            ood_label = -1 * np.ones_like(
                ood_scores_by_ref[first_ref_id]['label'], dtype=np.int64)
            attach_metadata_to_debug(ood_debug, ood_label, split, name)
            for ref_id, ood_scores in ood_scores_by_ref.items():
                ref_dir = reference_dir(scheme_dir, ref_id)
                response_dir = ref_dir / 'tta_response'
                output = reference_outputs[ref_id]
                for score_rule in score_rules:
                    save_npz(output_score_dir(ref_dir, args, score_rule) /
                             f'{name}.npz',
                             *score_tuple(ood_scores, score_rule, ood_label))
                    metrics = metric_summary(
                        output['metric_id_scores'][score_rule],
                        score_tuple(ood_scores, score_rule))
                    split_metrics_by_ref[ref_id][score_rule].append(metrics)
                    output['metric_rows'][score_rule].append(
                        format_metric_row(name, metrics))
                if save_response:
                    if use_sharded_tta_response(args):
                        output['tta_response_files']['ood'][split][name] = (
                            ood_tta_response_files[ref_id])
                    else:
                        cache_scores = dict(ood_scores)
                        cache_scores['label'] = ood_label
                        save_tta_response(response_dir / f'{name}.npz', cache_scores)
                        output['tta_response_files']['ood'][split][name] = str(
                            response_dir / f'{name}.npz')
                output['processed_counts']['ood'][split][name] = int(
                    ood_scores['label'].size)

        for score_rule in score_rules:
            for ref_id, split_metrics in split_metrics_by_ref.items():
                if split_metrics[score_rule]:
                    mean_metrics = np.mean(np.asarray(split_metrics[score_rule]),
                                           axis=0)
                    reference_outputs[ref_id]['metric_rows'][score_rule].append(
                        format_metric_row(f'{split}ood', mean_metrics))

    all_debug_rows = postprocessor.sample_debug[scheme_debug_start:]
    for ref_id, output in reference_outputs.items():
        ref_dir = reference_dir(scheme_dir, ref_id)
        for score_rule in score_rules:
            write_metrics_csv(output_metrics_path(ref_dir, args, score_rule),
                              output['metric_rows'][score_rule])
        save_debug_csv(output_debug_path(ref_dir, args),
                       [row for row in all_debug_rows
                        if row.get('reference_config_id') == ref_id])
    resolved_names = dataset_names_for_scheme(args, evaluator, scheme)
    train_metadata = train_candidate_metadata_info(postprocessor)
    reference_set_records = reference_set_info(postprocessor)
    anchor_set_records = anchor_set_info(postprocessor)
    timings = timing_info(postprocessor)
    write_json(
        scheme_dir / 'scheme_manifest.json',
        {
            'schema_version': 1,
            'dataset': args.dataset,
            'baseline_protocol': args.baseline_protocol,
            'scheme': scheme,
            'fsood_metric_id_side': 'both' if scheme == 'fsood' else None,
            'score_rule_arg': args.score_rule,
            'ablation_type': args.ablation_type,
            'expanded_score_rules': score_rules,
            'score_direction': SCORE_DIRECTION,
            'delta_definition': DELTA_DEFINITION,
            'cache_schema_version': CACHE_SCHEMA_VERSION,
            'tta_config_id': tta_config_id(args, postprocessor.runtime_mode),
            'perturbation_config_id': perturbation_config_id(args),
            'scoring_config_id': args.score_rule,
            'protocol_config_id': protocol_config_id(args),
            'cache_run_id': args.run_id or default_run_id(args),
            'tta_config': tta_config_dict(
                args, postprocessor.runtime_mode,
                postprocessor.tta_update_impl),
            'perturbation_config': perturbation_config_dict(args),
            'reference_configs': [
                config.to_dict() for config in postprocessor.reference_configs
            ],
            'train_candidate_metadata': train_metadata,
            'train_candidate_metadata_path': train_metadata.get('metadata_dir'),
            'train_candidate_metadata_id': train_metadata.get('candidate_id'),
            'train_candidate_metadata_identity': train_metadata.get('identity'),
            'rebuild_train_candidate_metadata':
            bool(args.rebuild_train_candidate_metadata),
            'reference_set_root': str(reference_set_root(args)),
            'rebuild_reference_set':
            bool(args.rebuild_reference_set),
            'reference_sets': reference_set_records,
            'reference_stats': postprocessor.reference_stats,
            'anchor_set_root': str(anchor_set_root(args)),
            'use_anchor_reference': bool(args.use_anchor_reference),
            'anchor_sets': anchor_set_records,
            'anchor_stats': postprocessor.anchor_stats,
            'probe_config': probe_config_dict(args),
            'scoring_config': scoring_config_dict(args),
            'protocol_config': protocol_config_dict(
                args, resolve_protocol_csid(args.dataset,
                                            args.baseline_protocol)),
            'artifact_identity': artifact_identity_dict(
                args, args.checkpoint or DEFAULT_CHECKPOINT[args.dataset],
                postprocessor),
            'is_full_run': is_full_run_args(args),
            'sample_limits': {
                'max_samples': args.max_samples,
                'max_id_samples': args.max_id_samples,
                'max_ood_samples': args.max_ood_samples,
            },
            'target_shard': target_shard_config_dict(args),
            'batch_sizes': {
                'batch_size': args.batch_size,
                'reference_set_batch_size': args.reference_set_batch_size,
                'train_candidate_batch_size':
                train_candidate_batch_size(args),
            },
            'tta_response': {
                'storage': ('sharded_npz'
                            if use_sharded_tta_response(args) else 'single_npz'),
                'shard_size': int(args.tta_response_shard_size),
                'debug_output_mode': args.debug_output_mode,
            },
            'resolved_dataset_names': resolved_names,
            'dataset_manifest': dataset_manifest(args, data_root,
                                                 resolved_names),
            'reference_config_ids': list(reference_outputs.keys()),
            'processed_counts': {
                ref_id: output['processed_counts']
                for ref_id, output in reference_outputs.items()
            },
            'timing': timings,
            'tta_response_files': {
                ref_id: output['tta_response_files']
                for ref_id, output in reference_outputs.items()
            } if save_response else {},
        },
    )
    return {
        ref_id: output['metric_rows']
        for ref_id, output in reference_outputs.items()
    }


def write_run_info(path, args, checkpoint, postprocessor, elapsed, csid_names):
    path.parent.mkdir(parents=True, exist_ok=True)
    cuda_name = 'unavailable'
    if torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(torch.cuda.current_device())
    tta_config = tta_config_dict(
        args, postprocessor.runtime_mode, postprocessor.tta_update_impl)
    reference_configs = [
        config.to_dict() for config in postprocessor.reference_configs
    ]
    train_metadata = train_candidate_metadata_info(postprocessor)
    reference_set_records = reference_set_info(postprocessor)
    anchor_set_records = anchor_set_info(postprocessor)
    timings = timing_info(postprocessor)
    lines = [
        '# TARR Run',
        '',
        f'- command: {" ".join(sys.argv)}',
        f'- run_id: {args.run_id or default_run_id(args)}',
        f'- experiment_tag: {args.experiment_tag}',
        f'- ablation_type: {args.ablation_type}',
        f'- output_root: {args.output_root}',
        f'- dataset: {args.dataset}',
        f'- baseline_protocol: {args.baseline_protocol}',
        f'- resolved_csid_datasets: {",".join(csid_names)}',
        f'- model_arch: {MODEL_ARCH[args.dataset].__name__}',
        f'- classifier_layer: {postprocessor.classifier_name}',
        f'- checkpoint: {checkpoint}',
        f'- checkpoint_resolved: {Path(checkpoint).resolve()}',
        f'- checkpoint_sha256: {file_sha256(checkpoint)}',
        f'- tta_config_id: {tta_config_id(args, postprocessor.runtime_mode)}',
        f'- tta_config: {json.dumps(tta_config, sort_keys=True)}',
        f'- perturbation_config_id: {perturbation_config_id(args)}',
        f'- perturbation_config: {json.dumps(perturbation_config_dict(args), sort_keys=True)}',
        f'- reference_config_ids: {",".join(config.id for config in postprocessor.reference_configs)}',
        f'- reference_configs: {json.dumps(reference_configs, sort_keys=True)}',
        f'- train_candidate_metadata_path: {train_metadata.get("metadata_dir", "")}',
        f'- train_candidate_metadata_id: {train_metadata.get("candidate_id", "")}',
        f'- train_candidate_metadata_reused: {train_metadata.get("reused", "")}',
        f'- train_candidate_metadata_identity: {json.dumps(train_metadata.get("identity", {}), sort_keys=True)}',
        f'- rebuild_train_candidate_metadata: {args.rebuild_train_candidate_metadata}',
        f'- reference_set_root: {reference_set_root(args)}',
        f'- rebuild_reference_set: {args.rebuild_reference_set}',
        f'- reference_sets: {json.dumps(reference_set_records, sort_keys=True)}',
        f'- reference_stats: {json.dumps(postprocessor.reference_stats, sort_keys=True)}',
        f'- anchor_set_root: {anchor_set_root(args)}',
        f'- use_anchor_reference: {args.use_anchor_reference}',
        f'- anchor_sets: {json.dumps(anchor_set_records, sort_keys=True)}',
        f'- anchor_stats: {json.dumps(postprocessor.anchor_stats, sort_keys=True)}',
        f'- probe_config_id: {probe_config_id(args)}',
        f'- probe_config: {json.dumps(probe_config_dict(args), sort_keys=True)}',
        f'- tta_mode: {args.tta_mode}',
        f'- normal_objective: {args.objective if args.tta_mode == "normal" else ""}',
        f'- accept_probe_types: {",".join(args.accept_probe_type_bank)}',
        f'- reject_probe_types: {",".join(args.reject_probe_type_bank)}',
        f'- steps: {args.steps}',
        f'- lr: {args.lr:g}',
        f'- update_scope: {args.update_scope}',
        f'- freeze_bn_stats: {args.freeze_bn_stats}',
        f'- runtime_mode_arg: {args.runtime_mode}',
        f'- runtime_mode: {postprocessor.runtime_mode}',
        f'- runtime_impl_version: {RUNTIME_IMPL_VERSION}',
        f'- tta_update_impl: {postprocessor.tta_update_impl}',
        f'- reference_feature_cache_enabled: {postprocessor.reference_feature_cache_enabled}',
        f'- target_feature_cache_enabled: {postprocessor.target_feature_cache_enabled}',
        f'- optimizer_reuse_enabled: {postprocessor.tta_update_impl == "reused_torch_optimizer"}',
        f'- tta_response_storage: {"sharded_npz" if use_sharded_tta_response(args) else "single_npz"}',
        f'- tta_response_shard_size: {args.tta_response_shard_size}',
        f'- debug_output_mode: {args.debug_output_mode}',
        f'- score_rule: {args.score_rule}',
        f'- expanded_score_rules: {",".join(selected_score_rules(args.score_rule))}',
        f'- score_direction: {SCORE_DIRECTION}',
        f'- delta_definition: {DELTA_DEFINITION}',
        f'- cache_schema_version: {CACHE_SCHEMA_VERSION}',
        f'- save_tta_response: {args.save_tta_response or args.score_rule in {"all", "probe_all"}}',
        f'- scheme: {args.scheme}',
        '- fsood_metric_id_side: both',
        f'- near_datasets: {args.near_datasets}',
        f'- far_datasets: {args.far_datasets}',
        f'- max_samples: {args.max_samples}',
        f'- max_id_samples: {args.max_id_samples}',
        f'- max_ood_samples: {args.max_ood_samples}',
        f'- target_shard_count: {args.target_shard_count}',
        f'- target_shard_index: {args.target_shard_index}',
        f'- seed: {args.seed}',
        f'- batch_size: {args.batch_size}',
        f'- reference_set_batch_size: {args.reference_set_batch_size}',
        f'- train_candidate_batch_size: {train_candidate_batch_size(args)}',
        f'- num_workers: {args.num_workers}',
        f'- timing: {json.dumps(timings, sort_keys=True)}',
        f'- cuda_available: {torch.cuda.is_available()}',
        f'- cuda_device: {cuda_name}',
        f'- total_runtime_sec: {elapsed:.4f}',
        '',
    ]
    path.write_text('\n'.join(lines))


def main():
    args = parse_args(sys.argv[1:])
    require_sharded_tta_response_for_large_full_run(args)
    set_seed(args.seed)
    wall_timer = time.perf_counter()
    checkpoint = args.checkpoint or DEFAULT_CHECKPOINT[args.dataset]
    run_id = args.run_id or default_run_id(args)
    output_dir = (Path(args.output_root) / 'outputs' / args.dataset /
                  args.baseline_protocol / f'seed{args.seed}' / run_id)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f'Output directory already exists: {output_dir}. '
            'Use a unique --run-id or pass --overwrite.')

    net = build_model(args.dataset)
    net.load_state_dict(load_checkpoint(checkpoint))
    net.cuda()
    net.eval()

    postprocessor = TARRPostprocessor(args, NUM_CLASSES[args.dataset])
    data_root = ROOT_DIR / 'data'
    base_dataloader_factory = evaluator_module.get_id_ood_dataloader
    evaluator_module.get_id_ood_dataloader = with_runtime_dataloader_kwargs(
        base_dataloader_factory)
    try:
        evaluator = Evaluator(
            net,
            id_name=args.dataset,
            data_root=str(data_root),
            config_root=str(ROOT_DIR / 'configs'),
            preprocessor=None,
            postprocessor=postprocessor,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
    finally:
        evaluator_module.get_id_ood_dataloader = base_dataloader_factory
    csid_names = apply_protocol_dataloaders(args, evaluator, data_root)

    inference_timer = time.perf_counter()
    schemes = ['ood', 'fsood'] if args.scheme == 'both' else [args.scheme]
    for scheme in schemes:
        print(f'Running TARR scheme: {scheme}', flush=True)
        evaluate_scheme(args, evaluator, net, postprocessor, output_dir, scheme,
                        data_root)

    inference_elapsed = time.perf_counter() - inference_timer
    elapsed = time.perf_counter() - wall_timer
    postprocessor.timing['driver_inference_wall_sec'] = inference_elapsed
    postprocessor.timing['driver_total_wall_sec'] = elapsed
    write_run_info(output_dir / 'run_info.md', args, checkpoint, postprocessor,
                   elapsed, csid_names)
    train_metadata = train_candidate_metadata_info(postprocessor)
    reference_set_records = reference_set_info(postprocessor)
    anchor_set_records = anchor_set_info(postprocessor)
    timings = timing_info(postprocessor)
    write_json(
        output_dir / 'run_manifest.json',
        {
            'schema_version': 1,
            'command': sys.argv,
            'run_id': run_id,
            'ablation_type': args.ablation_type,
            'dataset': args.dataset,
            'baseline_protocol': args.baseline_protocol,
            'seed': args.seed,
            'schemes': schemes,
            'resolved_csid_datasets': csid_names,
            'output_dir': str(output_dir),
            'output_root': args.output_root,
            'checkpoint': checkpoint,
            'checkpoint_resolved': str(Path(checkpoint).resolve()),
            'checkpoint_sha256': file_sha256(checkpoint),
            'model_arch': MODEL_ARCH[args.dataset].__name__,
            'num_classes': NUM_CLASSES[args.dataset],
            'classifier_layer': postprocessor.classifier_name,
            'tta_config_id': tta_config_id(args, postprocessor.runtime_mode),
            'perturbation_config_id': perturbation_config_id(args),
            'scoring_config_id': args.score_rule,
            'protocol_config_id': protocol_config_id(args),
            'cache_run_id': run_id,
            'tta_config': tta_config_dict(
                args, postprocessor.runtime_mode,
                postprocessor.tta_update_impl),
            'perturbation_config': perturbation_config_dict(args),
            'reference_configs': [
                config.to_dict() for config in postprocessor.reference_configs
            ],
            'train_candidate_metadata': train_metadata,
            'train_candidate_metadata_path': train_metadata.get('metadata_dir'),
            'train_candidate_metadata_id': train_metadata.get('candidate_id'),
            'train_candidate_metadata_identity': train_metadata.get('identity'),
            'rebuild_train_candidate_metadata':
            bool(args.rebuild_train_candidate_metadata),
            'reference_set_root': str(reference_set_root(args)),
            'rebuild_reference_set':
            bool(args.rebuild_reference_set),
            'reference_sets': reference_set_records,
            'reference_stats': postprocessor.reference_stats,
            'anchor_set_root': str(anchor_set_root(args)),
            'use_anchor_reference': bool(args.use_anchor_reference),
            'anchor_sets': anchor_set_records,
            'anchor_stats': postprocessor.anchor_stats,
            'probe_config': probe_config_dict(args),
            'scoring_config': scoring_config_dict(args),
            'protocol_config': protocol_config_dict(args, csid_names),
            'artifact_identity': artifact_identity_dict(
                args, checkpoint, postprocessor),
            'runtime_mode_arg': args.runtime_mode,
            'runtime_mode': postprocessor.runtime_mode,
            'runtime_impl_version': RUNTIME_IMPL_VERSION,
            'tta_update_impl': postprocessor.tta_update_impl,
            'freeze_bn_stats': bool(args.freeze_bn_stats),
            'score_rule_arg': args.score_rule,
            'expanded_score_rules': selected_score_rules(args.score_rule),
            'score_direction': SCORE_DIRECTION,
            'delta_definition': DELTA_DEFINITION,
            'cache_schema_version': CACHE_SCHEMA_VERSION,
            'is_full_run': is_full_run_args(args),
            'sample_limits': {
                'max_samples': args.max_samples,
                'max_id_samples': args.max_id_samples,
                'max_ood_samples': args.max_ood_samples,
            },
            'target_shard': target_shard_config_dict(args),
            'batch_sizes': {
                'batch_size': args.batch_size,
                'reference_set_batch_size': args.reference_set_batch_size,
                'train_candidate_batch_size':
                train_candidate_batch_size(args),
            },
            'tta_response': {
                'storage': ('sharded_npz'
                            if use_sharded_tta_response(args) else 'single_npz'),
                'shard_size': int(args.tta_response_shard_size),
                'debug_output_mode': args.debug_output_mode,
            },
            'timing': timings,
        },
    )
    append_runtime_benchmark_csv(
        Path(args.output_root) / 'summary' / 'runtime_benchmark.csv',
        runtime_benchmark_rows(args, run_id, postprocessor),
    )
    print(f'output_dir: {output_dir}')
    print(f'total_runtime_sec: {elapsed:.4f}')


if __name__ == '__main__':
    main()
