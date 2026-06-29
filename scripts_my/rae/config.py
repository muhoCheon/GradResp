"""Configuration constants and CLI helpers for standalone RAE."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from openood.networks import ResNet18_224x224, ResNet18_32x32, ResNet50

ROOT_DIR = Path(__file__).resolve().parents[2]

SUPPORTED_DATASETS = ('cifar10', 'cifar100', 'imagenet', 'imagenet200')
SUPPORTED_SCHEMES = ('ood', 'fsood')

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

REFERENCE_FILTERS = (
    'all',
    'correct',
    'high_confidence',
    'correct_high_confidence',
)
GRADIENT_SPACES = ('classifier', 'last_block', 'all')
CANDIDATE_MODES = ('all', 'pred')
REJECTION_RULES = ('off', 'uniform')
VALIDATION_RULES = (
    'pairwise_rank',
    'pairwise_margin',
    'same_mean',
    'mean_margin',
    'soft_margin',
)
DEFAULT_VALIDATION_RULE = 'pairwise_rank'
DEFAULT_VALIDATION_TEMPERATURE = 0.1
DEFAULT_REJECTION_RULE = 'off'
DEFAULT_REJECTION_POWER = 1.0
SCORE_RULES = (
    'neglog_eid',
    'neg_eid',
    'geom_effdim_mean',
    'geom_rawnorm_mean',
    'geom_cos_pred_mean',
    'geom_proto_cos_max',
    'geom_proto_eid',
)
CLI_DEFAULT_SCORE_RULES = (
    'neglog_eid',
    'neg_eid',
    'geom_effdim_mean',
    'geom_rawnorm_mean',
    'geom_cos_pred_mean',
)
DEFAULT_SCORE_RULES = ('neglog_eid', 'neg_eid')
DEFAULT_REFERENCE_PER_CLASS_GRID = (4, 8, 16, 32, 64)
DEFAULT_REFERENCE_PER_CLASS_GRID_ARG = ','.join(
    str(value) for value in DEFAULT_REFERENCE_PER_CLASS_GRID)

NUMERIC_EPS = 1e-12
CACHE_SCHEMA_VERSION = 2
SCORE_DIRECTION = 'higher_is_ood'


@dataclass(frozen=True)
class ReferenceConfig:
    dataset: str
    per_class: int
    filter_name: str
    min_confidence: float
    seed: int

    @property
    def id(self) -> str:
        min_conf = str(self.min_confidence).replace('.', 'p')
        if self.filter_name in {'high_confidence', 'correct_high_confidence'}:
            return f'{self.filter_name}_conf{min_conf}_rpc{self.per_class}'
        return f'{self.filter_name}_rpc{self.per_class}'


def parse_csv_values(value: str | Iterable[str]) -> List[str]:
    items = value.split(',') if isinstance(value, str) else value
    return [str(item).strip() for item in items if str(item).strip()]


def parse_score_rules(value: str | Iterable[str] | None) -> List[str]:
    if value is None:
        rules = list(DEFAULT_SCORE_RULES)
    else:
        rules = parse_csv_values(value)
    if not rules:
        raise ValueError('At least one score rule must be selected')
    unknown = sorted(set(rules) - set(SCORE_RULES))
    if unknown:
        raise ValueError(f'Unknown RAE score rule(s): {unknown}')
    return rules


def make_run_id(args) -> str:
    ref = ReferenceConfig(
        dataset=args.dataset,
        per_class=args.reference_per_class,
        filter_name=args.reference_filter,
        min_confidence=args.reference_min_confidence,
        seed=args.reference_seed,
    )
    candidate = args.candidate_mode
    subset = ''
    if getattr(args, 'max_target_samples', None):
        subset = f'_subset{int(args.max_target_samples)}'
    validation_rule = getattr(args, 'validation_rule', DEFAULT_VALIDATION_RULE)
    validation = ''
    if validation_rule != DEFAULT_VALIDATION_RULE:
        validation = f'_val{validation_rule}'
        if validation_rule == 'soft_margin':
            temperature = str(
                getattr(args, 'validation_temperature',
                        DEFAULT_VALIDATION_TEMPERATURE)).replace('.', 'p')
            validation += f'_t{temperature}'
    rejection_rule = getattr(args, 'rejection_rule', DEFAULT_REJECTION_RULE)
    rejection = ''
    if rejection_rule != DEFAULT_REJECTION_RULE:
        power = str(
            getattr(args, 'rejection_power',
                    DEFAULT_REJECTION_POWER)).replace('.', 'p')
        rejection = f'_rej{rejection_rule}_b{power}'
    run_id = (
        f'rae_{args.gradient_space}_{ref.id}_{candidate}'
        f'{validation}{rejection}_refseed{ref.seed}{subset}'
    )
    experiment_id = getattr(args, 'experiment_id', None)
    if experiment_id:
        return f'{experiment_id}_{run_id}'
    return run_id
