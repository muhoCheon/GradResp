"""Target-only TTA configuration helpers for TARR."""

import hashlib
import json

OBJECTIVES = [
    'predicted_label_ce',
    'entropy',
    'memo_marginal_entropy',
    'view_consistency_kl',
    'view_consistency_js',
    'entropy_consistency',
]
UPDATE_SCOPES = ['classifier', 'all']
RUNTIME_MODES = ['auto', 'full_forward', 'classifier_feature_cache']
PERTURBATION_RESPONSES = ['none', 'pixel', 'feature']
PERTURBATION_KINDS = ['gaussian', 'sign_ce']
PERTURBATION_CACHE_POLICIES = ['auto', 'error_on_feature_cache']
RUNTIME_IMPL_VERSION = 'tarr_runtime_v8_eval_hot_path'


def resolve_runtime_mode(runtime_mode, update_scope):
    if runtime_mode == 'auto':
        return 'classifier_feature_cache' if update_scope == 'classifier' else 'full_forward'
    if runtime_mode == 'classifier_feature_cache' and update_scope != 'classifier':
        raise ValueError(
            'classifier_feature_cache runtime mode requires --update-scope classifier.')
    return runtime_mode


def resolve_tta_update_impl(update_scope):
    return 'reused_torch_optimizer' if update_scope == 'classifier' else 'torch_optimizer'


def tta_config_id(args, resolved_runtime_mode):
    if getattr(args, 'tta_mode', 'normal') == 'ar_bank':
        bank_payload = {
            'accept': list(getattr(args, 'accept_probe_type_bank', [])),
            'reject': list(getattr(args, 'reject_probe_type_bank', [])),
        }
        digest = hashlib.sha1(
            json.dumps(bank_payload, sort_keys=True).encode('utf-8')
        ).hexdigest()[:10]
        objective = (
            f'arbank_a{len(bank_payload["accept"])}'
            f'_r{len(bank_payload["reject"])}_{digest}')
    else:
        objective = {
            'predicted_label_ce': 'plce',
            'entropy': 'ent',
            'memo_marginal_entropy': 'memo',
            'view_consistency_kl': 'vckl',
            'view_consistency_js': 'vcjs',
            'entropy_consistency': 'hcons',
        }.get(args.objective, args.objective)
    lr = f'{args.lr:g}'.replace('.', 'p').replace('-', 'm')
    scope = {'classifier': 'cls', 'all': 'all'}.get(args.update_scope, args.update_scope)
    bn = 'fbn' if args.freeze_bn_stats else 'ubn'
    response_steps = getattr(args, 'response_steps', [args.steps])
    if response_steps == [args.steps]:
        step_tag = f's{args.steps}'
    else:
        step_tag = f's{args.steps}_save{"-".join(str(s) for s in response_steps)}'
    config_id = (
        f'{objective}_{step_tag}_lr{lr}_{scope}_{resolved_runtime_mode}_{bn}'
    )
    return config_id


def tta_config_dict(args, resolved_runtime_mode, update_impl):
    config = {
        'tta_mode': getattr(args, 'tta_mode', 'normal'),
        'objective': args.objective if getattr(args, 'tta_mode', 'normal') == 'normal' else None,
        'steps': args.steps,
        'response_steps': list(getattr(args, 'response_steps', [args.steps])),
        'lr': args.lr,
        'update_scope': args.update_scope,
        'runtime_mode_arg': args.runtime_mode,
        'runtime_mode': resolved_runtime_mode,
        'runtime_impl_version': RUNTIME_IMPL_VERSION,
        'optimizer_policy': update_impl,
        'freeze_bn_stats': bool(args.freeze_bn_stats),
    }
    if getattr(args, 'tta_mode', 'normal') == 'ar_bank':
        config.update({
            'accept_probe_types': list(getattr(args, 'accept_probe_type_bank', [])),
            'reject_probe_types': list(getattr(args, 'reject_probe_type_bank', [])),
        })
    return config


def _float_token(value):
    return f'{value:g}'.replace('.', 'p').replace('-', 'm')


def perturbation_config_id(args):
    response = args.perturbation_response
    if response == 'none':
        return 'pert-none'
    eps = _float_token(args.perturbation_eps)
    kind = args.perturbation_kind
    repeats = args.perturbation_repeats
    seed = args.perturbation_seed
    cache_policy = {
        'auto': 'auto',
        'error_on_feature_cache': 'errcache',
    }.get(args.perturbation_cache_policy, args.perturbation_cache_policy)
    return f'pert-{response}_{kind}_eps{eps}_r{repeats}_seed{seed}_{cache_policy}'


def perturbation_config_dict(args):
    return {
        'response': args.perturbation_response,
        'kind': args.perturbation_kind,
        'eps': args.perturbation_eps,
        'repeats': args.perturbation_repeats,
        'seed': args.perturbation_seed,
        'cache_policy': args.perturbation_cache_policy,
        'enabled': args.perturbation_response != 'none',
    }
