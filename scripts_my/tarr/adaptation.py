"""Target-only TTA configuration helpers for TARR."""

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
    config_id = (
        f'{objective}_s{args.steps}_lr{lr}_{scope}_{resolved_runtime_mode}_{bn}'
    )
    return config_id


def tta_config_dict(args, resolved_runtime_mode, update_impl):
    return {
        'objective': args.objective,
        'steps': args.steps,
        'lr': args.lr,
        'update_scope': args.update_scope,
        'runtime_mode_arg': args.runtime_mode,
        'runtime_mode': resolved_runtime_mode,
        'runtime_impl_version': RUNTIME_IMPL_VERSION,
        'optimizer_policy': update_impl,
        'freeze_bn_stats': bool(args.freeze_bn_stats),
    }


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
