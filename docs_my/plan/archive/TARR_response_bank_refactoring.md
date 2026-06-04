# TARR A/R Response Bank 리팩터링 계획

## Summary

A/R Stage 3를 현재의 “accept 1개 + reject 1개 조합 run”에서 “accept bank + reject bank 생성” 구조로 바꾼다. Stage 3는 각 branch response를 독립 저장하고, Stage 4는 accept-only, reject-only, accept x reject cross-product score를 후처리로 계산한다. 기존 singleton A/R cache와 일반 TARR cache는 계속 scoring 가능해야 한다.

## Key Changes

- Stage 3 CLI:
  - 추가: `--accept-probe-types predicted_label_ce,entropy_min,view_consistency`
  - 추가: `--reject-probe-types entropy_max,uniform,logit_suppression`
  - 기존 `--accept-probe-type`, `--reject-probe-type`는 singleton alias로 유지한다.
  - plural 옵션이 있으면 bank mode, 없으면 기존 singleton mode처럼 동작한다.
  - branch id는 v1에서 probe type 문자열을 그대로 사용한다. 중복 probe type은 허용하지 않는다.
  - `topk_ce`, `allclass_ce`는 accept bank에 포함하지 않는다.

- Stage 3 cache schema:
  - 기존 primary fields는 유지한다:
    - `accept_delta`, `reject_delta`: `[N,S,C]`
    - `accept_damage`, `reject_damage`: `[N,S]`
    - `accept_target_gain`, `reject_target_entropy_delta`: `[N,S]`
  - 새 bank fields를 추가한다:
    - `accept_delta_bank`: `[N,S,A,C]`
    - `reject_delta_bank`: `[N,S,R,C]`
    - `accept_damage_bank`, `accept_target_gain_bank`: `[N,S,A]`
    - `reject_damage_bank`, `reject_target_entropy_delta_bank`: `[N,S,R]`
    - `accept_branch_ids`, `accept_branch_probe_types`: `[A]`
    - `reject_branch_ids`, `reject_branch_probe_types`: `[R]`
    - `primary_accept_branch_id`, `primary_reject_branch_id`
    - `response_bank_schema_version = 1`
  - step axis는 항상 axis 1로 둔다. `--response-step` 선택 후 bank fields는 `[N,A,C]`, `[N,R,C]`, `[N,A]`, `[N,R]`가 된다.

- Stage 3 implementation:
  - `_target_probe_loss()`가 `self.args.accept_probe_type` / `reject_probe_type`를 직접 읽지 않고 `probe_type` 인자를 받도록 바꾼다.
  - `_run_update()`에 `probe_type`과 `branch_id`를 넘긴다.
  - A/R mode에서는 accept branches를 각각 `theta_0`에서 실행하고, reject branches도 각각 `theta_0`에서 실행한다.
  - primary legacy fields는 primary branch에서 뽑는다. primary branch 기본값은 각 bank의 첫 번째 branch다.
  - `response_steps`와 `--save-steps` 동작은 그대로 유지한다.
  - `probe_config_id`는 bank mode에서 deterministic serialization hash를 포함한다. singleton mode의 기존 id는 유지한다.

- Stage 4 CLI:
  - 추가: `--accept-branch auto|all|legacy|<name_or_index>[,...]`
  - 추가: `--reject-branch auto|all|legacy|<name_or_index>[,...]`
  - 추가: `--branch-combine cross|zip`, 기본 `cross`
  - legacy cache에서는 기존 output layout을 유지한다.
  - bank cache에서 `auto`는 score rule에 따라 필요한 branch를 모두 평가한다:
    - accept-only score: 모든 accept branch
    - reject-only score: 모든 reject branch
    - paired A/R score: accept x reject cross-product

- Stage 4 scoring:
  - 기존 score formula는 유지한다.
  - branch selector가 bank field에서 선택 branch를 legacy field 형태로 materialize한 뒤 기존 formula를 재사용한다.
  - `reject_efficiency`는 reject branch만 사용한다.
  - `accept_efficiency`는 accept branch만 사용한다.
  - `ar_efficiency_contrast`, `damage_contrast`, `id_compatibility`는 accept/reject pair를 사용한다.
  - branch-bank output layout:
    - `score_results/<rule>/step_<k>/accept_<a>/...`
    - `score_results/<rule>/step_<k>/reject_<r>/...`
    - `score_results/<rule>/step_<k>/accept_<a>__reject_<r>/...`
  - `--response-step final`, integer step, `all`은 기존처럼 동작한다.

## Subagent Breakdown

- Subagent A: Stage 3 Worker
  - 담당: `eval.py`, `adaptation.py`
  - 구현: plural probe CLI, branch parsing, branch-bank Stage 3 loop, bank cache fields, primary legacy fields, probe config metadata.
  - 주의: 다른 worker 변경을 되돌리지 않는다.

- Subagent B: Stage 4 Worker
  - 담당: `scoring.py`, `cache.py`
  - 구현: branch selector parser, bank field validation, response-step slicing for bank fields, branch materialization, branch-specific output dirs.
  - 주의: legacy v5/singleton cache scoring 결과를 바꾸지 않는다.

- Subagent C: Runner + Docs Worker
  - 담당: runner scripts, `docs_my/TARR/*`, `docs_my/research/TARR_acceptance_rejection_summary.md`
  - 구현: `--accept-probe-types`, `--reject-probe-types`, `--accept-branch`, `--reject-branch`, cross-product scoring 예시 반영.
  - 문서에서는 claim 가능한 accept probes를 `predicted_label_ce`, `entropy_min`, `view_consistency` 중심으로 설명한다.

- Subagent D: Verification Worker
  - 담당: smoke tests and shape checks
  - 확인: normal TARR legacy, singleton A/R legacy, banked A/R, Stage 4 branch selection, `--response-step all`.
  - GPU가 필요한 Stage 3 smoke는 sandbox 밖 CUDA 접근으로 실행한다.

## Test Plan

- Static:
  - `python -m py_compile scripts_my/tarr/eval.py scripts_my/tarr/cache.py scripts_my/tarr/scoring.py scripts_my/tarr/adaptation.py`
  - `bash -n` for updated runner scripts
  - `git diff --check`

- Legacy compatibility:
  - 기존 singleton A/R smoke:
    - `--accept-probe-type predicted_label_ce --reject-probe-type entropy_max`
    - `--steps 3 --save-steps 1,2,3`
  - 기존 score output path와 shape 유지 확인.
  - old cache에서 `--response-step final`, `1`, `all` scoring 확인.

- Banked Stage 3 smoke:
  - CIFAR-10 small smoke:
    - `--accept-probe-types predicted_label_ce,entropy_min`
    - `--reject-probe-types entropy_max,uniform`
    - `--steps 3 --save-steps 1,2,3`
  - shape assert:
    - `accept_delta_bank`: `[N,3,2,C]`
    - `reject_delta_bank`: `[N,3,2,C]`
    - scalar banks: `[N,3,2]`
    - primary fields: `[N,3,C]`, `[N,3]`

- Stage 4 bank scoring:
  - `--score-rule reject_efficiency --reject-branch all --response-step all`
  - `--score-rule accept_efficiency --accept-branch all --response-step all`
  - `--score-rule ar_efficiency_contrast --accept-branch all --reject-branch all --branch-combine cross --response-step all`
  - output dirs include branch tags and do not overwrite each other.
  - branch-name mismatch across ID/OOD caches fails loudly.

## Assumptions

- v1 branch id는 probe type 문자열이다. 같은 role 안에서 중복 probe type은 금지한다.
- `response_steps`는 유일한 step axis이며, bank axis는 step axis 뒤에 둔다.
- 기존 primary fields는 backward compatibility와 기존 score path를 위해 계속 저장한다.
- `logit_suppression`은 reject bank에 남길 수 있지만, 문서에서는 semantic rejection claim이 아니라 evidence/energy suppression branch로 분리한다.
- `topk_ce`, `allclass_ce`는 새 accept bank와 문서/runner 기본 실험에서 제외한다.
