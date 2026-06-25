# TARR Acceptance/Rejection Probe 요약

날짜: 2026-06-02.

이 문서는 TARR Acceptance/Rejection Probe branch의 현재 실험 기준을 정리한다.
목표는 claim 가능한 A/R mechanism 안에서 SOTA에 가까운 성능을 찾는 것이다.
자세한 command log와 full table은 `docs_my/TARR/experiments.md`에 기록되어 있다.

## 짧은 결론

Acceptance/Rejection probing은 유용한 신호를 만들지만, 아직 standalone
SOTA full-spectrum OOD detector라고 보기는 어렵다.

- ImageNet-200에서는 default TARR보다 크게 개선됐다.
- 일부 설정에서는 원하는 ordering을 만든다.

```text
clean ID < csID < semantic OOD
```

- ImageNet-200 near-OOD에는 강하지만, far-OOD는 ASH보다 약하다.
- 현재 테스트한 Anchor reference는 의미 있는 개선을 만들지 못했다.
- 기존 best empirical score는 `reject_efficiency`다.
- 다만 `reject_efficiency`는 target-side 변화가 rejection 쪽에만 들어가는
  비대칭 score다. Claim 가능한 semantic A/R score로는 `accept_efficiency`와
  `ar_efficiency_contrast`도 함께 봐야 한다.
- CIFAR-100 best였던 `topk_ce`/`allclass_ce` 계열은 claim과 ablation이
  어렵기 때문에 새 코드/문서 기본 실험에서 제외한다.

## 실험 상태 용어

| 상태 | 의미 | 실험 운영상 해석 |
| --- | --- | --- |
| 후보 | 연구 아이디어로 제안됨 | 아직 코드나 결과가 없을 수 있음 |
| 구현됨 | CLI와 code path가 존재함 | 실행 가능하지만 성능 검증은 아님 |
| 제한 테스트 | smoke, sanity, 작은 범위 확인 | 성능 판단 근거로는 약함 |
| focused test | 하나의 reference config로 full target coverage 실행 | branch 자체가 유망한지 빠르게 판단 |
| reference-grid test | 15개 reference config 전체로 확장 | reference filter/rpc 최적화 |
| promoted | 다음 실험의 기본 출발점 | 후속 dataset/grid/refseed로 확장할 후보 |
| not promoted | 기본 후보로 쓰지 않음 | 실패, 개선 작음, 또는 claim에 불리함 |

`focused test`는 branch 자체를 보는 단계이고, `reference-grid test`는 유망한
branch에서 reference 설정을 최적화하는 단계다.

## Stage 3: Response Bank 정의

Stage 3는 target sample `x`에 대해 여러 counterfactual update를 수행하고,
그때 Probe reference set `P`와 target `x`가 어떻게 변하는지 저장한다.
Score는 여기서 만들지 않고 Stage 4에서 후처리한다.

### 기본 질문

기존 TARR TTA가 묻는 질문은 다음에 가깝다.

```text
target x에 맞춰 model을 adapt했을 때 reference surface가 얼마나 변하는가?
```

A/R probing은 이 질문을 둘로 나눈다.

```text
Acceptance:
  ID reference surface를 보존하면서 x를 ID로 accept하기 쉬운가?

Rejection:
  ID reference surface를 보존하면서 x를 OOD-like하게 reject하기 쉬운가?
```

각 probe update는 항상 `theta_0`에서 시작한다. Target 하나를 실제로 계속
학습시키는 것이 아니라, accept/reject 방향의 임시 update를 실행하고 response를
측정한 뒤 model을 다시 `theta_0`로 되돌린다.

| 방식 | Target update | Reference response | Model state |
| --- | --- | --- | --- |
| Normal TARR | `x`에 대해 한 번 TTA update | update 전후 `P` 변화 측정 | sample마다 `theta_0`에서 시작 |
| A/R TARR | accept/reject update를 따로 실행 | accept/reject 각각의 `P` 변화 측정 | 각 probe 뒤 `theta_0`로 reset |

### 실행 설정

새 Stage 3 run은 step-wise response-bank 형태로 저장한다.

```text
--tta-mode ar_bank
--steps 30
--save-steps 5,10,30
--update-scope classifier
--lr {1e-2,3e-2}
```

`--steps 30 --save-steps 5,10,30`은 같은 `lr`에서 5/10/30 step run을 따로
돌리는 중복을 없애기 위한 설정이다. 한 번 30 step까지 update하면서 5, 10,
30 step snapshot을 저장하고, Stage 4에서 `--response-step`으로 step을 고른다.

### Objective bank

Acceptance bank는 claim 가능한 TTA objective만 포함한다.

```text
--accept-probe-types predicted_label_ce,entropy_min,view_consistency
```

| Accept branch | Target-side 의미 | Claim 적합성 |
| --- | --- | --- |
| `predicted_label_ce` | pretrained top-1 pseudo label로 CE를 줄인다 | 가장 표준적인 primary 후보 |
| `entropy_min` | target entropy를 줄여 class commitment를 키운다 | TTA literature와 연결됨 |
| `view_consistency` | perturbation view 사이 prediction consistency를 높인다 | consistency-based TTA로 설명 가능 |

다음 acceptance branch는 새 실험과 문서 기본값에서 제외한다.

```text
topk_ce
allclass_ce
```

제외 이유는 성능이 아니라 claim 가능성이다. 이 둘은 top-K/all-class hypothesis
search가 왜 semantic acceptance인지 별도 설명해야 하고 ablation 축도 크게
늘린다.

Semantic rejection bank는 다음 두 개를 기본으로 한다.

```text
--reject-probe-types entropy_max,uniform
```

| Reject branch | Target-side 의미 | Claim 적합성 |
| --- | --- | --- |
| `entropy_max` | target output entropy를 키운다 | semantic rejection primary 후보 |
| `uniform` | target output을 uniform distribution에 가깝게 만든다 | semantic rejection primary 후보 |

`logit_suppression`은 semantic rejection bank에 넣지 않는다. 필요하면 별도
suppressive regularizer ablation으로만 돌린다. 이 branch의 loss는 다음이다.

```text
L_logit_suppression = 0.5 * ||z||_2^2
```

여기서 `z`는 target `x`의 raw logit vector다. 이 loss는 raw logit vector를
원점으로 shrink하는 regularizer다. Common logit offset과 class 간 contrast를
모두 줄이므로 softmax confidence를 낮출 수는 있지만, softmax distribution을
직접 uniform하게 만드는 loss는 아니다. 또한 `logsumexp(z)` 기반 energy score와도
다르다. 성능이 나오더라도 semantic rejection claim이나 energy-based claim의
중심으로 쓰지 않는다.

### Probe reference response

Probe sample `p`의 reference CE loss를 다음처럼 둔다.

```text
ref_loss(theta; p) = CE(f_theta(p.image), p.label)
```

`P_c`는 Probe set `P` 중 label이 class `c`인 sample들의 집합이다.

```text
L_P(theta, c) = mean_{p in P_c} ref_loss(theta; p)
```

Acceptance branch `a`로 update한 뒤 class `c`의 reference loss 변화:

```text
Delta_acc^a(c) =
  L_P(theta_acc^a, c) - L_P(theta_0, c)
```

Rejection branch `r`로 update한 뒤 class `c`의 reference loss 변화:

```text
Delta_rej^r(c) =
  L_P(theta_rej^r, c) - L_P(theta_0, c)
```

`Delta(c)`가 양수면 해당 class의 Probe reference loss가 증가했다는 뜻이고,
음수면 감소했다는 뜻이다. 현재 active spec에서는 이 값을
`ref_loss_delta`라고 부른다.

```text
ref_loss_delta = after-before reference CE loss

accept_ref_loss_delta^a(c) = Delta_acc^a(c)
reject_ref_loss_delta^r(c) = Delta_rej^r(c)
```

`ref_loss_delta`는 class별 raw response field다. Stage 4는 이 vector를
score rule에 맞는 scalar로 바꾼다. 이 문서에서는 그 scalar를
`ref_delta_penalty`라고 부른다. 기본 penalty는 reference CE loss가 증가한
부분만 평균낸 `pos_mean`이다.

```text
ref_delta_penalty(pos_mean) =
  mean_c max(ref_loss_delta(c), 0)
```

하지만 `pos_mean`만 쓰면 clipping이 성능에 필요한지 알 수 없다. 따라서 Stage
4에서는 다음 penalty/scalarization ablation도 함께 본다.

| Penalty mode | Formula | 목적 |
| --- | --- | --- |
| `pos_mean` | `mean_c max(ref_loss_delta(c), 0)` | reference CE가 나빠진 부분만 본다 |
| `signed_mean` | `mean_c ref_loss_delta(c)` | clipping 없이 signed 평균이 충분한지 본다 |
| `abs_mean` | `mean_c abs(ref_loss_delta(c))` | 좋아짐/나빠짐 방향과 무관한 변화 크기를 본다 |
| `pred` | `max(ref_loss_delta(y_hat), 0)` | target의 pretrained predicted class에 해당하는 reference 변화만 본다 |
| `target_weighted` | `sum_c p_theta0(c given x) max(ref_loss_delta(c), 0)` | target posterior가 큰 class의 reference 변화를 더 크게 반영한다 |

과거 positive-loss-increase 이름은 historical result label로만 해석하고, 새
active A/R spec의 canonical field 이름으로 쓰지 않는다.

따라서 A/R response field는 다음 이름을 쓴다.

```text
accept_ref_loss_delta
reject_ref_loss_delta
```

Rejection update가 Probe reference loss를 덜 증가시킬수록 OOD-like하다는
직관은 `ref_delta_penalty`가 efficiency denominator에 들어가는 방식으로
반영한다.

### Target-side response

Target-side response는 선택된 target branch objective의 after-before delta로
정의한다. Reference side와 달리 target side objective는 branch별로 달라질 수
있으므로 canonical 이름은 `target_objective_delta`다.

```text
target_objective_delta = after-before target branch objective

accept_target_objective_delta^a =
  L_accept^a(theta_acc^a, x) - L_accept^a(theta_0, x)

reject_target_objective_delta^r =
  L_reject^r(theta_rej^r, x) - L_reject^r(theta_0, x)
```

`target_objective_delta < 0`이면 해당 accept/reject target branch objective가
update 후 줄었다는 뜻이다. Canonical efficiency는 방향을 맞추기 위해
`-target_objective_delta`를 numerator로 쓴다.

`reject_target_entropy_delta`는 target entropy 증가량 진단 필드다.
Rejection branch의 의미를 해석할 때 참고할 수 있지만, active A/R score의
canonical input이나 efficiency numerator는 아니다.

```text
reject_target_entropy_delta^r =
  H(theta_rej^r, x) - H(theta_0, x)
```

`reject_target_objective_delta`가 더 작을수록 target `x`를 reject objective 방향으로
이동시키기 쉬웠다는 뜻이다.

### Step-wise bank schema

```text
N = target sample 수
S = 저장한 response_steps 수
C = ID class 수
A = acceptance branch 수
R = rejection branch 수
```

Normal TARR/Anchor-only:

```text
response_steps: [S]
delta: [N,S,C]
adapted_reference_loss: [N,S,C]
target_tta_loss_after: [N,S]
adapted_target_conf/entropy/margin/energy: [N,S]
target_conf_delta/entropy_delta/margin_delta/energy_delta: [N,S]
```

A/R response bank:

```text
accept_ref_loss_delta_bank: [N,S,A,C]
reject_ref_loss_delta_bank: [N,S,R,C]
accept_target_objective_delta_bank: [N,S,A]
reject_target_objective_delta_bank: [N,S,R]
reject_target_entropy_delta_bank: [N,S,R]  # diagnostic only, not efficiency numerator
```

Primary singleton fields are also stored from the primary branch:

```text
accept_ref_loss_delta, reject_ref_loss_delta: [N,S,C]
accept_target_objective_delta, reject_target_objective_delta: [N,S]
reject_target_entropy_delta: [N,S]  # diagnostic only, not efficiency numerator
```

### Probe와 Anchor

| 이름 | 의미 | 용도 |
| --- | --- | --- |
| Probe set `P` | 기존 `reference_set` | 각 probe 이후 response 측정 |
| Anchor set `A` | disjoint optional reference | update 중 optional regularization |

의도한 관계는 다음이다.

```text
A ∩ P = ∅
```

현재 결과 기준으로 Anchor CE/distill/param_reg는 no-anchor A/R보다 의미 있는
개선을 만들지 못했다. 따라서 새 성능 탐색에서는 `use_anchor_reference=false`를
기본으로 둔다.

## Stage 4: Score 정의와 Sweep

Stage 4는 저장된 `tta_response`에서 step과 branch를 선택한 뒤 scalar score를
계산한다. 이 섹션이 score rule의 canonical definition이다.

```text
1. response_step 선택
2. accept/reject branch 선택
3. bank field를 primary singleton field 형태로 materialize
4. score rule 계산
```

모든 score 방향은 다음으로 통일한다.

```text
higher score = more OOD-like
```

아래 수식은 선택된 accept branch `a`, reject branch `r`, response step `s`에
대해 쓴다. 표기를 줄이기 위해 step index는 생략한다.

```text
A_ref_delta = accept_ref_loss_delta^a
R_ref_delta = reject_ref_loss_delta^r

A_ref_delta_penalty(mode) = scalarize(A_ref_delta, mode)
R_ref_delta_penalty(mode) = scalarize(R_ref_delta, mode)

A_target_delta = accept_target_objective_delta^a
R_target_delta = reject_target_objective_delta^r
```

기본 score에서 `mode=pos_mean`을 쓴다.

```text
A_ref_delta_penalty(pos_mean) = mean_c max(A_ref_delta(c), 0)
R_ref_delta_penalty(pos_mean) = mean_c max(R_ref_delta(c), 0)
```

### Raw delta diagnostics

| Diagnostic | Formula | 의미 |
| --- | --- | --- |
| acceptance target objective delta | `A_target_delta` | accept target branch objective의 after-before 변화 |
| rejection target objective delta | `R_target_delta` | reject target branch objective의 after-before 변화 |
| acceptance ref penalty | `A_ref_delta_penalty(mode)` | accept update가 만든 scalar reference CE penalty |
| rejection ref penalty | `R_ref_delta_penalty(mode)` | reject update가 만든 scalar reference CE penalty |

### Efficiency score

Efficiency score는 target `x`의 branch-specific target objective가 얼마나
줄었는지를, 그 update가 ID Probe set `P`의 reference CE loss penalty를 얼마나
만들었는지로 나눈 값이다.

```text
efficiency =
  -target_objective_delta / (eps + ref_delta_penalty(mode))
```

Acceptance와 rejection은 다음처럼 대응된다.

```text
accept_efficiency:
  numerator   = -A_target_delta
  denominator = A_ref_delta_penalty(pos_mean)

reject_efficiency:
  numerator   = -R_target_delta
  denominator = R_ref_delta_penalty(pos_mean)
```

| Score | Formula | 의미 |
| --- | --- | --- |
| `accept_efficiency` | `-A_target_delta / (eps + A_ref_delta_penalty(pos_mean))` | positive reference CE penalty를 작게 유지하며 accept target objective를 줄이는 정도 |
| `reject_efficiency` | `-R_target_delta / (eps + R_ref_delta_penalty(pos_mean))` | positive reference CE penalty를 작게 유지하며 reject target objective를 줄이는 정도 |
| `log_reject_efficiency` | `log1p(max(-R_target_delta,0)) - log1p(max(R_ref_delta_penalty(pos_mean),0))` | ratio 대신 log scale로 안정화한 reject efficiency |
| `ar_efficiency_contrast` | `reject_efficiency - accept_efficiency` | reject는 쉽고 accept는 어려우면 OOD-like |

Penalty denominator ablation은 같은 numerator를 유지하고 denominator
scalarization만 바꾼다.

| Penalty mode | Accept score | Reject score | Paired score |
| --- | --- | --- | --- |
| `pos_mean` | `accept_efficiency` | `reject_efficiency` | `ar_efficiency_contrast` |
| `abs_mean` | `accept_abs_ref_efficiency` | `reject_abs_ref_efficiency` | `ar_abs_ref_efficiency_contrast` |
| `pred` | `accept_pred_ref_efficiency` | `reject_pred_ref_efficiency` | `ar_pred_ref_efficiency_contrast` |
| `target_weighted` | `accept_target_weighted_ref_efficiency` | `reject_target_weighted_ref_efficiency` | `ar_target_weighted_ref_efficiency_contrast` |

Log-scale reject efficiency도 같은 penalty mode별로 확인한다.

```text
log_reject_efficiency
log_reject_abs_ref_efficiency
log_reject_pred_ref_efficiency
log_reject_target_weighted_ref_efficiency
```

`signed_mean`은 음수가 될 수 있으므로 efficiency denominator로 쓰지 않는다.
대신 raw delta diagnostic으로만 확인한다.

`reject_efficiency`는 지금까지 가장 유용했던 empirical score다. 하지만
target-side 변화가 rejection에만 들어가므로 최종 semantic A/R claim에는
비대칭이다. `ar_efficiency_contrast`가 비슷하거나 더 좋으면 논리적으로 더
방어 가능한 primary score가 된다.

### Contrast diagnostics

| Score | Formula | 의미 |
| --- | --- | --- |
| `target_objective_delta_contrast` | `A_target_delta - R_target_delta` | reject target objective는 더 줄고 accept target objective는 덜 줄면 OOD-like |
| `ref_loss_delta_contrast` | `A_ref_delta_penalty(pos_mean) - R_ref_delta_penalty(pos_mean)` | accept update가 reject update보다 reference CE penalty를 더 만들면 OOD-like |

과거 positive-reference-delta contrast와 compatibility-style score 이름은 여러
run에서 불안정했다. 새 active spec에서는 canonical A/R score로 promote하지 않고,
필요하면 위 contrast diagnostic으로만 해석한다.

### Stage 4 sweep 범위

Stage 4는 Stage 3보다 싸기 때문에 efficiency score만 보지 않는다. 같은
`tta_response`에서 다음을 모두 계산한다.

| 그룹 | Score |
| --- | --- |
| target-side raw | `accept_target_objective_delta`, `reject_target_objective_delta`, `target_objective_delta_contrast` |
| Probe delta scalarization | `accept_pos_ref_loss_delta_mean`, `reject_pos_ref_loss_delta_mean`, signed/abs/pred/target-weighted variants |
| efficiency with denominator ablation | `accept_efficiency`, `reject_efficiency`, `ar_efficiency_contrast`, abs/pred/target-weighted variants |
| contrast/diagnostic | `target_objective_delta_contrast`, `ref_loss_delta_contrast`, `reject_target_entropy_delta` diagnostics |

현재 구현되어 바로 실행 가능한 score는 다음이다.

```text
accept_efficiency
reject_efficiency
log_reject_efficiency
ar_efficiency_contrast

accept_abs_ref_efficiency
reject_abs_ref_efficiency
log_reject_abs_ref_efficiency
ar_abs_ref_efficiency_contrast

accept_pred_ref_efficiency
reject_pred_ref_efficiency
log_reject_pred_ref_efficiency
ar_pred_ref_efficiency_contrast

accept_target_weighted_ref_efficiency
reject_target_weighted_ref_efficiency
log_reject_target_weighted_ref_efficiency
ar_target_weighted_ref_efficiency_contrast

accept_pos_ref_loss_delta_mean
reject_pos_ref_loss_delta_mean
reject_pos_ref_loss_delta_ood
accept_signed_ref_loss_delta_mean
reject_signed_ref_loss_delta_mean
signed_ref_loss_delta_contrast
accept_abs_ref_loss_delta_mean
reject_abs_ref_loss_delta_mean
abs_ref_loss_delta_contrast
accept_pred_ref_loss_delta
reject_pred_ref_loss_delta
pred_ref_loss_delta_contrast
accept_target_weighted_ref_loss_delta
reject_target_weighted_ref_loss_delta
target_weighted_ref_loss_delta_contrast

accept_target_objective_delta
reject_target_objective_delta
target_objective_delta_contrast
ref_loss_delta_contrast
```

따라서 `--score-rule probe_all`은 efficiency primary뿐 아니라 clipping,
absolute magnitude, predicted-class, target-posterior weighted penalty가
필요한지까지 한 번에 확인하는 broad Stage 4 sweep이다.

## 실행 Command

```bash
# Stage 3: step-wise A/R response bank
python scripts_my/tarr/eval.py run-response \
  --dataset <dataset> \
  --baseline-protocol eval_api \
  --run-id <run_id> \
  --use-prebuilt-reference-set \
  --reference-config <ref_id>:<ref_config> \
  --tta-mode ar_bank \
  --accept-probe-types predicted_label_ce,entropy_min,view_consistency \
  --reject-probe-types entropy_max,uniform \
  --steps 30 \
  --save-steps 5,10,30 \
  --lr <1e-2_or_3e-2> \
  --perturbation-response pixel \
  --perturbation-kind gaussian \
  --perturbation-eps 0.01 \
  --perturbation-repeats 4 \
  --perturbation-seed 0 \
  --update-scope classifier \
  --save-tta-response
```

Stage 4는 FSOOD `both`를 기본으로 실행한다. 즉 clean ID와 csID를 모두 ID
side로 묶어 near/far OOD와 비교한다. `clean`, `csid`는 진단용으로 추가 실행한다.

```bash
# Reject-only
python scripts_my/tarr/cache.py score \
  --run-dir <run_dir> \
  --scheme fsood \
  --reference-config-id <ref_id> \
  --dataset <dataset> \
  --fsood-id-side both \
  --score-rule reject_efficiency \
  --reject-branch all \
  --response-step all

# Accept-only
python scripts_my/tarr/cache.py score \
  --run-dir <run_dir> \
  --scheme fsood \
  --reference-config-id <ref_id> \
  --dataset <dataset> \
  --fsood-id-side both \
  --score-rule accept_efficiency \
  --accept-branch all \
  --response-step all

# Paired A/R
python scripts_my/tarr/cache.py score \
  --run-dir <run_dir> \
  --scheme fsood \
  --reference-config-id <ref_id> \
  --dataset <dataset> \
  --fsood-id-side both \
  --score-rule ar_efficiency_contrast \
  --accept-branch all \
  --reject-branch all \
  --branch-combine cross \
  --response-step all
```

Score result에는 반드시 다음 identity를 함께 기록한다.

```text
dataset
reference_config_id
lr
response_step
accept_branch_id, if used
reject_branch_id, if used
score_rule
```

## 성능 판단 기준

| 기준 | 봐야 하는 값 | 이유 |
| --- | --- | --- |
| full-spectrum 성능 | both/near/far AUROC | near만 좋아지고 far가 무너지면 full-spectrum claim이 약함 |
| clean/csID ordering | clean ID < csID < semantic OOD | TARR가 목표로 하는 csID-aware ordering |
| Group 1 대비 | MSP/KLM/RMDS/ASH best와 gap | 단순 baseline보다 강한지 확인 |
| score 대칭성 | reject-only vs accept-only vs paired | 비대칭 score에 의존하는지 확인 |
| post-hoc ordering | clean/csID/near/far pairwise AUROC와 mean score 순서 | FSOOD AUROC가 좋아도 `clean < csID < nearOOD < farOOD` 의미와 맞는지 확인 |
| reference 민감도 | focused vs reference-grid vs refseed | reference 선택으로만 나온 결과인지 확인 |

## 새 실험 실행 계획

현재 코드와 cache schema는 `target_objective_delta` 기반 response bank로 크게
바뀌었다. 따라서 새 claim-bearing 결과는 과거 Stage 3 artifact를 재사용하지
않고 Stage 3부터 다시 만든다. 과거 결과는 방향성 확인용으로만 사용한다.

목표는 default TARR를 넘는 수준이 아니라, ImageNet-200과 CIFAR-100에서 Group
1 post-hoc baseline best와 비교해도 의미 있는 gap을 만드는 것이다. 따라서
focused run에서도 `both`, `near`, `far`, `clean`, `csID`, Group 1 gap을 모두
본다.

| 순서 | Dataset | Stage 3 | Stage 4 | 다음 단계 조건 |
| --- | --- | --- | --- | --- |
| 0 | CIFAR-10 | `all_rpc8`, tiny smoke | `probe_all`, `response-step all` | cache shape, branch output, score output만 확인 |
| 1 | ImageNet-200 | `correct_rpc32`, fresh response bank | broad score sweep, FSOOD `both/clean/csid` | 완료. Group 1 best를 +3.38pp 넘어서 최우선 follow-up 후보 |
| 2 | ImageNet-200 | `correcthigh09_rpc16` reduced-bank fresh run | 같은 sweep | 완료. 과거 far-biased reference가 새 response bank에서는 `correct_rpc32`보다 낮음 |
| 3 | CIFAR-100 | `correct_rpc32`와 `all_rpc16`, fresh response bank | 같은 sweep | 완료. `all_rpc16`이 Group 1 best를 +0.58pp 넘어서 follow-up 후보 |
| 4 | 유망 dataset/reference | reference-grid fresh run | best score family만 확장 | 완료. CIFAR-100 15-reference follow-up과 ImageNet-200 `correcthigh09_rpc16` follow-up 완료 |
| 5 | 유망 best | refseed robustness fresh run | selected score만 반복 | 논문 표 후보일 때만 실행 |

운영상 주의:

단일 `eval.py run-response` 프로세스로 full FSOOD focused run을 바로 돌리면
GPU 활용이 낮고, `view_consistency`가 포함된 semantic bank에서는 CIFAR-100과
ImageNet-200 모두 수 시간 단위로 늘어날 수 있다. 따라서 새 full run은 다음
순서를 따른다.

```text
1. small max-sample precheck로 command/schema/scoring을 먼저 확인한다.
2. full target coverage는 target-shard runner로 실행한다.
3. 단일 프로세스 full FSOOD run은 runtime baseline 확인 외에는 사용하지 않는다.
```

2026-06-04 기준으로 ImageNet-200/CIFAR-100 `correct_rpc32` small precheck는
통과했다. 각 dataset에서 `max_id_samples=8`, `max_ood_samples=8`,
`steps/save_steps=5/5`로 response bank schema와 Stage 4 `probe_all`
`both/clean/csid` expansion이 정상 동작했다. Claim-bearing AUROC 비교에는
사용하지 않는다.

CIFAR-100에서는 2-shard target-shard precheck도 통과했다. 각 shard를 별도
run으로 만든 뒤 `merge_tta_response_shards.py`로 병합했고, merged cache가
`cache.py validate`와 Stage 4 `probe_all` `both/clean/csid` scoring을 통과했다.
Merged ID response의 확인 shape는 다음과 같다.

```text
accept_ref_loss_delta_bank: [8,1,3,100]
reject_ref_loss_delta_bank: [8,1,2,100]
accept_target_objective_delta_bank: [8,1,3]
reject_target_objective_delta_bank: [8,1,2]
target_shard_index: -1
target_shard_count: 2
score_manifest count after probe_all both/clean/csid: 360
```

따라서 full focused run의 기본 실행 entrypoint는 다음으로 둔다.

```bash
scripts_my/runners/tarr_ar_bank_sharded_focused.sh \
  <cifar100|imagenet200> \
  <ref_id> \
  "<ref_spec>" \
  <shard_count>
```

예:

```bash
scripts_my/runners/tarr_ar_bank_sharded_focused.sh \
  imagenet200 \
  correct_rpc32 \
  "per_class=32,filter=correct,seed=0" \
  8
```

Stage 3 fresh response bank의 기본 축은 다음으로 고정한다.

```text
accept bank = predicted_label_ce, entropy_min, view_consistency
reject bank = entropy_max, uniform
steps/save_steps = 30 / 5,10,30
lr = 1e-2, 3e-2
update_scope = classifier
```

Stage 4 broad sweep은 같은 `tta_response`에서 다음을 모두 계산한다.

```text
target objective delta scores
Probe ref_loss_delta scalarization scores
efficiency scores with denominator ablation
accept-only, reject-only, and accept x reject paired scores
```

Reference-grid 단계에서는 discovery용 full bank를 그대로 쓰지 않는다. Full
bank는 branch 탐색에는 좋지만 reference 15개 전체로 확장하기에는 Stage 3 비용이
너무 크다. Grid follow-up은 focused 결과에서 유망했던 score family에 필요한
branch만 남긴 reduced bank로 실행한다.

```text
Current reduced bank for grid:
  accept bank = predicted_label_ce
  reject bank = uniform

Reason:
  ImageNet-200 current best = accept_efficiency with predicted_label_ce
  CIFAR-100 current best = reject_efficiency with uniform
```

실행 시에는 broad discovery run과 cache가 섞이지 않도록 `RUN_SUFFIX`를 붙인다.

```bash
RUN_SUFFIX=minbank_pce_uniform \
ACCEPT_PROBE_TYPES=predicted_label_ce \
REJECT_PROBE_TYPES=uniform \
scripts_my/runners/tarr_ar_bank_sharded_focused.sh \
  <dataset> <ref_id> "<ref_spec>" 8
```

15-reference wrapper를 사용할 때는 느린 full-tree collector를 run 끝에 붙이지
않도록 `SKIP_COLLECT_SCORE=1`을 사용한다.

추가로 Stage 4-only post-hoc ordering/sign-combination sweep을 수행한다. 이
sweep은 새 Stage 3 response를 만들지 않고, 이미 저장된 score artifact의 부호와
accept/reject 조합만 바꿔 본다.

```text
single score:
  +score
  -score

paired score:
  +A+R, +A-R, -A+R, -A-R
  max/min over all (+/-A, +/-R) sign pairs
```

목적은 다음 ordering을 만드는 수학적 조합이 있는지 확인하는 것이다.

```text
clean ID < csID < nearOOD < farOOD
```

다만 이 sweep은 test split의 clean/csID/near/far label을 보고 부호와 조합을
고르는 diagnostic이다. 따라서 바로 claim-bearing score로 쓰지 않는다. 유망한
조합이 있으면 이후 별도 validation/selection protocol 또는 refseed robustness로
다시 고정해서 검증해야 한다.

진행 기준:

```text
1. CIFAR-10 smoke는 코드 검증용이다. 성능 판단에 쓰지 않는다.
2. ImageNet-200을 최우선으로 본다. 기존 A/R이 near/csID에 강했기 때문이다.
3. ImageNet-200 far가 ASH와 큰 gap을 유지하면 현재 semantic A/R response bank만으로는
   far-OOD가 부족하다고 판단하고, 별도 far-OOD 개선 계획을 세운다.
4. CIFAR-100은 topk/logit-suppression 없이 claim-compatible bank로 재평가한다.
5. Focused 결과가 Group 1을 넘으면 reference-grid follow-up을 완료하기 전에는
   실험 목표를 완료로 보지 않는다.
6. ImageNet-200 far-OOD가 의미 있게 개선되기 전에는 ImageNet-1K를 실행하지 않는다.
```

## Fresh response-bank focused 결과

아래 결과는 새 `target_objective_delta` response-bank schema로 Stage 3부터 다시
만든 fresh run이다. `both avg`는 FSOOD `nearood` AUROC와 `farood` AUROC의
단순 평균이다. Group 1 gap은 같은 dataset의 post-hoc baseline best `both avg`
AUROC와 비교한다. Focused run에서 Group 1을 넘더라도 reference-grid와 refseed
robustness 전에는 최종 SOTA claim으로 보지 않는다.

| Dataset | Reference | Best both score | Step | Branch | Near | Far | Both avg | Group 1 best | Gap | 판단 |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- | ---: | --- |
| CIFAR-100 | `correct_rpc32` | `accept_abs_ref_efficiency` | 5 | accept=`entropy_min` | 63.37 | 65.89 | 64.63 | RMDS 66.20 | -1.57 | default 개선은 보이나 Group 1 best에는 부족 |
| CIFAR-100 | `all_rpc16` | `reject_efficiency` | 30 | reject=`uniform` | 63.76 | 69.81 | 66.78 | RMDS 66.20 | +0.58 | 현재 CIFAR-100 focused 후보 |
| ImageNet-200 | `correct_rpc32` | `accept_efficiency` | 5 | accept=`predicted_label_ce` | 63.16 | 74.68 | 68.92 | SCALE 65.545 | +3.38 | 현재 최우선 focused 후보 |

CIFAR-100 `all_rpc16`는 이 branch에서 처음으로 claim-compatible A/R bank가
Group 1 post-hoc best를 넘은 focused 결과다. 다만 margin이 `+0.58pp`로 작으므로
바로 최종 claim으로 쓰지 않고, reference-grid와 refseed robustness로 재확인한다.

ImageNet-200 `correct_rpc32`는 fresh response bank에서 Group 1 best를
`+3.38pp` 넘었다. Best score가 paired A/R score가 아니라 accept-only
`accept_efficiency`라는 점은 중요하다. 즉 현재 가장 강한 신호는 "ID reference
CE penalty를 작게 유지하면서 predicted-label CE objective를 줄이기 쉬운가"에
있다. 이는 성능상 유망하지만, semantic accept/reject contrast claim을 위해서는
`ar_efficiency_contrast`가 왜 약한지 함께 분석해야 한다.

## Reference follow-up 결과

Focused result가 Group 1을 넘은 뒤에는 reference-grid follow-up이 필요하다.
아래 표는 CIFAR-100 15-reference follow-up과 ImageNet-200 `correcthigh09_rpc16`
follow-up을 완료한 뒤의 결과다. CIFAR-100 추가 reference는 focused 결과에서
필요한 branch만 남긴 reduced bank로 실행했다. Partial shard만 있는 run은
성능 결과로 사용하지 않는다.

완료된 fresh reference follow-up:

| Dataset | Reference | Best FSOOD score | Step | Branch | Near | Far | Both avg | Group 1 best | Gap | 해석 |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- | ---: | --- |
| CIFAR-100 | `all_rpc8` | `target_weighted_ref_loss_delta_contrast` | 30 | accept=`predicted_label_ce`, reject=`uniform` | 64.35 | 64.62 | 64.49 | RMDS 66.20 | -1.72 | `all_rpc16`보다 낮음 |
| CIFAR-100 | `all_rpc16` | `reject_efficiency` | 30 | reject=`uniform` | 63.76 | 69.81 | 66.78 | RMDS 66.20 | +0.58 | 현재 CIFAR-100 best reference |
| CIFAR-100 | `all_rpc32` | `accept_abs_ref_efficiency` | 10 | accept=`predicted_label_ce` | 64.34 | 65.43 | 64.89 | RMDS 66.20 | -1.31 | `all_rpc16`보다 낮음 |
| CIFAR-100 | `correct_rpc8` | `accept_abs_ref_efficiency` | 10 | accept=`predicted_label_ce` | 65.63 | 65.23 | 65.43 | RMDS 66.20 | -0.77 | `all_rpc16`보다 낮지만 `all_rpc8`/`correct_rpc32`보다 좋음 |
| CIFAR-100 | `correct_rpc16` | `reject_efficiency` | 30 | reject=`uniform` | 62.95 | 69.61 | 66.28 | RMDS 66.20 | +0.08 | Group 1을 근소하게 넘지만 `all_rpc16`보다 낮음 |
| CIFAR-100 | `correct_rpc32` | `accept_abs_ref_efficiency` | 5 | accept=`entropy_min` | 63.37 | 65.89 | 64.63 | RMDS 66.20 | -1.57 | Group 1보다 낮음 |
| CIFAR-100 | `highconf09_rpc8` | `accept_abs_ref_efficiency` | 30 | accept=`predicted_label_ce` | 65.17 | 65.54 | 65.36 | RMDS 66.20 | -0.84 | `all_rpc16`보다 낮고 Group 1보다 낮음 |
| CIFAR-100 | `highconf09_rpc16` | `accept_efficiency` | 10 | accept=`predicted_label_ce` | 64.44 | 67.58 | 66.01 | RMDS 66.20 | -0.19 | `highconf09_rpc8`보다 좋지만 `all_rpc16`과 Group 1보다 낮음 |
| CIFAR-100 | `highconf09_rpc32` | `reject_efficiency` | 30 | reject=`uniform` | 64.16 | 65.59 | 64.88 | RMDS 66.20 | -1.32 | `highconf09_rpc16`보다 낮고 Group 1보다 낮음 |
| CIFAR-100 | `correcthigh09_rpc8` | `accept_abs_ref_efficiency` | 30 | accept=`predicted_label_ce` | 65.17 | 65.54 | 65.36 | RMDS 66.20 | -0.84 | Group 1보다 낮고 `all_rpc16`보다 낮음 |
| CIFAR-100 | `correcthigh09_rpc16` | `accept_efficiency` | 10 | accept=`predicted_label_ce` | 64.44 | 67.58 | 66.01 | RMDS 66.20 | -0.19 | `highconf09_rpc16`과 동률이며 `all_rpc16`과 Group 1보다 낮음 |
| CIFAR-100 | `correcthigh09_rpc32` | `reject_efficiency` | 30 | reject=`uniform` | 64.16 | 65.59 | 64.88 | RMDS 66.20 | -1.32 | `highconf09_rpc32`와 동률이며 `all_rpc16`과 Group 1보다 낮음 |
| CIFAR-100 | `strat_rpc8` | `reject_efficiency` | 30 | reject=`uniform` | 63.39 | 70.02 | 66.70 | RMDS 66.20 | +0.50 | Group 1을 넘지만 `all_rpc16`보다 낮음 |
| CIFAR-100 | `strat_rpc16` | `reject_efficiency` | 30 | reject=`uniform` | 64.41 | 66.09 | 65.25 | RMDS 66.20 | -0.95 | `all_rpc16`과 Group 1보다 낮음 |
| CIFAR-100 | `strat_rpc32` | `reject_efficiency` | 30 | reject=`uniform` | 64.89 | 64.34 | 64.62 | RMDS 66.20 | -1.58 | `all_rpc16`과 Group 1보다 낮음 |
| ImageNet-200 | `correct_rpc32` | `accept_efficiency` | 5 | accept=`predicted_label_ce` | 63.16 | 74.68 | 68.92 | SCALE 65.545 | +3.38 | 현재 ImageNet-200 best reference |
| ImageNet-200 | `correcthigh09_rpc16` | `accept_efficiency` | 5 | accept=`predicted_label_ce` | 58.22 | 70.01 | 64.12 | SCALE 65.545 | -1.43 | far-biased reference follow-up은 `correct_rpc32`보다 낮음 |

Clean-only OOD diagnostic:

| Dataset | Reference | Best clean-only score | Step | Branch | Near | Far | Avg | Group 1 clean avg | Gap |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| CIFAR-100 | `all_rpc8` | `target_weighted_ref_loss_delta_contrast` | 30 | accept=`view_consistency`, reject=`uniform` | 80.85 | 80.70 | 80.78 | 81.535 | -0.76 |
| CIFAR-100 | `all_rpc16` | `target_weighted_ref_loss_delta_contrast` | 30 | accept=`view_consistency`, reject=`uniform` | 81.11 | 81.70 | 81.41 | 81.535 | -0.13 |
| CIFAR-100 | `all_rpc32` | `reject_pos_ref_loss_delta_ood` | 30 | reject=`uniform` | 81.17 | 80.22 | 80.69 | 81.535 | -0.85 |
| CIFAR-100 | `correct_rpc8` | `reject_pos_ref_loss_delta_ood` | 30 | reject=`uniform` | 80.93 | 80.70 | 80.81 | 81.535 | -0.72 |
| CIFAR-100 | `correct_rpc16` | `reject_pos_ref_loss_delta_ood` | 30 | reject=`uniform` | 81.18 | 81.81 | 81.50 | 81.535 | -0.04 |
| CIFAR-100 | `correct_rpc32` | `target_weighted_ref_loss_delta_contrast` | 30 | accept=`view_consistency`, reject=`uniform` | 81.20 | 80.14 | 80.67 | 81.535 | -0.87 |
| CIFAR-100 | `highconf09_rpc8` | `reject_pos_ref_loss_delta_ood` | 30 | reject=`uniform` | 81.23 | 79.50 | 80.37 | 81.535 | -1.17 |
| CIFAR-100 | `highconf09_rpc16` | `reject_pos_ref_loss_delta_ood` | 30 | reject=`uniform` | 81.22 | 80.68 | 80.95 | 81.535 | -0.59 |
| CIFAR-100 | `highconf09_rpc32` | `reject_pos_ref_loss_delta_ood` | 30 | reject=`uniform` | 81.40 | 80.61 | 81.00 | 81.535 | -0.54 |
| CIFAR-100 | `correcthigh09_rpc8` | `reject_pos_ref_loss_delta_ood` | 30 | reject=`uniform` | 81.23 | 79.50 | 80.37 | 81.535 | -1.17 |
| CIFAR-100 | `correcthigh09_rpc16` | `reject_pos_ref_loss_delta_ood` | 30 | reject=`uniform` | 81.22 | 80.68 | 80.95 | 81.535 | -0.59 |
| CIFAR-100 | `correcthigh09_rpc32` | `reject_pos_ref_loss_delta_ood` | 30 | reject=`uniform` | 81.40 | 80.61 | 81.00 | 81.535 | -0.54 |
| CIFAR-100 | `strat_rpc8` | `reject_pos_ref_loss_delta_ood` | 30 | reject=`uniform` | 81.07 | 81.59 | 81.33 | 81.535 | -0.21 |
| CIFAR-100 | `strat_rpc16` | `reject_pos_ref_loss_delta_ood` | 30 | reject=`uniform` | 81.24 | 80.45 | 80.84 | 81.535 | -0.70 |
| CIFAR-100 | `strat_rpc32` | `reject_pos_ref_loss_delta_ood` | 30 | reject=`uniform` | 81.36 | 80.07 | 80.72 | 81.535 | -0.82 |
| ImageNet-200 | `correct_rpc32` | `target_weighted_ref_loss_delta_contrast` | 10 | accept=`view_consistency`, reject=`uniform` | 84.91 | 91.84 | 88.38 | 89.41 | -1.03 |
| ImageNet-200 | `correcthigh09_rpc16` | `reject_pos_ref_loss_delta_ood` | 10 | reject=`uniform` | 83.99 | 92.47 | 88.23 | 89.41 | -1.18 |

Reference follow-up notes:

```text
CIFAR-100:
  all_rpc32 minbank_pce_uniform is now a valid reduced-bank merged8 run.
  It is below all_rpc16 and correct_rpc16 for FSOOD both.
  correct_rpc16 minbank_pce_uniform is also a valid reduced-bank merged8 run.
  It barely exceeds Group 1 for FSOOD both, but remains below all_rpc16.
  highconf09_rpc8 minbank_pce_uniform is a valid reduced-bank merged8 run.
  It remains below all_rpc16 and below Group 1 for FSOOD both.
  highconf09_rpc16 minbank_pce_uniform is a valid reduced-bank merged8 run.
  It improves over highconf09_rpc8 but remains below all_rpc16 and Group 1.
  highconf09_rpc32 minbank_pce_uniform is a valid reduced-bank merged8 run.
  It drops below highconf09_rpc16 and remains below all_rpc16 and Group 1.
  correcthigh09_rpc8 minbank_pce_uniform is a valid reduced-bank merged8 run.
  It remains below all_rpc16 and Group 1 for FSOOD both.
  correcthigh09_rpc16 minbank_pce_uniform is a valid reduced-bank merged8 run.
  It matches highconf09_rpc16 and remains below all_rpc16 and Group 1.
  correcthigh09_rpc32 minbank_pce_uniform is a valid reduced-bank merged8 run.
  It matches highconf09_rpc32 and remains below all_rpc16 and Group 1.
  strat_rpc8 minbank_pce_uniform is a valid reduced-bank merged8 run.
  It exceeds Group 1 for FSOOD both, but remains below all_rpc16.
  strat_rpc16 minbank_pce_uniform is a valid reduced-bank merged8 run.
  It remains below all_rpc16 and Group 1.
  strat_rpc32 minbank_pce_uniform is a valid reduced-bank merged8 run.
  It remains below all_rpc16 and Group 1.
  Older all_rpc32 shard-only artifacts are superseded by the merged8 run.

ImageNet-200:
  correcthigh09_rpc16 reduced-bank merged8 run completed and scored.
  It did not improve over correct_rpc32 and remains below the Group 1 best.
  Earlier GPU1-only/broad-bank attempts produced no valid merged score artifact
  and are superseded by the reduced-bank run.
```

따라서 현재 결론은 다음이다.

```text
1. CIFAR-100은 현재 all_rpc16이 all_rpc8/all_rpc32/correct_rpc8/
   correct_rpc16/correct_rpc32/highconf09_rpc8/highconf09_rpc16/
   highconf09_rpc32/correcthigh09_rpc8/correcthigh09_rpc16/
   correcthigh09_rpc32/strat_rpc8/strat_rpc16/strat_rpc32보다 좋다.
   Reference follow-up에서는 all_rpc16이 best로 유지됐다.
   다음 검증은 refseed robustness다.
2. ImageNet-200은 correct_rpc32가 강하고, 과거 far-biased reference인
   correcthigh09_rpc16 reduced-bank follow-up은 이를 넘지 못했다. 이후 grid는
   correct_rpc32 중심으로 reduced bank를 확장한다.
3. Reference-grid follow-up은 완료됐다. SOTA claim 전에는 refseed robustness와
   score-selection protocol을 추가로 확인한다.
```

## Post-hoc ordering/sign-combination 결과

다음 CSV는 fresh response-bank score artifact에서 single score 부호 반전과
accept/reject pair 내부 부호 조합을 모두 평가한 결과다.

```text
results_test/tarr/summary/tarr_probe_ordering_grid.csv
```

현재 sweep은 global FSOOD score leaf만 사용하고 `id_side_clean`, `id_side_csid`
전용 score leaf는 제외한다. Near/far group AUROC는 dataset별 평균이 아니라
해당 group sample을 concatenate해서 계산한 sample-weighted AUROC다.

각 candidate는 다음 값을 기록한다.

```text
clean_csid_auc
csid_near_auc
near_far_auc
ordering_min_auc = min(clean_csid_auc, csid_near_auc, near_far_auc)
fsood_both_avg
group1_gap
```

핵심 결과:

| Dataset/reference | Best ordering candidate | Step | Branch | clean/csID | csID/near | near/far | ordering min | both avg | Group 1 gap | 해석 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ImageNet-200 `correct_rpc32` | `ref_abs:-A-R` | 30 | accept=`predicted_label_ce`, reject=`uniform` | 70.79 | 60.79 | 59.00 | 59.00 | 68.87 | +3.33 | 가장 균형 잡힌 ordering 후보. 다만 near/far가 60 아래라 완전한 ordering score는 아님 |
| ImageNet-200 `correcthigh09_rpc16` | `reject_efficiency` | 30 | reject=`uniform` | 46.47 | 56.78 | 54.84 | 46.47 | 62.70 | -2.85 | far-biased reference follow-up은 ordering과 FSOOD 모두 `correct_rpc32`보다 낮음 |
| CIFAR-100 `all_rpc16` | `accept_abs_ref_loss_delta_mean` | 30 | accept=`view_consistency` | 64.86 | 56.11 | 55.75 | 55.75 | 64.23 | -1.97 | mean ordering은 맞지만 Group 1보다 낮고 csID/near 분리가 약함 |
| CIFAR-100 `correct_rpc32` | `ref_signed:+A+R` | 30 | accept=`predicted_label_ce`, reject=`entropy_max` | 55.11 | 54.33 | 54.48 | 54.33 | 57.58 | -8.62 | ordering과 FSOOD 성능 모두 부족 |

Best FSOOD candidate와 ordering은 분리해서 봐야 한다.

| Dataset/reference | Best FSOOD candidate | Step | Branch | clean/csID | csID/near | near/far | ordering min | both avg | Group 1 gap |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ImageNet-200 `correct_rpc32` | `ref_abs:-A-R` | 10 | accept=`predicted_label_ce`, reject=`uniform` | 72.91 | 61.47 | 57.33 | 57.33 | 69.27 | +3.73 |
| ImageNet-200 `correcthigh09_rpc16` | `eff_abs:+A+R` | 10 | accept=`predicted_label_ce`, reject=`uniform` | 44.92 | 53.67 | 56.23 | 44.92 | 64.70 | -0.84 |
| CIFAR-100 `all_rpc16` | `-reject_signed_ref_loss_delta_mean` | 5 | reject=`uniform` | 77.85 | 50.80 | 57.77 | 50.80 | 66.91 | +0.71 |
| CIFAR-100 `correct_rpc32` | `target_objective:+A+R` | 10 | accept=`entropy_min`, reject=`uniform` | 72.61 | 54.96 | 51.24 | 51.24 | 65.21 | -0.99 |

해석:

- ImageNet-200에서는 sign-combination이 기존 best보다 약간 더 좋은 focused
  FSOOD result를 만든다. `ref_abs:-A-R`는 accept/reject update가 만든 absolute
  reference CE 변화가 작을수록 OOD-like하게 보는 조합이다. 성능은 좋지만, 이것은
  semantic A/R efficiency라기보다 reference stability score에 가깝다.
- ImageNet-200 `correcthigh09_rpc16`은 과거 far-biased reference였지만, 새
  reduced-bank follow-up에서는 direct score와 post-hoc score 모두 Group 1과
  `correct_rpc32`를 넘지 못했다.
- CIFAR-100 `all_rpc16`에서는 Group 1을 넘는 candidate가 있지만 csID/near
  AUROC가 약 50.8로 거의 랜덤이다. 즉 현재 response bank는 CIFAR-100에서
  `clean`과 `farOOD`를 주로 분리하고, `csID`와 `nearOOD`를 안정적으로 나누지는
  못한다.
- 네 score set 모두 `clean/csID`, `csID/near`, `near/far` pairwise AUROC가 전부
  60 이상인 candidate는 없었다. 따라서 현재 scoring 조합만으로는 강한
  `clean < csID < nearOOD < farOOD` ordering을 만들었다고 보기 어렵다.

## 과거 실험 결과

### CIFAR-10

역할: sanity/control.

확인한 claim-compatible A/R branch는 `predicted_label_ce + entropy_max`,
`view_consistency + entropy_max`, `entropy_min + entropy_max` 등이다.

| Setting | Score | both AUROC | Decision |
| --- | --- | ---: | --- |
| default TARR | `positive_loss_increase_mean` | 78.02 | baseline |

CIFAR-10만 보면 A/R을 promote할 근거가 부족했다.

### CIFAR-100

기존 실험에서 CIFAR-100의 가장 좋은 A/R 성능은 `topk_ce + logit_suppression`
계열에서 나왔다. 그러나 `topk_ce`/`allclass_ce`는 claim과 ablation이 어렵다.
또한 `logit_suppression`은 semantic rejection이나 energy score가 아니라 raw
logit vector를 원점으로 shrink하는 simple suppressive regularizer다. 따라서 이
조합은 새 claim-bearing 실험 후보에서 제외한다.

기존 baseline:

| Setting | Score | both AUROC | clean AUROC | csID AUROC |
| --- | --- | ---: | ---: | ---: |
| default TARR | `positive_loss_increase_mean` | 62.00 | 77.14 | 48.38 |

새 claim-compatible fresh run에서는 `all_rpc16`이 both avg AUROC 66.78을 달성해
Group 1 best RMDS 66.20을 `+0.58pp` 넘었다. 기존 best가 claim하기 어려운
branch였기 때문에, 새 기준의 CIFAR-100 follow-up은 `all_rpc16` fresh bank에서
시작한다.

### ImageNet-200

유용했던 claim-compatible A/R branch:

```text
Acceptance: predicted_label_ce
Rejection: entropy_max
Score: reject_efficiency
```

Focused `correct_rpc32` 결과:

| Setting | Score | both AUROC | Delta vs default | clean AUROC | csID AUROC | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| default TARR | `positive_loss_increase_mean` | 53.72 | +0.00 | 82.09 | 46.20 | baseline |
| A/R predicted CE + entropy max | `reject_efficiency` | 63.24 | +9.53 | 74.12 | 60.38 | strong gain, clean drops |

저장된 15-reference grid best:

| Ref | Score | both avg | near | far | Ordering |
| --- | --- | ---: | ---: | ---: | --- |
| `correct_rpc32` | `reject_efficiency` | 63.24 | 64.52 | 61.97 | pass |
| `correcthigh09_rpc16` | `reject_efficiency` | 61.88 | 58.97 | 64.78 | pass, far-biased |

Group 1 비교:

| Aggregate | TARR AUROC | Best Group 1 | Group 1 AUROC | Gap |
| --- | ---: | --- | ---: | ---: |
| near-OOD | 64.52 | KLM | 57.26 | +7.26 |
| far-OOD | 64.78 | ASH | 75.48 | -10.70 |

해석:

ImageNet-200 A/R은 near-OOD와 csID ordering을 개선한다. 남은 문제는 far-OOD다.
현재 entropy-based rejection response는 ASH 계열의 far-OOD robustness를
따라가지 못한다.

새 fresh response-bank run에서는 `accept_efficiency`가 both avg AUROC 68.92를
달성했다. 이는 Group 1 post-hoc best SCALE 65.545보다 높고, far AUROC도 74.68로
과거 ASH far 75.48에 근접했다. 따라서 ImageNet-200은 `correct_rpc32`에서
reference-grid와 refseed robustness로 바로 확장할 가치가 있다. 단, best score가
accept-only이므로 최종 논문 claim에서는 paired A/R mechanism보다 acceptance-side
reference-preserving TTA response로 해석하는 것이 더 정확할 수 있다.

## 결정

새 실험에서 고정할 것:

```text
Claim-compatible accept bank:
  predicted_label_ce, entropy_min, view_consistency

Semantic reject bank:
  entropy_max, uniform

Stage 3:
  fresh response bank from theta_0
  steps/save_steps = 30 / 5,10,30
  lr = 1e-2, 3e-2
  update_scope = classifier

Stage 4:
  probe_all broad sweep
  response-step all
  FSOOD id side = both, plus clean/csid diagnostics

Primary comparison:
  default TARR
  Group 1 post-hoc best

Current CIFAR-100 focused candidate:
  all_rpc16 + reject_efficiency + uniform rejection at response_step=30

Reason:
  fresh claim-compatible response bank reaches both avg AUROC 66.78,
  exceeding CIFAR-100 Group 1 RMDS 66.20 by +0.58pp.

Current ImageNet-200 focused candidate:
  correct_rpc32 + accept_efficiency + predicted_label_ce at response_step=5

Reason:
  fresh claim-compatible response bank reaches both avg AUROC 68.92,
  exceeding ImageNet-200 Group 1 SCALE 65.545 by +3.38pp.
```

새 실험에서 기본 후보로 쓰지 않을 것:

```text
topk_ce/allclass_ce acceptance
logit_suppression as semantic rejection or energy claim
direct positive-reference-delta contrast as primary score
anchor-only CE/distill/param_reg
A/R + Anchor CE in current full-batch form
discarded Stage 4 calibration-heavy score-construction variants
```
