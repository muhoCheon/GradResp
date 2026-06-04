# 목표: TARR을 **reference-response 기반 standalone full-spectrum OOD detector**로 발전시키는 것

Acceptance/Rejection Probe 실험의 짧은 정리는
`docs_my/research/TARR_acceptance_rejection_summary.md`를 먼저 볼 것.

### `clean ID / csID`는 낮게, `semantic OOD`는 높게 scoring

## **세부 목표**

- CIFAR-10/CIFAR-100/ImageNet-200/ImageNet-1K에서 TARR 다른 Post hoc 방법들 (현재 repo 기준 group1 method)과 비교하여 competitive한 성능을 가지도록 함
- TARR score가 semantic novelty를 측정하도록 하는 TTA / reference / score 설정 방법을 찾아야 함.

    ```
    Desired ordering:
    clean ID < csID << semantic OOD
    ```


---

## **현재 상황**

ImageNet-R/C에 대하여 pretrained model의 정확도가 낮음 → confidence 기준 pseudo label을 사용한 TTA를 수행하면 wrong pseudo-label을 과신하도록 update 될 수 있음.

현재 TARR는 semantic novelty 자체보다, target TTA update가 clean reference surface를 얼마나 흔드는지를 측정하는 경향이 있을 수 있음.

따라서 ImageNet-R/C처럼 accuracy가 낮은 csID는 semantic OOD보다 더 disruptive하게 보일 수 있음.

---

# 새롭게 시도해볼 방향 1: Acceptance/Rejection Probe

```
target sample을 먼저 ID/OOD로 결정하지 않는다.

대신 같은 target sample에 대해 두 가지 가정을 각각 따로 시험한다.

1. ID라고 가정하고 잠깐 update 해본다. (Acceptance Probe)
2. OOD/reject라고 가정하고 잠깐 update 해본다. (Rejection Probe)

각 update 후 reference response를 측정하고,
어느 가정이 reference surface와 더 잘 맞는지 비교한다.

각 가정의 실험이 끝나면 model은 항상 원래 θ0로 되돌린다.

최종 OOD score는 Acceptance Probe와 Rejection Probe의
reference compatibility contrast로 정의한다. (?)
```

#### **Acceptance Probe**

target sample을 known ID label space 안으로 받아들인다고 가정하고 수행하는 probe.

entropy minimization / class-conditional CE / view consistency 방향으로 가상 update를 수행한다.
그 후 reference response가 ID-compatible한지 측정한다.

단, Acceptance Probe는 predicted label을 그대로 신뢰한다는 뜻이 아니다.
Target이 어떤 ID class explanation과 양립 가능한지 확인하는 probe이다.

**의미**: "이 sample을 ID로 받아들여도 reference surface가 자연스럽게 유지되는가?"

#### **Rejection Probe**:

target sample을 known ID label space로 받아들이지 않는다고 가정하고 수행하는 probe.

entropy maximization / uniformization / logit suppression 방향으로 가상 update를 수행한다.
그 후 ID reference surface를 얼마나 적게 손상시키는지 측정한다.

**의미**: "이 sample을 OOD로 보고 reject하는 것이 reference surface와 더 잘 맞는가?"

---

# 새롭게 시도해볼 방향 2: Probe/Anchor Reference

1. **Probe / Measurement Reference `P`  (기존 TARR에 이미 구현된 reference data)**

    Adaptation 전후 reference response를 측정하는 기준면이다.

    최종 TARR score는 `P`에서의 class-wise loss/logit 변화를 기반으로 계산된다.

2. **Anchor Reference `A`  (새롭게 도입된 Reference data)**

    Acceptance/Rejection Probe 과정에서 model update가 원래 pretrained model이 학습한 ID 분류 기준에서 크게 벗어나지 않도록 제한하는 ID 기준 샘플이다.

    Acceptance Probe에서는 wrong pseudo-label 또는 과도한 entropy minimization으로 인해 model이 특정 target에 과적응하는 것을 막는다.

    Rejection Probe에서는 entropy maximization / uniformization이 ID decision surface 전체를 무너뜨리는 것을 막는다.

    기존 TARR에는 없던 요소이므로 `w/o anchor` vs `with anchor` ablation이 필요하다.

3. **Augmented Reference `P_aug` 또는 `A_aug`  (이건 보류, 다른 것들 수행 후 도입)**

    Clean ID만이 아니라 label-preserving appearance shift에도 안정적인 reference surface를 만들기 위한 reference이다.

    ImageNet-R/C 같은 csID를 semantic OOD처럼 과하게 penalize하지 않기 위해 필요하다.


`A`와 `P`는 분리한다. 같은 reference를 adaptation anchor와 response measurement에 동시에 사용하면 score가 circular해질 수 있기 때문이다.

```
A = anchor reference set
P = probe / measurement reference set
A ∩ P = ∅
```

---

# 새롭게 시도해볼 방향 정리

새롭게 시도해볼 방향 1, 2를 모두 적용하면, Stage 1을 제외한 TARR pipeline은 다음처럼 된다.

```
Stage 2:
  reference_set을 anchor_A와 probe_P로 분리해서 저장

Stage 3:
  target x에 대해 θ0에서 시작
  Acceptance Probe update: x + anchor_A
  Rejection Probe update: x + anchor_A
  각각에 대해 probe_P response 측정
  θ0로 reset

Stage 4:
  ID-probe response와 OOD-probe response의 contrast로 OOD score 산출
```

새롭게 시도해볼 방향들에 대한 코드 구현, ablation study가 필요함. 세부 구현은 한 번에 모든 요소를 추가하지 않음.

```
1. Anchor reference 없이 Acceptance/Rejection Probe만 추가
-> 기존 reference만 사용해서, 두 probe의 response contrast가 유효한 접근인지 확인.

2. Acceptance/Rejection Probe없이 Anchor reference 추가
-> 기존 TTA 전략만 사용해서, probe update의 drift/collapse를 줄일 수 있는지 확인.

3. Anchor reference와 Acceptance/Rejection Probe 모두 사용
-> 1. 2.의 접근 방법을 모두 사용하여 성능 향상

4. Augmented Reference는 위 구조가 유효하다는 것이 확인된 뒤 csID robustness 개선 목적으로 도입한다.

```

---

# 세부 구현 사항

```
Acceptance/Rejection Probe와 Anchor Reference는 서로 다른 개선 축이다.

- Acceptance/Rejection Probe:
  target을 ID로 받아들이는 update와 reject하는 update를 모두 시험하여,
  두 가설의 reference compatibility 차이를 score에 반영하는 구조적 변화이다.

- Anchor Reference:
  probe 종류와 상관없이 TTA update가 원래 ID decision surface에서 과도하게 벗어나지 않도록 제한하는 regularization 요소이다.

따라서 Anchor는 Acceptance/Rejection Probe의 부가 요소가 아니라,
기존 TARR objective에도 독립적으로 적용 가능한 개선 요소이다.
```

## 1) Acceptance/Rejection Probe

### 목적

Acceptance/Rejection Probe는 target sample을 먼저 ID/OOD로 결정하지 않고, 두 가지 가정을 각각 시험하기 위한 TTA probe이다.

```
Acceptance Probe:
  target을 ID label space 안으로 받아들인다고 가정한 update

Rejection Probe:
  target을 ID label space로 받아들이지 않는다고 가정한 update
```

각 probe 후에는 probe reference `P`에서 response 변화를 측정하고, model은 항상 원래 `θ0`로 reset한다.

목표는 다음을 확인하는 것이다.

```
clean ID:
  Acceptance Probe가 Rejection Probe보다 reference surface와 더 잘 맞음

csID:
  clean ID보다 불안정할 수 있지만,
  여전히 Acceptance Probe가 Rejection Probe보다 더 잘 맞아야 함

semantic OOD:
  Acceptance Probe보다 Rejection Probe가 reference surface와 더 잘 맞아야 함
```

---

### 공통 실행 방식

각 target sample `x`에 대해 다음 과정을 수행한다.

```
1. θ0에서 시작한다.

2. Acceptance Probe update를 수행한다.
   θ_acc = Update(θ0, L_accept(x))

3. probe reference P에서 response를 측정한다.
   Δ_acc = Response(P; θ_acc) - Response(P; θ0)

4. model을 θ0로 reset한다.

5. Rejection Probe update를 수행한다.
   θ_rej = Update(θ0, L_reject(x))

6. probe reference P에서 response를 측정한다.
   Δ_rej = Response(P; θ_rej) - Response(P; θ0)

7. θ0로 reset한다.

8. Δ_acc와 Δ_rej의 차이를 이용해 OOD score를 계산한다.
```

초기 구현에서는 기존 TARR와 동일하게 `P only`로 시작한다.

Anchor reference `A`는 별도 ablation으로 추가한다.

---

### Acceptance Probe 후보

Acceptance Probe는 target을 ID label space 안으로 받아들인다고 가정하는 update이다. 단, predicted label을 무조건 신뢰한다는 뜻은 아니다.

#### A1. Predicted-label CE Acceptance Probe

```
L_accept(x) = CE(fθ(x), y_hat)
```

- 기존 TARR의 `predicted_label_ce`와 가장 가까운 형태.
- 구현이 쉽고 baseline 역할을 한다.
- 단점: ImageNet-R/C처럼 prediction이 틀리는 csID에서 wrong pseudo-label을 과신할 수 있다.

**역할:**

가장 단순한 Acceptance Probe baseline.

---

#### A2. Entropy-minimization Acceptance Probe

```
L_accept(x) = H(fθ(x))
```

- target prediction을 더 confident하게 만드는 방향.
- label을 직접 고정하지 않으므로 `predicted_label_ce`보다 덜 hard하다.
- 단점: semantic OOD도 known class로 confident하게 밀어 넣을 수 있다.

**역할:**

label-free Acceptance Probe baseline.

---

#### A3. View-consistency Acceptance Probe

```
L_accept(x) = JS(fθ(aug_1(x)), ..., fθ(aug_m(x)))
```

또는

```
L_accept(x) = H(mean_i fθ(aug_i(x)))
```

- target의 augmentation views가 일관된 prediction을 갖도록 update한다.
- MEMO / consistency 계열 objective와 가까움.
- 단점: semantic OOD도 view-consistent할 수 있으므로 단독으로는 부족할 수 있다.

**역할:**

augmentation-stable ID explanation이 존재하는지 확인하는 probe.

---

#### 제거한 class-hypothesis search 후보

여러 ID class를 별도 hypothesis로 두고 각각 CE update를 수행하는 acceptance
probe는 현재 연구 경로에서 제거한다.

이 경로는 일부 실험에서 성능 신호를 만들었지만, reviewer에게 설명하기
어렵고 ablation 비용도 크다. 또한 독자가 이미 받아들이기 쉬운
`predicted_label_ce`, `entropy_min`, `view_consistency` 계열보다 claim의
방어 가능성이 낮다.

따라서 현재 A/R claim path에서는 아래 acceptance probe만 유지한다.

```text
predicted_label_ce
entropy_min
view_consistency
```

---

### Rejection Probe 후보

Rejection Probe는 target을 known ID class로 밀어 넣지 않는다고 가정하는 update이다.

#### R1. Entropy-maximization Rejection Probe

```
L_reject(x) = -H(fθ(x))
```

- target prediction을 uniform하게 만드는 방향.
- known class 중 하나로 confident하게 밀어 넣는 것을 막는다.
- 단점: update가 너무 강하면 ID decision surface 전체를 손상시킬 수 있다.

**역할:**

가장 기본적인 Rejection Probe.

---

#### R2. Uniformization Rejection Probe

```
L_reject(x) = KL(fθ(x) || Uniform)
```

또는

```
L_reject(x) = KL(Uniform || fθ(x))
```

- target output을 uniform distribution에 가깝게 만든다.
- entropy maximization과 비슷하지만 구현/gradient 특성이 다를 수 있다.
- 단점: 마찬가지로 과도한 update 시 collapse 위험이 있다.

**역할:**

entropy maximization의 대체 후보.

---

#### R3. Logit-suppression Rejection Probe

```
L_reject(x) = 0.5 * ||zθ(x)||²
```

- target logit magnitude를 줄여 특정 ID class로 강하게 할당되는 것을 완화한다.
- entropy maximization보다 안정적일 수 있다.
- 단점: logit norm이 semantic OOD와 항상 직접 대응하지는 않는다.

**역할:**

안정적인 Rejection Probe 후보.

```
Acceptance/Rejection Probe의 목적은 target을 먼저 ID/OOD로 hard decision하는 것이 아니다.
같은 target에 대해 ID로 받아들이는 update와 reject하는 update를 각각 수행한 뒤,
어느 update가 ID reference surface와 더 양립 가능한지 비교하는 것이다.
```

---

## 2) Probe/Anchor Reference

TARR에서 reference는 두 가지 역할로 나눈다.

- `P`: **Probe / Measurement Reference**
- `A`: **Anchor Reference**

`P`와 `A`는 모두 ID train split에서 선택한 reference sample이지만, 사용 목적이 다르다.

---

### 2.1 Probe / Measurement Reference `P`

`P`는 기존 TARR에서 사용하던 reference와 동일한 역할을 한다.

**역할:**
adaptation 전후 response 변화를 측정하는 기준 set이다.

```
Δ_P = Response(P; θ_after) - Response(P; θ0)
```

최종 TARR score는 `P`에서 측정한 class-wise loss/logit 변화를 기반으로 계산한다.

**주의:**

`P`는 기본적으로 update objective에 사용하지 않는다.

`P`를 update loss에도 사용하면, update가 이미 `P`에 맞춰진 상태에서 다시 `P`의 response를 측정하게 되므로 score가 circular해질 수 있다.

---

### 2.2 Anchor Reference `A`

`A`는 probe update가 pretrained model의 ID decision surface에서 과도하게 벗어나지 않도록 제한하는 ID 기준 set이다.

**역할:**

target sample에 대한 TTA update가 너무 강하게 drift/collapse하지 않도록 제한한다.

Acceptance Probe에서는 wrong pseudo-label 또는 과도한 entropy minimization으로 인해 model이 target 하나에 과적응하는 것을 막는다.

Rejection Probe에서는 entropy maximization / uniformization / logit suppression이 ID decision surface 전체를 무너뜨리는 것을 막는다.

Anchor Reference는 Acceptance/Rejection Probe의 보조 요소일 수 있지만, 그 자체로도 기존 TARR objective에 적용 가능한 독립적인 regularization 요소이다.

따라서 독립적인 ablation이 필요하다.

---

### 2.3 `A`와 `P`를 분리하는 이유

`A`와 `P`는 반드시 분리한다.

```
A = anchor reference set
P = probe / measurement reference set
A ∩ P = ∅
```

같은 reference를 update objective와 response measurement에 동시에 사용하면 score가 과대평가될 수 있다.

```
A: update를 안정화하는 ID 기준 set
P: update 결과를 측정하는 독립적인 reference surface
```

---

### 2.4 Anchor loss 구현 후보

Anchor Reference `A`는 target update objective에 regularization term으로 추가한다.

```
L_update = L_target_objective(x) + λ_anchor L_anchor(A)
```

초기 구현에서는 anchor loss를 단순하게 둔다.

```
L_anchor(A) = mean_{(a,y) in A} CE(fθ(a), y)
```

즉, target에 대한 probe update를 수행하되, 동시에 anchor reference `A`에 대해서는 원래 ID label을 계속 맞추도록 제한한다.

**초기 후보: CE anchor**

- anchor sample의 true label을 계속 맞추도록 하는 loss
- 가장 단순하고 구현이 쉬움
- anchor 효과를 해석하기 좋음

**후속 후보: Distillation anchor**

```
L_anchor(A) = mean_{a in A} KL(fθ0(a) || fθ(a))
```

- update 전 모델 `θ0`의 prediction과 update 후 모델 `θ`의 prediction이 anchor sample에서 크게 달라지지 않도록 하는 loss
- true label만 유지하는 것보다 원래 모델의 soft prediction structure를 더 잘 보존할 수 있음

**후속 후보: Parameter regularization**

```
L_reg = ||θ - θ0||²
```

- update 후 parameter가 원래 pretrained parameter에서 너무 멀어지지 않도록 제한
- reference sample 없이도 적용 가능하지만, ID reference response를 직접 유지하는 것은 아님

초기 실험에서는 모든 항을 동시에 쓰지 않는다.

먼저 `CE anchor`를 사용하고, 필요하면 `Distillation anchor` 또는 `Parameter regularization`을 별도로 비교한다.

---

### 2.5 Reference 구성 방법

초기 구현에서는 reference selection 자체를 크게 바꾸지 않는다.

기존 TARR의 reference config를 유지하되, 선택된 reference를 `A`와 `P`로 나눈다.

```
기존 reference_set → A / P로 split
A ∩ P = ∅
```

예시:

```
A: correct_rpc8
P: correct_rpc8
단, sample overlap 없음
```

초기 목표는 reference selection을 새로 최적화하는 것이 아니라, Anchor Reference 도입 자체가 성능에 미치는 영향을 확인하는 것이다.

---

### 2.6 Augmented Reference는 후순위

Augmented Reference는 clean ID뿐 아니라 label-preserving appearance shift에도 안정적인 reference surface를 만들기 위한 후보이다.

```
P_aug 또는 A_aug
```

ImageNet-R/C 같은 csID가 semantic OOD처럼 과하게 penalize되는 것을 줄이기 위해 필요할 수 있다.

하지만 초기 구현에서는 보류한다.

Acceptance/Rejection Probe와 Anchor Reference의 효과를 먼저 확인한 뒤, csID robustness가 부족할 때 추가한다.

---

## 3) 최종 Score

Acceptance/Rejection Probe가 도입되면, 각 target sample `x`에 대해 두 종류의 reference response가 저장된다.

```
Δ_acc = Response(P; θ_acc) - Response(P; θ0)
Δ_rej = Response(P; θ_rej) - Response(P; θ0)
```

- `Δ_acc`: target을 ID로 받아들였을 때 reference surface가 어떻게 변하는가
- `Δ_rej`: target을 reject/OOD로 보았을 때 reference surface가 어떻게 변하는가

최종 score는 이 두 response 중 어느 쪽이 reference surface와 더 잘 맞는지를 비교하여 계산한다.

Desired ordering은 다음과 같다.

```
clean ID < csID << semantic OOD
```

---

### 3.1 기본 방향

ID/csID target이라면:

```
Acceptance Probe가 reference surface와 더 잘 맞아야 함
Rejection Probe는 상대적으로 부자연스러워야 함
```

semantic OOD target이라면:

```
Acceptance Probe가 reference surface를 더 많이 손상시키거나,
ID-compatible한 response를 만들지 못해야 함

Rejection Probe가 상대적으로 더 자연스러워야 함
```

따라서 최종 score는 다음 형태를 기본으로 한다.

```
OOD_score = Acceptance_incompatibility - Rejection_incompatibility
```

높을수록 OOD-like하다.

---

### 3.2 Score 후보 1: Reference delta-penalty contrast

가장 단순한 score이다.

```
A_ref_delta_penalty = mean_c max(accept_ref_loss_delta_c, 0)
R_ref_delta_penalty = mean_c max(reject_ref_loss_delta_c, 0)

score = A_ref_delta_penalty - R_ref_delta_penalty
```

**해석:**

- Acceptance update가 reference CE loss delta penalty를 크게 만들면 OOD-like
- Rejection update가 reference CE loss delta penalty를 작게 만들면 OOD-like

**장점:**

구현이 쉽고 기존 TARR의 reference CE loss delta와 연결된다.

**단점:**

ImageNet-R/C처럼 csID도 큰 reference CE loss delta penalty를 만들 수 있으므로
단순 magnitude score만으로는 부족할 수 있다.

---

### 3.3 Score 후보 2: ID compatibility contrast

Acceptance response가 단순히 작아야 하는 것이 아니라, **ID-compatible한 방향**이어야 한다고 본다.

```
local_help_acc = max_c max(0, -Δ_acc,c)
off_ref_delta_penalty_acc = mean_c max(Δ_acc,c, 0)

ID_cost_acc = off_ref_delta_penalty_acc - β * local_help_acc
```

Rejection 쪽도 유사하게 `ref_delta_penalty`를 계산한다.

```
R_ref_delta_penalty = mean_c max(Δ_rej,c, 0)

score = ID_cost_acc - α * R_ref_delta_penalty
```

**해석:**

- ID/csID는 Acceptance Probe에서 특정 ID class 또는 class neighborhood에 local improvement가 있어야 함
- semantic OOD는 local improvement 없이 off-class ref delta penalty가 커질 가능성이 높음
- Rejection Probe가 reference를 덜 손상시키면 OOD-like

**장점:**

단순 reference delta penalty보다 “ID로 설명 가능한가?”를 더 직접적으로 본다.

**단점:**

`local_help` 기준을 predicted class로 고정해야 한다. 여러 class hypothesis를
동시에 탐색하는 방식은 claim과 ablation이 어려워 현재 경로에서 제거한다.

---

### 3.4 Score 후보 3: Rejection efficiency score

Rejection Probe가 target을 reject 방향으로 쉽게 이동시키면서도 reference를 덜 손상시키는지를 본다.

```
target_objective_delta = L_reject(θ_rej, x) - L_reject(θ0, x)
ref_delta_penalty(mode) = scalarize(reject_ref_loss_delta, mode)

Reject_efficiency = -target_objective_delta / (ε + ref_delta_penalty(mode))
```

여기서 `ref_loss_delta` 계열은 항상 Probe reference CE loss delta이고,
`target_objective_delta` 계열은 선택된 target branch objective의 delta다.
`reject_target_entropy_delta`가 저장되더라도 entropy diagnostic일 뿐 canonical
efficiency numerator는 아니다.

최종 score:

```
score = Reject_efficiency
```

**해석:**

- semantic OOD는 reject 방향으로 밀기 쉽고, 그 과정에서 ID reference를 덜 손상시킬 가능성이 있음
- ID/csID는 reject 방향으로 밀면 ID decision surface에 더 큰 손상을 줄 수 있음

**장점:**

Rejection Probe 정보를 더 적극적으로 활용한다.

**단점:**

entropy gain이 confidence/MSP와 강하게 correlated될 수 있으므로, 최종
claim에서는 MSP/entropy 계열과 다른 신호를 제공하는지 별도로 확인해야 한다.

---

### 3.5 초기 구현 우선순위

처음부터 모든 score를 사용하지 않고, 아래 순서로 확인한다.

```
S1. Reference delta-penalty contrast
score = A_ref_delta_penalty - R_ref_delta_penalty

S2. ID compatibility contrast
score = ID_cost_acc - α * R_ref_delta_penalty

S3. Rejection efficiency
score = Reject_efficiency
```

초기 실험에서는 `S1`, `S2`부터 시작한다.

`S1/S2`에서 signal이 보이면, rejection-side efficiency를 별도 score로
확인한다. 여러 class hypothesis를 탐색하는 acceptance score는 현재 경로에서
확장하지 않는다.

---

### 3.6 Score 선택 시 주의점

Score rule을 test OOD 성능을 보고 고르면 selection bias가 생긴다.

따라서 score 후보는 적게 유지하고, 선택 기준을 명확히 둔다.

권장 비교 기준:

```
1. clean ID < csID < semantic OOD ordering이 유지되는가
2. csID false positive가 줄어드는가
3. semantic OOD AUROC가 유지되거나 개선되는가
4. 기존 MSP/entropy와 다른 신호를 제공하는가
```

특히 TARR score가 단순 MSP 변형인지 확인하기 위해, MSP bin 안에서도 score가 csID와 semantic OOD를 구분하는지 분석한다.

> **최종 score는 reference loss 변화의 절대 크기가 아니라, Acceptance 가설과 Rejection 가설 중 어느 쪽이 ID reference surface와 더 양립 가능한지를 비교하는 compatibility contrast로 정의한다.**
>

---

## Ablation 설계

Acceptance/Rejection Probe와 Anchor Reference는 서로 다른 개선 축으로 본다.

- **Acceptance/Rejection Probe**
target을 ID로 받아들이는 update와 reject하는 update를 모두 수행하고, 두 가설의 reference compatibility 차이를 score에 반영하는 구조적 변화이다.
- **Anchor Reference**
probe 종류와 상관없이 TTA update가 원래 ID decision surface에서 과도하게 벗어나지 않도록 제한하는 regularization 요소이다.
Acceptance/Rejection Probe를 안정화하는 보조 요소일 수도 있지만, 기존 TARR objective에도 독립적으로 적용 가능한 개선 요소이다.

따라서 다음 2×2 ablation을 수행한다.

```
Ablation 0: Baseline TARR
- 기존 objective
- Probe reference P only
- Anchor 없음
- Acceptance/Rejection 없음

Ablation 1: Anchor only
- 기존 objective 유지
- Anchor A 추가
- Probe reference P에서 response 측정
- Acceptance/Rejection 없음

Ablation 2: Acceptance/Rejection only
- Acceptance Probe + Rejection Probe
- Anchor 없음
- Probe reference P에서 response 측정

Ablation 3: Acceptance/Rejection + Anchor
- Acceptance Probe + Rejection Probe
- Anchor A 추가
- Probe reference P에서 response 측정
```

### 해석 기준

```
Ablation 1만 좋아짐:
  핵심은 Anchor에 의한 update stabilization.

Ablation 2만 좋아짐:
  핵심은 Acceptance/Rejection contrast.

Ablation 1과 2 모두 좋아짐:
  두 개선 축이 독립적으로 유효함.

Ablation 3이 가장 좋음:
  Anchor와 Acceptance/Rejection Probe가 상호보완적임.

Ablation 3이 Ablation 1 또는 2보다 나쁨:
  Anchor가 probe contrast를 약하게 만들었거나,
  Acceptance/Rejection update와 anchor regularization이 충돌했을 가능성 있음.
```

| Setting | Acceptance/Rejection | Anchor | 목적 |
| --- | --- | --- | --- |
| Baseline TARR | X | X | 기존 TARR 기준 성능 |
| Anchor only | X | O | anchor 자체의 update stabilization 효과 확인 |
| Acceptance/Rejection only | O | X | 두 probe의 compatibility contrast 효과 확인 |
| Acceptance/Rejection + Anchor | O | O | 두 개선 축의 상호보완성 확인 |
- `Anchor only`만 좋아지면 anchor에 의한 update stabilization이 핵심이다.
- `Acceptance/Rejection only`만 좋아지면 두 가설의 compatibility contrast가 핵심이다.
- 둘 다 좋아지면 두 개선 축이 독립적으로 유효하다.
- `Acceptance/Rejection + Anchor`가 가장 좋으면 두 요소가 상호보완적이다.
- `Acceptance/Rejection + Anchor`가 단독 ablation보다 나쁘면 anchor가 contrast를 약하게 만들었거나, probe update와 anchor regularization이 충돌했을 수 있다.

---

## 2026-06-02 실험 후 상태

현재 구현과 실험은 위 가설의 일부를 지지하지만, TARR을 standalone
full-spectrum OOD detector로 주장하기에는 아직 부족하다.

### 확인된 점

- Acceptance/Rejection Probe는 ImageNet-200에서 유효한 semantic-ordering
  신호를 만든다.
- CIFAR-100에서 성능이 좋았던 과거 class-hypothesis search branch는
  claim과 ablation이 어려워 현재 코드와 문서 경로에서 제거했다. 따라서
  CIFAR-100은 claim-compatible A/R branch로 다시 확인해야 한다.
- 현재 유지하는 primary score는
  `reject_efficiency = -reject_target_objective_delta / (eps + ref_delta_penalty(mode))`다.
  여기서 numerator는 reject branch objective delta이며
  `reject_target_entropy_delta` diagnostic이 아니다.
- Desired ordering인 `clean ID < csID < semantic OOD`는 A/R
  `reject_efficiency` 계열에서 CIFAR-100과 ImageNet-200 모두 개선된다.
- Anchor-only는 CE/distill/param_reg 모두 현재 설정에서 유의미한 gain을
  만들지 못했다.
- A/R + Anchor도 A/R-only 대비 뚜렷한 gain을 만들지 못했고 runtime만
  증가했다.

### 현재 best 결과

```text
CIFAR-100:
  claim-compatible A/R branch로 재실험 필요
  과거 class-hypothesis search 결과는 현재 claim evidence로 사용하지 않음

ImageNet-200:
  A/R predicted_label_ce + entropy_max
  score_rule = reject_efficiency
  best reference = correct_rpc32
  both AUROC = 63.25
  near-OOD는 Group1 best보다 높음, far-OOD는 ASH보다 크게 낮음

CIFAR-10:
  A/R branch는 default TARR보다 낮음
  control/sanity 역할로 유지
```

### 결론

```text
현재 TARR A/R은 promising한 부분 결과를 만들었지만,
Group1 post-hoc methods를 전역적으로 이기는 standalone full-spectrum OOD
detector는 아니다.
```

논문 claim 후보로는 다음처럼 약하게 정리하는 것이 더 정확하다.

```text
Acceptance/Rejection probing can improve semantic ordering and selected
dataset/aggregate AUROC, but far-OOD robustness remains unresolved.
```

### 다음 연구 방향

1. ImageNet-1K로 바로 확장하지 않는다.
   ImageNet-200 far-OOD가 아직 약하기 때문이다.
2. Direct positive-reference-delta contrast는 primary score에서 제외하고,
   delta-based `reject_efficiency`를 기본 A/R score로 사용한다.
3. CIFAR-100은 `all_rpc16`, ImageNet-200은 `correct_rpc32`를 우선 reference
   설정으로 본다.
4. ImageNet-scale far-OOD 실패 원인을 먼저 분석한다.
   특히 `reject_efficiency`가 near semantic OOD에는 강하지만 far-OOD에는
   Group1 ASH보다 약한 이유를 response distribution 기준으로 확인해야 한다.
5. Augmented Reference는 아직 도입하지 않는다.
   현재 A/R bottleneck과 far-OOD 실패가 먼저 설명되어야 한다.

## 2026-06-02 Stage 4-only score construction audit

저장된 A/R `tta_response`만 사용해서 `reject_efficiency`를 대체하거나 보완할
수 있는 score construction 후보를 확인했다. 새 Stage 3 TTA는 실행하지 않았다.

결론:

```text
Primary score는 reject_efficiency로 유지한다.
저장된 `tta_response`만 재조합하는 Stage 4 score-construction variant는
성능 개선이 없었고 calibration 방법도 사용하지 않는다.
따라서 관련 score rule은 현재 code path와 search space에서 제거한다.
```

핵심 관찰:

- CIFAR-100에서는 `reject_efficiency`가 clean/csID/near/far ordering을 가장
  안정적으로 만든다.
- ImageNet-200 far-OOD에는 entropy-like signal이 일부 있지만, 현재 저장된
  response field에 score-only 방식으로 섞으면 near-OOD 장점이 사라진다.
- 따라서 다음 개선은 Stage 4 score formula가 아니라 Stage 3에서
  far-shift-sensitive response field를 새로 측정하는 방향이어야 한다.

다음 TTA 아이디어:

- 기존 A/R은 semantic compatibility branch로 유지한다.
- 별도 lightweight reject probe에서 feature drift, activation sparsity,
  energy/entropy response 같은 far-shift branch를 측정한다.
- 새 response field가 유효한지 먼저 확인한 뒤, semantic branch와 far-shift
  branch를 단순하게 결합한다.
