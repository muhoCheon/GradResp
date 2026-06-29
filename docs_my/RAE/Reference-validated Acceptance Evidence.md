# RAE: Reference-validated Acceptance Evidence for Out-of-Distribution Detection

## Step 0. OOD를 무엇으로 볼 것인가?

기존 post-hoc OOD detection 방법들은 주로 target sample에 대한 모델의 confidence 또는 representation proximity를 이용해 ID-like 여부를 판단한다. Confidence 기반 방법은 모델이 target sample에 대해 얼마나 확신하는지를 보고, distance 기반 방법은 target representation이 ID representation distribution에 얼마나 가까운지를 본다.

반면 우리는 OOD detection을 다음 질문으로 정의한다.

> target sample $x$에 대해, 모델이 확신하는 ID class 예측이 존재하고, 그 예측이 ID reference samples에 의해서도 검증되는가?
> 

즉, target sample $x$가 ID로 설명되기 위해서는 아래 두 조건이 필요하다.

**첫째**, 모델이 target sample $x$을 어떤 class $c$에 대해 충분한 confidence를 가져야 한다. 즉, target sample $x$를 설명할 수 있는 plausible한 ID class hypothesis가 존재해야 한다. 

$$
q_c(x)=P_\theta(Y=c\mid x)
$$

여기서 $q_c(x)$는 class $c$를 target $x$에 대한 그럴듯한(plausible)한 explanation인지를 나타낸다. 즉, $q_c(x)$는 class hypothesis plausibility 또는 hypothesis prior로 해석할 수 있다.

**둘째**, target sample $x$을 어떤 class $c$라고 가정했을 때, 이미 알고 있는 ID reference samples와도 잘 맞아야 한다. 구체적으로, target sample $x$을 class $c$로 받아들이는 direction이 known labeled ID reference objectives와 양립 가능해야 한다. 이를 **reference validation score**라고 부르고, $V_c(x)$로 나타낸다.

여기서 $V_c(x)$는 target $x$를 class $c$로 받아들이는 방향이 reference set 전체에 의해 얼마나 검증되는지를 나타낸다. 즉, $V_c(x)$는 하나의 reference와의 agreement가 아니라, 여러 labeled ID references를 종합한 class-level validation score이다.

즉, $q_c(x)$와 $V_c(x)$는 아래와 같이 구분된다.

> $q_c(x)$: target $x$에 대해 class $c$를 그럴듯한(plausible) explanation으로 볼 수 있는가?
> 

> $V_c(x)$: reference set 전체가 target $x$에 대한 class $c$ hypothesis를 얼마나 검증하는가?
> 

이 구분은 $q_c(x)$와 $V_c(x)$의 역할을 분리한다. $q_c(x)$는 model prediction에서 나온 class hypothesis plausibility이고, $V_c(x)$는 그 class hypothesis가 labeled ID references에 의해 검증되는 정도이다.

이 두 값을 이용하여 class $c$에 대한 ID evidence는 다음과 같이 정의한다.

$$
E_c(x)=q_c(x)V_c(x)
$$

$E_c(x)$는 target $x$가 class $c$에 대해 plausible하게 예측될 뿐 아니라, 그 예측이 ID reference samples에 의해서도 검증되는지를 나타낸다. 따라서 $E_c(x)$가 높으려면 두 조건이 모두 필요하다. 모델이 class $c$를 강하게 예측하더라도, class $c$로 받아들이는 방향이 ID references와 맞지 않으면 ID evidence는 낮아진다. 반대로 ID references와 잘 맞는 방향이 있더라도, class $c$ 자체가 그럴듯한 예측이 아니라면 ID evidence는 낮아진다. 이 관점에서 OOD sample은 다음과 같이 볼 수 있다.

> 어떤 ID class $c$에 대해서도, 모델의 예측이 충분히 그럴듯하면서 동시에 그 예측이 ID reference samples에 의해 검증되는 근거를 갖지 못하는 sample
> 

제안 방법의 핵심은 모델의 confidence를 그대로 믿지 않는 것이다. 모델이 예측한 class를 ID evidence로 바로 사용하지 않고, 그 class로 target sample을 받아들이는 방향이 labeled ID references와도 잘 맞는지 **first-order reference compatibility**로 한 번 더 확인한다.

## Step 1. Candidate Acceptance Objective

Step 0에서는 target sample $x$가 ID로 설명되기 위해서는, 어떤 ID class $c$가 $x$에 대해 confidence 가 높은 (그럴듯한) 설명이어야 하고, 동시에 그 설명이 ID reference samples에 의해 검증되어야 한다고 보았다.

이제 Step 1에서는 target sample $x$를 class $c$로 **“받아들인다(accept)”(예측을 강화한다)**는 것이 무엇을 의미하는지 정의한다. 여기서 accept는 class $c$를 $x$에 대한 더 강한 설명이 되도록 만들기 위해, 현재 모델을 어떤 방향으로 움직여야 하는지를 묻는 개념이다.

즉, accept direction은 모델이 $x$를 class $c$로 더 잘 설명하도록 만드는 방향이다.

pretrained classifier를 $f_\theta$라고 하고, target sample $x$에 대한 logit vector를

$$
z(x;\theta)
=
\left[
z_1(x;\theta),\dots,z_C(x;\theta)
\right]
\in \mathbb{R}^C
$$

라고 하자. Softmax probability는 다음과 같이 정의된다.

$$
q_c(x)
=
P_\theta(Y=c\mid x)
=
\frac{\exp z_c(x;\theta)}
{\sum_{j=1}^{C}\exp z_j(x;\theta)}
$$

각 candidate class $c$에 대해, class-conditional cross-entropy objective를 다음과 같이 정의한다.

$$
L_{x,c}(\theta)
=
CE(f_\theta(x),c)
=
-\log q_c(x)
$$

여기서 $L_{x,c}$는 모델이 $x$에 대해 class $c$를 더 강한 explanation으로 보도록 만드는 방향을 정의하기 위한 objective이다.

이제 acceptance direction을 정의한다. 먼저 모델 파라미터 중에서 gradient를 계산할 특정 부분 집합을 $\theta_\ell$라고 하자. 예를 들어 $\theta_\ell$은 classifier head, last shared block, 또는 high-level blocks이 될 수 있다. Pretrained parameter를 $\theta_0$라고 할 때, target $x$를 class $c$로 더 잘 받아들이기 위한 방향은 현재 파라미터 지점에서 $L_{x,c}$를 가장 빠르게 줄이는 방향이다.

이를 first-order acceptance direction라고 정의한다.

$$
\rho_c(x)
=
-
\frac{
\nabla_{\theta_\ell} L_{x,c}(\theta_0)
}{
\left\lVert
\nabla_{\theta_\ell} L_{x,c}(\theta_0)
\right\rVert
+
\epsilon
}
$$

여기서 $\epsilon$은 numerical stability를 위한 작은 양수이다.

이 방향을 first-order acceptance direction이라고 부르는 이유는, 이 방향이 현재 pretrained model $\theta_0$에서 gradient, 즉 1차 미분 정보 만으로 정의되기 때문이다. 

작은 step size $\eta>0$와 $\theta_\ell$ 공간의 unit direction $u$를 생각하자. 현재 모델에서 $u$ 방향으로 아주 조금 움직였을 때, $L_{x,c}$의 변화는 Taylor 1차 근사로 다음과 같이 쓸 수 있다.

$$
L_{x,c}(\theta_0+\eta u)
\approx
L_{x,c}(\theta_0)
+
\eta
\nabla_{\theta_\ell}L_{x,c}(\theta_0)^\top u.
$$

이 1차 근사에서 $L_{x,c}$를 가장 빠르게 감소시키는 unit direction은 negative gradient direction이다.

$$
u^\star
=
-
\frac{
\nabla_{\theta_\ell}L_{x,c}(\theta_0)
}{
\left\lVert
\nabla_{\theta_\ell}L_{x,c}(\theta_0)
\right\rVert
}
$$

따라서 $\rho_c(x)$는 모델을 실제로 업데이트 한 뒤의 결과를 보는 것이 아니라, 현재 모델이 class $c$를 더 강한 explanation으로 만들기 위한 first-order acceptance direction을 나타낸다.

이제 이 first-order acceptance direction이 어떤 정보를 담고 있는지 더 명확히 살펴본다.

Softmax 정의에 의해,

$$
q_c(x)
=
\frac{\exp z_c(x;\theta)}
{\exp z_c(x;\theta)+\sum_{j\neq c}\exp z_j(x;\theta)}
$$

분자와 분모를 $\exp z_c(x;\theta)$로 나누면,

$$
q_c(x)
=
\frac{1}
{
1+
\frac{
\sum_{j\neq c}\exp z_j(x;\theta)
}{
\exp z_c(x;\theta)
}
}
$$

여기서

$$
\frac{
\sum_{j\neq c}\exp z_j(x;\theta)
}{
\exp z_c(x;\theta)
}
=
\exp
\left(
\log\sum_{j\neq c}\exp z_j(x;\theta)
-
z_c(x;\theta)
\right)
$$

이므로, class $c$에 대한 log-odds margin을 다음과 같이 정의할 수 있다.

$$
m_c(x;\theta)
=
\log\sum_{j\neq c}\exp z_j(x;\theta)
-
z_c(x;\theta)
$$

$m_c(x;\theta)$은 class $c$가 아닌 나머지 class 전체가 class $c$보다 얼마나 강한지를 나타내는 margin을 나타낸다. $m_c(x;\theta)$가 작아질수록 class $c$가 나머지 class들보다 더 강해진다. 따라서 $x$를 class $c$로 accept한다는 것은 $m_c(x;\theta)$를 줄이는 방향으로 움직인다는 뜻이다.

그러면 softmax probability는 다음과 같이 다시 쓸 수 있다.

$$
q_c(x)
=
\frac{1}{1+\exp m_c(x;\theta)}
$$

또한,

$$
-m_c(x;\theta)
=
\log\frac{q_c(x)}{1-q_c(x)}
$$

이제 class-conditional CE objective를 $m_c(x;\theta)$를 이용해 다시 표현해보자.

$$
L_{x,c}(\theta)
=
-\log q_c(x)
$$

위에서

$$
q_c(x)
=
\frac{1}{1+\exp m_c(x;\theta)}
$$

였으므로,

$$
-\log q_c(x)
=
-\log
\left(
\frac{1}{1+\exp m_c(x;\theta)}
\right)
$$

따라서

$$
L_{x,c}(\theta)
=
\log
\left(
1+\exp m_c(x;\theta)
\right)
$$

즉,

$$
CE(f_\theta(x),c)
=
-\log q_c(x)
=
\log
\left(
1+\exp m_c(x;\theta)
\right)
$$

이제 양변을 $\theta$에 대해 미분한다. Chain rule에 의해,

$$
\nabla_\theta CE(f_\theta(x),c)
=
\nabla_\theta
\log
\left(
1+\exp m_c(x;\theta)
\right)
$$

스칼라 함수 $\log(1+\exp m)$의 $m$에 대한 미분은

$$
\frac{d}{dm}\log(1+\exp m)
=
\frac{\exp m}{1+\exp m}
$$

이다. 따라서

$$
\nabla_\theta CE(f_\theta(x),c)
=
\frac{\exp m_c(x;\theta)}
{1+\exp m_c(x;\theta)}
\nabla_\theta m_c(x;\theta)
$$

한편,

$$
q_c(x)
=
\frac{1}{1+\exp m_c(x;\theta)}
$$

이므로,

$$
1-q_c(x)
=
1-
\frac{1}{1+\exp m_c(x;\theta)}
=
\frac{\exp m_c(x;\theta)}
{1+\exp m_c(x;\theta)}
$$

따라서 다음 관계를 얻는다.

$$
\nabla_\theta CE(f_\theta(x),c)
=
(1-q_c(x))
\nabla_\theta m_c(x;\theta)
$$

즉, class-conditional CE gradient는 다음 두 요소의 곱으로 정확히 분해된다.

$$
\nabla_\theta CE(f_\theta(x),c)
=
\underbrace{(1-q_c(x))}_{\text{confidence headroom}}
\cdot
\underbrace{
\nabla_\theta m_c(x;\theta)
}_{\text{log-odds gradient}}
$$

여기서 $1-q_c(x)$는 class $c$에 대해 남아 있는 confidence headroom을 나타내는 scalar factor이다. 모델이 이미 class $c$에 대해 매우 confident하면 $q_c(x)$가 크고, 따라서 $1-q_c(x)$는 작아진다. 이 경우 CE gradient norm도 작아진다. 반대로 모델이 class $c$에 대해 confident하지 않으면 $1-q_c(x)$가 커지고 CE gradient norm도 커진다.

하지만 제안 방법은 gradient magnitude를 사용하지 않는다. 우리는 normalized gradient direction만 사용한다. 따라서 gradient가 0이 아닌 경우,

$$
\frac{
\nabla_\theta CE(f_\theta(x),c)
}{
\left\lVert
\nabla_\theta CE(f_\theta(x),c)
\right\rVert
}
=
\frac{
\nabla_\theta m_c(x;\theta)
}{
\left\lVert
\nabla_\theta m_c(x;\theta)
\right\rVert
}
$$

즉, familiar한 class-conditional CE objective를 이용해 first-order acceptance direction을 정의하더라도, 그 normalized direction은 log-odds margin의 normalized negative gradient direction과 정확히 동일하다.

$$
-
\frac{
\nabla_\theta CE(f_\theta(x),c)
}{
\left\lVert
\nabla_\theta CE(f_\theta(x),c)
\right\rVert
}
=
-
\frac{
\nabla_\theta m_c(x;\theta)
}{
\left\lVert
\nabla_\theta m_c(x;\theta)
\right\rVert
}
$$

이 동치 관계는 중요하다. 제안 방법은 CE gradient를 normalized하여 scalar confidence headroom을 제거하고, class $c$의 log-odds margin를 증가시키기 위해 필요한 방향 정보만을 남긴다.

정리하면, Step 1에서는 candidate class $c$에 대한 first-order acceptance direction을 다음과 같이 정의한다.

$$
\rho_c(x)
=
-
\frac{
\nabla_{\theta_\ell} CE(f_{\theta_0}(x),c)
}{
\left\lVert
\nabla_{\theta_\ell} CE(f_{\theta_0}(x),c)
\right\rVert
+
\epsilon
}
=
=
-
\frac{
\nabla_\theta m_c(x;\theta)
}{
\left\lVert
\nabla_\theta m_c(x;\theta)
\right\rVert
+
\epsilon
}
$$

따라서 $\rho_c(x)$는 현재 pretrained model에서 target $x$를 class $c$로 받아들이기 위해 필요한 first-order directional information을 나타낸다.

---

## Step 2. First-order Reference Compatibility

Step 1에서는 target sample $x$를 candidate class $c$로 **받아들인다(accept)**는 것이 무엇인지 정의하였다. 구체적으로, target $x$를 class $c$로 더 잘 설명하도록 만드는 first-order acceptance direction을 다음과 같이 정의하였다.

$$
\rho_c(x)
=
-
\frac{
\nabla_{\theta_\ell} L_{x,c}(\theta_0)
}{
\left\lVert
\nabla_{\theta_\ell} L_{x,c}(\theta_0)
\right\rVert
+
\epsilon
}
$$

여기서

$$
L_{x,c}(\theta)
=
CE(f_\theta(x),c)
$$

이고, $\theta_0$는 pretrained parameter, $\theta_\ell$은 gradient를 계산하는 parameter subset이다. 즉, $\rho_c(x)$는 현재 pretrained model에서 target $x$를 class $c$로 받아들이기 위해 모델이 움직여야 하는 unit direction이다.

이제 Step 2에서는 이 acceptance direction이 labeled ID reference objectives와 양립 가능한지를 살펴본다. 핵심 질문은 다음과 같다.

> target $x$를 class $c$로 받아들이는 방향이, 이미 알고 있는 ID reference samples의 objectives에도 도움이 되는 방향인가?
> 

이를 **first-order reference compatibility** 관점으로 정의한다.

Reference sample $p$의 ground-truth label을 $y_p$라고 하자. Reference objective는 다음과 같이 정의한다.

$$
L_p(\theta)
=
CE(f_\theta(p),y_p)
$$

이 objective는 reference sample $p$가 자신의 true label $y_p$로 잘 설명되도록 만드는 loss이다.

이제 target $x$를 class $c$로 받아들이기 위해, pretrained parameter $\theta_0$에서 acceptance direction $\rho_c(x)$을 따라 작은 step을 수행한다고 하자.

$$
\theta_0'
=
\theta_0
+
\eta \rho_c(x)
$$

여기서 $\eta>0$는 작은 step size이다. $\rho_c(x)$는 이미 negative gradient direction이므로, $\theta_0+\eta\rho_c(x)$는 target objective $L_{x,c}$를 줄이는 방향으로의 이동이다.

이 update가 reference objective $L_p$에 어떤 영향을 주는지 살펴보자. 여기서 중요한 점은 ****$\rho_c(x)$**가 target objective** $L_{x,c}$**를 줄이기 위해 정의된 방향이**라는 것이다. 하지만 우리는 **이 방향이 target에만 좋은 방향인지, 아니면 labeled ID reference objective에도 좋은 방향인지를 확인**하고자 한다.

이를 위해 reference loss $L_p$를 현재 pretrained parameter $\theta_0$ 근처에서 1차 Taylor 근사한다. 일반적으로 어떤 differentiable function $F(\theta)$에 대해, 현재 지점 $\theta_0$에서 작은 displacement $\Delta\theta$만큼 움직이면 함수값 변화는 다음과 같이 근사할 수 있다.

$$
F(\theta_0+\Delta\theta)
\approx
F(\theta_0)
+
\nabla_\theta F(\theta_0)^\top \Delta\theta
$$

즉, 현재 지점에서 아주 조금 움직였을 때 함수값이 얼마나 변하는지는 gradient와 이동 방향의 내적에 의해 1차적으로 결정된다. Gradient와 이동 방향이 같은 방향이면 함수값은 증가하고, 반대 방향이면 함수값은 감소한다. 두 방향이 거의 직교하면 1차 변화는 작다.

이제 이 일반적인 1차 근사를 reference objective $L_p$에 적용한다. Target $x$를 class $c$로 받아들이기 위해 acceptance direction $\rho_c(x)$을 따라 작은 step을 수행한다고 하자.

$$
\theta_0+
\eta \rho_c(x)
$$

여기서 $\eta>0$는 작은 step size이다. 즉, parameter displacement는

$$
\Delta\theta
=
\eta\rho_c(x)
$$

이다. 이를 reference loss $L_p$의 Taylor 1차 근사에 대입하면,

$$
L_p(\theta_0+\eta\rho_c(x))
\approx
L_p(\theta_0)
+
\eta
\nabla_{\theta_\ell}L_p(\theta_0)^\top
\rho_c(x)
$$

이다. 따라서 reference loss 변화량은

$$
L_p(\theta_0+\eta\rho_c(x))
-
L_p(\theta_0)
\approx
\eta
\nabla_{\theta_\ell}L_p(\theta_0)^\top
\rho_c(x)
$$

로 쓸 수 있다.

이 식은 target acceptance direction이 reference objective에 어떤 영향을 주는지를 직접 보여준다. $\rho_c(x)$는 target $x$를 class $c$로 받아들이기 위한 방향이다. 그런데 이 방향이 reference loss gradient $\nabla_{\theta_\ell}L_p(\theta_0)$와 반대 방향이면, reference loss도 감소한다. 반대로 같은 방향이면 reference loss는 증가한다.

따라서 target $x$를 class $c$로 받아들이는 작은 update가 reference objective에 supportive한지 interfering한지는 다음 내적의 부호에 의해 1차적으로 결정된다.

$$
\nabla_{\theta_\ell}L_p(\theta_0)^\top
\rho_c(x)
$$

만약

$$
\nabla_{\theta_\ell}L_p(\theta_0)^\top
\rho_c(x)
<0
$$

이면, target acceptance direction으로 움직였을 때 reference loss가 감소한다. 즉, target을 class $c$로 받아들이는 방향은 reference objective에도 도움이 되는 방향이다.

반대로

$$
\nabla_{\theta_\ell}L_p(\theta_0)^\top
\rho_c(x)
>0
$$

이면, target acceptance direction으로 움직였을 때 reference loss가 증가한다. 즉, target을 class $c$로 받아들이는 방향은 reference objective와 충돌하는 방향이다.

이제 위의 내적을 target acceptance direction과 reference acceptance direction의 관계로 다시 표현한다. Reference sample $p$에 대해서도 자신의 label $y_p$로 받아들이는 reference acceptance direction을 정의할 수 있다.

$$
\rho(p)
=
-
\frac{
\nabla_{\theta_\ell} L_p(\theta_0)
}{
\left\lVert
\nabla_{\theta_\ell} L_p(\theta_0)
\right\rVert
+
\epsilon
}
$$

즉, $\rho(p)$는 현재 pretrained model에서 reference sample $p$의 CE loss를 가장 빠르게 줄이는 direction이다.

그러면 target acceptance direction과 reference acceptance direction 사이의 directional agreement를 다음과 같이 정의할 수 있다.

$$
K_c(x,p)
=
\langle
\rho_c(x),
\rho(p)
\rangle
$$

이를 **pairwise acceptance agreement**라고 부른다. $K_c(x,p)$는 target-class hypothesis $(x,c)$와 하나의 labeled reference objective $L_p$ 사이의 pairwise directional agreement를 측정한다.

$K_c(x,p)>0$이면 target $x$를 class $c$로 받아들이는 direction과 reference $p$를 자신의 label로 받아들이는 direction이 서로 정렬되어 있다는 뜻이다. 반대로 $K_c(x,p)<0$이면 두 acceptance directions가 서로 충돌한다는 뜻이다.

이제 $K_c(x,p)$가 first-order reference loss change와 어떻게 연결되는지 살펴보자. Reference gradient를

$$
g_p
=
\nabla_{\theta_\ell} L_p(\theta_0)
$$

라고 두면,

$$
\rho(p)
=
-
\frac{g_p}{\left\lVert g_p \right\rVert+
\epsilon}
$$

이다. $\epsilon$을 생략하고 보면,

$$
g_p
=
-\left\lVert g_p \right\rVert\rho(p)
$$

이므로,

$$
\begin{aligned}
\nabla_{\theta_\ell} L_p(\theta_0)^\top
\rho_c(x)
&=
g_p^\top\rho_c(x) \\
&=
-\left\lVert g_p \right\rVert
\langle
\rho(p),
\rho_c(x)
\rangle \\
&=
-\left\lVert g_p \right\rVert K_c(x,p)
\end{aligned}
$$

이다.

따라서 reference loss 변화는 다음과 같이 쓸 수 있다.

$$
L_p(\theta_0+\eta\rho_c(x))
-
L_p(\theta_0)
\approx
-
\eta
\left\lVert g_p \right\rVert
K_c(x,p)
$$

즉,

$$
K_c(x,p)>0
$$

이면,

$$
L_p(\theta_0+\eta\rho_c(x))
-
L_p(\theta_0)
<0
$$

이므로 target $x$를 class $c$로 받아들이는 update가 reference loss도 줄인다. 이 경우 target acceptance direction은 reference objective에 대해 **supportive transfer**를 일으킨다고 볼 수 있다.

반대로,

$$
K_c(x,p)<0
$$

이면,

$$
L_p(\theta_0+\eta\rho_c(x))
-
L_p(\theta_0)
>0
$$

이므로 target $x$를 class $c$로 받아들이는 update가 reference loss를 증가시킨다. 이 경우 target acceptance direction은 reference objective와 **interfering**하거나 **conflicting**한다고 볼 수 있다.

따라서 $K_c(x,p)$는 단순히 두 gradient directions의 cosine similarity를 계산한 값으로만 해석하면 안 된다. 위의 first-order approximation에 의해, $K_c(x,p)$는 target $x$를 class $c$로 받아들이는 방향으로 모델을 조금 움직였을 때 reference loss $L_p$가 1차적으로 감소할지, 아니면 증가할지를 나타낸다.

이런 의미에서 $K_c(x,p)$는 target-class hypothesis $(x,c)$와 labeled ID reference objective $L_p$ 사이의 **pairwise acceptance agreement**이다. 즉, $K_c(x,p)$는 target $x$를 class $c$로 받아들이는 것이 하나의 ID reference objective와 방향적으로 양립 가능한지 판단하는 pairwise compatibility score이다.

여기서 중요한 점은 실제 first-order reference gain이 $K_c(x,p)$만으로 결정되는 것은 아니라는 것이다. 실제 reference loss 감소량은 1차 근사에서

$$
\eta \left\lVert g_p \right\rVert K_c(x,p)
$$

에 비례한다. 즉, reference gradient norm $\left\lVert g_p \right\rVert$도 실제 loss 변화 magnitude에 영향을 준다.

그러나 제안 방법은 reference gradient norm을 직접 사용하지 않는다. 우리는 reference loss가 얼마나 크게 변하는지가 아니라, target을 class $c$로 받아들이는 방향이 reference objective와 같은 방향인지 반대 방향인지를 보고자 한다.

이는 reference sample의 confidence 또는 CE saturation 정도가 score에 직접 섞이는 것을 막기 위한 것이다. Reference sample이 이미 매우 confident하게 맞춰져 있으면 $\left\lVert g_p \right\rVert$가 작고, 상대적으로 덜 confident하면 $\left\lVert g_p \right\rVert$가 커질 수 있다. 이러한 magnitude를 포함하면 score가 reference confidence나 loss scale에 영향을 받을 수 있다. 반면 $K_c(x,p)$는 이러한 scalar magnitude를 제거하고, target acceptance direction과 reference acceptance direction 사이의 순수한 directional compatibility를 측정한다.

정리하면, Step 2에서는 target $x$를 class $c$로 받아들이는 acceptance direction $\rho_c(x)$이 labeled ID reference objective $L_p$와 양립 가능한지를 first-order reference compatibility 관점에서 측정한다. 이를 위해 target-reference pairwise agreement를 다음과 같이 정의한다.

$$
K_c(x,p)
=
\langle
\rho_c(x),
\rho(p)
\rangle
$$

여기서 $K_c(x,p)>0$이면 target acceptance direction은 reference objective에 supportive하고, $K_c(x,p)<0$이면 reference objective와 conflicting하다. 따라서 $K_c(x,p)$는 이후 class $c$에 대한 **reference validation score** $V_c(x)$를 구성하는 기본 pairwise unit이 된다.

즉, $K_c(x,p)$는 하나의 reference와의 **pairwise acceptance agreement**이고, $V_c(x)$는 이러한 pairwise agreements를 reference set 전체에서 종합한 class-level validation score이다.

## Step 3. Reference Validation Score $V_c(x)$

Step 2에서는 target-class hypothesis $(x,c)$와 하나의 labeled reference objective $L_p$ 사이의 pairwise acceptance agreement를 다음과 같이 정의하였다.

$$
K_c(x,p)
=
\langle
\rho_c(x),
\rho(p)
\rangle
$$

여기서 $K_c(x,p)$는 target $x$를 class $c$로 받아들이는 direction이 reference sample $p$를 자신의 label로 받아들이는 direction과 얼마나 잘 맞는지를 나타낸다. 즉, $K_c(x,p)$는 하나의 reference sample에 대한 pairwise-level agreement이다.

그러나 하나의 reference sample과 잘 맞는다는 사실만으로 target $x$가 class $c$로 충분히 검증되었다고 말할 수는 없다. Class $c$ hypothesis가 ID reference samples에 의해 지지된다고 말하려면, target을 class $c$로 받아들이는 direction이 class $c$의 reference objectives와 일관되게 양립 가능해야 한다. 또한 단순히 어떤 reference와 잘 맞는 것이 아니라, class $c$의 references와 다른 classes의 references를 구분할 수 있어야 한다.

따라서 Step 3에서는 pairwise acceptance agreement $K_c(x,p)$들을 reference set 전체에서 종합하여, class $c$ hypothesis에 대한 reference-level validation score를 정의한다. 이를 **reference validation score**라고 부르고 $V_c(x)$로 나타낸다.

$$
K_c(x,p)
\quad\longrightarrow\quad
V_c(x)
$$

즉,

$$
K_c(x,p)
=
\text{one-reference pairwise agreement}
$$

이고,

$$
V_c(x)
=
\text{reference-set-level validation for class }c
$$

이다.

이를 정의하기 위해 class-wise reference set을 생각하자. Class $c$에 대한 reference set을

$$
P_c
=
\{p : y_p=c,\; \hat y_{\theta_0}(p)=c\}
$$

라고 둔다. 즉, $P_c$는 ground-truth label이 class $c$이고 pretrained model $\theta_0$도 class $c$로 올바르게 예측한 ID reference samples의 집합이다. 이러한 reference samples는 class $c$에 대한 reliable labeled ID objectives를 제공한다.

Target $x$를 class $c$로 받아들이는 direction이 class $c$ reference와 잘 맞는지 확인하기 위해 same-class reference를

$$
p^+ \sim P_c
$$

라고 둔다. 또한 class $c$가 아닌 다른 ID classes의 reference와 비교하기 위해 other-class reference를

$$
p^- \sim P_{\neg c}
$$

라고 둔다. 여기서

$$
P_{\neg c}
=
\bigcup_{j\neq c}P_j
$$

이다.

가장 단순한 생각은 same-class references에 대한 average agreement를 사용하는 것이다.

$$
\frac{1}{|P_c|}
\sum_{p\in P_c}
K_c(x,p)
$$

하지만 이 값만으로는 class $c$ hypothesis가 충분히 검증되었다고 보기 어렵다. 같은 class reference와 어느 정도 잘 맞더라도, 다른 class references와도 비슷하게 잘 맞는다면 class $c$에 특이적인 evidence라고 보기 어렵기 때문이다. 또한 same-class reference와의 agreement가 other-class reference보다 크더라도, 그 값 자체가 음수라면 target acceptance direction은 same-class reference objective에도 supportive하지 않다.

따라서 reference validation score는 두 조건을 동시에 확인해야 한다.

첫째, target을 class $c$로 받아들이는 direction은 same-class reference objective에 대해 supportive해야 한다.

$$
K_c(x,p^+)>0
$$

둘째, 그 supportive agreement는 other-class reference와의 agreement보다 커야 한다.

$$
K_c(x,p^+)>K_c(x,p^-)
$$

첫 번째 조건은 target acceptance direction이 class $c$ reference objective와 실제로 양립 가능한지를 확인한다. 두 번째 조건은 그 양립성이 class $c$에 특이적인지를 확인한다.

따라서 class $c$에 대한 reference validation score를 다음과 같이 정의한다.

$$
V_c(x)
=
\Pr_{p^+\sim P_c,\;p^-\sim P_{\neg c}}
\left[
K_c(x,p^+)>0
\;\land\;
K_c(x,p^+)>K_c(x,p^-)
\right]
$$

이 정의에서 $V_c(x)$는 다음 질문에 답한다.

> same-class reference와 other-class reference를 비교했을 때, same-class reference가 positive agreement를 가지면서 동시에 other-class reference보다 더 잘맞는 경우가 얼마나 자주 발생하는가?
> 

이 확률이 높다면, target $x$를 class $c$로 받아들이는 방향은 labeled ID references에 의해 잘 검증된다고 볼 수 있다. 반대로 이 확률이 낮다면, 모델이 target $x$를 class $c$로 예측하더라도 그 class hypothesis는 reference objectives에 의해 충분히 지지되지 않는다.

실제로는 finite reference set을 사용하므로, $V_c(x)$는 empirical estimator로 계산한다. $P_c$에서 same-class references를, $P_{\neg c}$에서 other-class references를 사용하면,

$$
\widehat V_c(x)
=
\frac{1}{|P_c||P_{\neg c}|}
\sum_{p^+\in P_c}
\sum_{p^-\in P_{\neg c}}
\mathbf{1}
\left[
K_c(x,p^+)>0
\;\land\;
K_c(x,p^+)>K_c(x,p^-)
\right]
$$

로 쓸 수 있다.

이 empirical score는 $K_c(x,p)$를 단순 평균하는 것과 다르다. 

평균은 몇 개의 큰 값에 의해 쉽게 지배된다. 예를 들어 일부 same-class reference와는 매우 잘 맞지만, 많은 same-class reference와는 충돌하는 경우에도 평균은 높게 나올 수 있다. 하지만 우리가 원하는 것은 “몇몇 reference와 강하게 맞는가?”가 아니라, **class** $c$ **reference들에 대해 일관되게 검증되는가?**이다.

따라서, $V_c(x)$는 단순히 $K_c(x,p^+)$가 $K_c(x,p^-)$보다 평균적으로 조금 더 높은가를 묻는게 아니라 class $c$ hypothesis가 reference comparisons에서 반복적으로 검증되는가를 보는 것이다. 이는 반드시 실험으로 성능 비교가 필요하다

정리하면, Step 3의 핵심은 다음과 같다.

$$
K_c(x,p)
=
\text{pairwise acceptance agreement with one reference}
$$

$$
V_c(x)
=
\text{reference validation score aggregated over references}
$$

따라서 $V_c(x)$는 target $x$를 class $c$로 받아들이는 것이 ID reference set에 의해 얼마나 검증되는지를 나타낸다.

---

## Step 4. Class-wise ID Evidence $E_c(x)$

Step 0에서는 target sample $x$가 어떤 ID class로 설명되기 위해서는 두 조건이 필요하다고 보았다. 첫째, model이 class $c$를 target $x$에 대한 plausible한 class hypothesis로 보아야 한다. 둘째, 그 class hypothesis가 labeled ID references에 의해서도 검증되어야 한다.

Step 3까지의 내용을 통해 이 두 조건은 각각 다음 두 값으로 표현된다.

$$
q_c(x)
=
\text{class hypothesis plausibility}
$$

$$
V_c(x)
=
\text{reference validation score}
$$

여기서 $q_c(x)$는 pretrained model이 target $x$를 class $c$로 볼 plausibility를 나타내고, $V_c(x)$는 target $x$를 class $c$로 받아들이는 direction이 labeled ID references에 의해 얼마나 검증되는지를 나타낸다.

따라서 class $c$에 대한 ID evidence를 다음과 같이 정의한다.

$$
E_c(x)
=
q_c(x)V_c(x)
$$

이 값은 target $x$가 class $c$로 plausible하게 예측될 뿐 아니라, 그 예측이 ID reference objectives에 의해서도 검증되는지를 나타낸다.

곱셈 형태를 사용하는 이유는 $q_c(x)$와 $V_c(x)$가 서로 다른 두 조건을 나타내기 때문이다. $E_c(x)$가 높으려면 model confidence와 reference validation이 모두 높아야 한다.

만약 model이 class $c$를 강하게 예측하더라도,

$$
q_c(x)\text{ is high}
$$

reference validation이 낮다면,

$$
V_c(x)\text{ is low}
$$

class-wise ID evidence는 낮아진다.

$$
E_c(x)=q_c(x)V_c(x)\text{ is low}
$$

이는 high-confidence prediction이 항상 ID evidence가 되는 것은 아니라는 점을 반영한다. 즉, model confidence는 labeled ID references에 의해 한 번 더 검증되어야 한다.

반대로, 어떤 direction이 ID references와 잘 맞더라도 model이 class $c$ 자체를 plausible한 explanation으로 보지 않는다면,

$$
q_c(x)\text{ is low}
$$

역시 $E_c(x)$는 낮아진다. 이는 reference agreement만으로는 class $c$ explanation이 충분하지 않다는 점을 반영한다.

따라서 $E_c(x)$는 다음과 같은 soft-AND evidence로 해석할 수 있다.

$$
E_c(x)
=
\underbrace{q_c(x)}_{\text{prediction plausibility}}
\cdot
\underbrace{V_c(x)}_{\text{reference validation}}
$$

즉,

$$
E_c(x)
=
\text{plausible class hypothesis supported by ID references}
$$

이다.

이 관점에서 class $c$에 대한 ID explanation은 단순히 model이 class $c$를 예측한다는 것만으로 충분하지 않다. 그 예측을 강화하는 acceptance direction이 labeled ID reference objectives와도 양립 가능해야 한다. 따라서 $E_c(x)$는 target $x$에 대한 class-wise ID evidence로 사용된다.

---

## Step 5. Reference Set Construction and Interpretation

Step 3에서는 reference validation score $V_c(x)$를 정의하기 위해 class-wise reference set $P_c$를 사용하였다. 이제 Step 5에서는 이 reference set이 무엇을 대표하는지, 그리고 왜 이러한 reference set을 사용하는지 정리한다.

Class $c$에 대한 reference set은 다음과 같이 정의한다.

$$
P_c
=
\{p : y_p=c,\; \hat y_{\theta_0}(p)=c\}
$$

즉, $P_c$는 ground-truth label이 class $c$이고 pretrained model $\theta_0$도 class $c$로 올바르게 예측한 ID samples의 집합이다.

이 조건을 사용하는 이유는 reference sample이 class $c$에 대한 reliable labeled ID objective를 제공해야 하기 때문이다. 만약 model이 이미 잘못 예측하는 sample을 reference로 사용하면, 그 sample의 gradient direction은 class $c$의 안정적인 ID objective를 대표한다고 보기 어렵다. 따라서 correctly classified ID samples를 사용하여, pretrained model이 이미 안정적으로 설명하고 있는 class-wise ID objectives를 reference로 삼는다.

여기서 reference set은 prototype을 만들기 위한 것이 아니다. 기존 distance 기반 방법은 target representation이 ID distribution에 얼마나 가까운지를 본다. 반면 제안 방법에서 reference sample은 target과 가까운지 보기 위한 기준점이 아니라, target acceptance direction이 labeled ID objective와 양립 가능한지 검증하기 위한 기준 objective이다.

즉, reference set은

$$
\text{prototype anchor}
$$

가 아니라,

$$
\text{objective anchor}
$$

로 사용된다.

Same-class reference $p^+\sim P_c$는 target $x$를 class $c$로 받아들이는 direction이 class $c$ ID objective와 양립 가능한지를 확인하는 기준이다. Other-class reference $p^-\sim P_{\neg c}$는 그 agreement가 class $c$에 특이적인지를 확인하기 위한 비교 기준이다.

reference data를 뽑을 때는 먼저 ID train split 전체를 pretrained model로 평가하여 candidate metadata를 만든다. 이 metadata는 각 sample의 ground-truth label, pretrained prediction, confidence, correctness, dataset index를 기록한다. 그 다음 이 metadata에서 class별 filter를 적용하고, class $c$의 clean ID samples 중 pretrained model이 올바르게 예측한 samples에서 정확히 $N$개를 uniform random으로 선택하여 $P_c$를 구성한다.

여기서 class-wise quota는 엄격하게 해석한다. 어떤 class에서 filter 이후 eligible candidate가 $N$개보다 적으면 해당 reference configuration은 유효한 $P_c$를 만들 수 없으므로 실패로 처리한다. 부족한 만큼만 사용하여 계속 진행하면 $N$에 대한 reference-size stability 해석과 class-wise validation score 비교가 깨지기 때문이다.

이는 confidence-based selection 같은 추가 heuristic을 기본 formulation에 섞지 않도록 한다. 다만 full, high-confidence, correct-high-confidence 등은 별도 ablation으로 비교할 수 있다.

---

## Step 6. Final ID Evidence and OOD Score

Step 4에서는 각 class $c$에 대한 class-wise ID evidence를 다음과 같이 정의하였다.

$$
E_c(x)
=
q_c(x)V_c(x)
$$

이제 target sample $x$ 전체가 ID로 설명될 수 있는지를 판단하기 위해, class-wise ID evidence들을 하나의 final ID evidence로 종합한다.

핵심 질문은 다음과 같다.

> target sample $x$를 설명할 수 있는 충분히 plausible하고 reference-validated된 ID class가 존재하는가?
> 

가장 직접적인 aggregation은 class-wise evidence의 maximum을 사용하는 것이다.

$$
E_{\mathrm{ID}}(x)
=
\max_{c\in\{1,\dots,C\}}E_c(x)
$$

즉,

$$
E_{\mathrm{ID}}(x)
=
\max_{c\in\{1,\dots,C\}}q_c(x)V_c(x)
$$

이다.

이 max aggregation은 다음 의미를 갖는다. Target $x$가 ID라고 말하려면 모든 class에 대해 높은 evidence를 가질 필요는 없다. 적어도 하나의 ID class $c$가 target $x$를 plausible하게 설명하고, 그 explanation이 ID references에 의해 검증되면 충분하다.

따라서 $E_{\mathrm{ID}}(x)$는 target $x$에 대해 가장 강한 ID explanation의 evidence를 나타낸다.

$$
E_{\mathrm{ID}}(x)
=
\text{best reference-validated ID explanation for }x
$$

이 값이 높으면 target $x$는 어떤 ID class로 잘 설명될 수 있다고 본다. 반대로 이 값이 낮으면, 어떤 class에 대해서도 model prediction과 reference validation이 동시에 충분하지 않다는 뜻이다.

### Predicted-class-only variant

위의 $\max_c$ aggregation은 모든 class hypothesis를 평가하는 가장 직접적인 정의이다. 다만 실제 구현과 대규모 dataset에서는 모든 class에 대해 $V_c(x)$를 계산하는 비용이 커질 수 있다. 따라서 계산량을 줄이는 변형으로, pretrained classifier가 target $x$에 대해 가장 그럴듯하다고 보는 class 하나만 평가하는 옵션도 고려할 수 있다.

먼저 predicted class를 다음과 같이 정의한다.

$$
\hat c(x)
=
\arg\max_c q_c(x)
$$

그 다음 predicted-class-only ID evidence를

$$
E_{\mathrm{pred}}(x)
=
q_{\hat c(x)}(x)V_{\hat c(x)}(x)
$$

로 정의할 수 있다. 이에 대응하는 OOD score는

$$
S_{\mathrm{OOD,pred}}(x)
=
-
\log\left(E_{\mathrm{pred}}(x)+\epsilon\right)
$$

이다.

이 variant는 $E_{\mathrm{ID}}(x)=\max_c q_c(x)V_c(x)$와 동일한 estimator는 아니다. 즉, 모델의 top-1 class hypothesis가 reference validation에서 약하지만 다른 class hypothesis가 강한 경우, $\max_c$ version과 predicted-class-only version은 서로 다른 값을 낼 수 있다. 그러나 OOD detection 관점에서는 top-1 prediction이 실제로 대부분의 ID/OOD 구분 신호를 담고 있을 가능성이 있으며, 이 경우 predicted-class-only score는 계산량을 크게 줄이는 실용적인 instantiation이 될 수 있다.

따라서 predicted-class-only score는 원 formulation의 대체 정의로 고정하기보다, 실험적으로 평가할 candidate로 둔다. 특히 CIFAR-scale에서는 $\max_c$ version과 predicted-class-only version을 모두 비교하고, 성능 차이와 계산 비용을 함께 보고 최종 claim에서 어떤 version을 사용할지 판단해야 한다. 만약 predicted-class-only score가 비슷하거나 더 좋은 OOD/FSOOD 성능을 보인다면, 이는 RAE framework의 효율적인 variant로 주장할 수 있다. 반대로 성능이 유의하게 낮다면, 최종 claim-bearing formulation은 모든 class를 평가하는 $\max_c$ version을 사용해야 한다.

따라서 OOD score는 ID evidence의 음의 log로 정의할 수 있다.

$$
S_{\mathrm{OOD}}(x)
=
-
\log\left(E_{\mathrm{ID}}(x)+\epsilon\right)
$$

즉,

$$
S_{\mathrm{OOD}}(x)
=
-
\log\left(
\max_c q_c(x)V_c(x)
+
\epsilon
\right)
$$

이다.

여기서 $\epsilon$은 numerical stability를 위한 작은 양수이다.

이 정의에서 $S_{\mathrm{OOD}}(x)$가 낮으면 target $x$는 ID-like하다고 본다. 이는 높은 class-wise ID evidence를 가진 ID explanation이 존재하기 때문이다. 반대로 $S_{\mathrm{OOD}}(x)$가 높으면 target $x$는 OOD-like하다고 본다. 이는 어떤 ID class에 대해서도 confidence와 reference validation이 동시에 충분하지 않기 때문이다.

정리하면 최종 구조는 다음과 같다.

$$
K_c(x,p)
\quad\longrightarrow\quad
V_c(x)
\quad\longrightarrow\quad
E_c(x)=q_c(x)V_c(x)
\quad\longrightarrow\quad
E_{\mathrm{ID}}(x)
$$

그리고 최종 OOD score는

$$
S_{\mathrm{OOD}}(x)
=
-
\log\left(E_{\mathrm{ID}}(x)+\epsilon\right)
$$

이다.

이로써 제안 방법은 target sample $x$가 ID인지 여부를 다음 질문으로 판단한다.

> target $x$에 대해, model이 plausible하게 예측하고 그 예측이 labeled ID reference objectives에 의해 검증되는 ID class explanation이 존재하는가?
> 

이 질문에 대한 evidence가 낮을수록 sample은 OOD-like하다고 판단한다.

#### 구현 시 기본은 단순 부호 반전으로 1차구현

## Step 7. Choice of Gradient Space $\theta_\ell$

지금까지의 formulation에서 acceptance direction은 특정 parameter subset $\theta_\ell$에 대해 정의되었다.

$$
\rho_c(x)
=
-
\frac{g_c^{(\ell)}(x)}{\left\lVert g_c^{(\ell)}(x)\right\rVert+
\epsilon}
$$

$$
\rho(p)
=
-
\frac{g_p^{(\ell)}}{\left\lVert g_p^{(\ell)}\right\rVert+
\epsilon}
$$

따라서 $K_c(x,p)$, $V_c(x)$, 그리고 최종 OOD score는 모두 gradient를 어느 parameter space에서 계산하는지에 영향을 받는다. 즉, $\theta_\ell$의 선택은 단순한 implementation detail이 아니라, 제안 방법이 어떤 종류의 reference compatibility를 측정하는지를 결정한다.

핵심 질문은 다음과 같다.

> target $x$를 class $c$로 받아들이기 위해 모델의 어느 부분을 움직인다고 가정할 것인가?
> 

가장 단순한 선택은 classifier head에서 gradient를 계산하는 것이다. Classifier head가 linear layer이면 이 gradient는 빠르게 계산할 수 있고 해석도 쉽다. 다만 classifier head만 움직인다는 가정은 모델의 상위 표현을 고정한 채 마지막 decision layer만 조정하는 제한적인 compatibility를 본다. 따라서 classifier head gradient는 효율적인 baseline 또는 analytical control로 두는 것이 적절하다.

$$
z(x)=Wh(x)+b
$$

더 강한 후보는 last shared block gradient이다. Last shared block에서 gradient를 계산하면 fixed representation vector의 similarity만 보는 것이 아니라, target acceptance를 위해 high-level representation function이 어떻게 변해야 하는지를 반영할 수 있다.

Layer $\ell$에 대한 logit Jacobian을

$$
J_\ell(x)
=
\frac{\partial z(x)}{\partial \theta_\ell}
$$

라고 두면, class $c$에 대한 gradient는 다음처럼 쓸 수 있다.

$$
g_c^{(\ell)}(x)
=
J_\ell(x)^\top v_c(x)
$$

따라서 target-reference gradient inner product는

$$
g_c^{(\ell)}(x)^\top g_y^{(\ell)}(p)
=
v_c(x)^\top \Theta_\ell(x,p)v_y(p)
$$

로 쓸 수 있다. 여기서

$$
\Theta_\ell(x,p)=J_\ell(x)J_\ell(p)^\top
$$

이다. 즉, last shared block gradient는 단순한 representation similarity가 아니라, 모델의 high-level function이 target과 reference에 대해 어떻게 함께 변하는지를 반영한다.

따라서 실험에서는 다음 순서로 비교한다.

1. **FC head gradient**: 빠르고 해석 가능한 analytical control
2. **Last shared block gradient**: main candidate
3. **Multiple high-level blocks**: stronger variant 또는 ablation

정리하면, FC-head gradient는 빠르고 재현성이 좋은 기본 설정으로 사용할 수 있다. 반면 last shared block gradient와 all-parameter gradient는 target acceptance direction과 reference objectives 사이의 richer functional compatibility를 측정할 수 있으므로 gradient-space ablation에서 함께 검토한다.

---

## Step 8. Diagnostic Tests and Sanity Checks

Step 0–6에서는 제안 score의 논리적 구조를 정의하였다. 그러나 이 formulation이 실제로 OOD detection에 필요한 신호를 보는지는 실험적으로 확인해야 한다. Step 8의 목적은 최종 benchmark 성능을 보기 전에, 제안 score가 정말로 주장한 mechanism으로 동작하는지 검증하는 것이다.

핵심 질문은 다음과 같다.

> 제안 score는 단순 confidence, representation proximity, 또는 gradient norm이 아니라 reference-validated class hypothesis를 측정하고 있는가?
> 

이를 위해 다음 diagnostic tests를 수행한다.

### Gate 1. Acceptance direction sanity check

Step 1에서 정의한 acceptance direction이 실제로 target objective를 줄이는 방향인지 확인한다. 작은 step size $\eta$에 대해 다음이 성립해야 한다.

$$
L_{x,c}(\theta_0+\eta\rho_c(x))
<
L_{x,c}(\theta_0)
$$

이 gate는 acceptance direction 자체가 의도한 방향으로 정의되었는지를 확인한다.

### Gate 2. First-order sign prediction check

Step 2에서는 $K_c(x,p)>0$이면 target acceptance step이 reference loss를 1차적으로 줄이고, $K_c(x,p)<0$이면 reference loss를 증가시킨다고 해석하였다. 이를 실제 작은 step으로 확인한다.

$$
\Delta L_p
=
L_p(\theta_0+\eta\rho_c(x))-L_p(\theta_0)
$$

작은 $\eta$에서 다음 관계가 관찰되어야 한다.

$$
K_c(x,p)>0
\Rightarrow
\Delta L_p<0
$$

$$
K_c(x,p)<0
\Rightarrow
\Delta L_p>0
$$

이 gate는 $K_c(x,p)$가 단순 cosine이 아니라 first-order reference compatibility로 해석될 수 있는지 확인한다.

### Gate 3. Classifier gradient implementation check

Classifier gradient space를 사용할 때는 closed-form 계산과 dense gradient 계산이 같은 $K_c(x,p)$를 내는지 확인한다. 이 gate는 classifier implementation이 실제 gradient compatibility를 정확히 계산하는지 검증하기 위한 것이다.

$$
K_c^{\mathrm{closed}}(x,p)
\approx
K_c^{\mathrm{dense}}(x,p)
$$

이 gate는 score rule을 추가하는 실험이 아니라, classifier gradient 계산의 수치적 동등성을 확인하는 sanity check이다.

### Gate 4. Confidence-matched separation

제안 방법이 단순 confidence detector가 아니라면, 같은 confidence 수준 안에서도 clean ID, csID, near-OOD, far-OOD를 구분하는 추가 신호를 제공해야 한다.

이를 위해 target samples를 confidence bin으로 나눈다.

$$
\max_c q_c(x)\in \text{same bin}
$$

각 bin 안에서 다음 값들의 분포를 비교한다.

$$
V_{\hat c}(x),
\qquad
E_{\hat c}(x),
\qquad
S_{\mathrm{OOD}}(x)
$$

여기서 $\hat c=\arg\max_c q_c(x)$이다. 같은 confidence bin 안에서도 ID와 OOD가 분리된다면, 제안 score가 confidence 이상의 reference validation signal을 제공한다고 주장할 수 있다.

### Gate 5. External covariate-distance controlled separation

Gate 5는 아직 구현 우선순위가 낮은 deferred diagnostic이다. 목적은 RAE score가 단순한 visual/covariate similarity control 안에서도 추가적인 분리 신호를 제공하는지 확인하는 것이다. 이 gate는 RAE score rule 후보가 아니며, claim-bearing score를 만들기 위한 구성요소로 사용하지 않는다.

### Gate 6. Signed-support necessity check

Step 3에서 $V_c(x)$는 단순히 same-class reference가 other-class reference보다 더 잘 맞는지만 보지 않는다. Same-class reference에 대한 positive agreement 조건도 함께 요구한다.

$$
K_c(x,p^+)>0
\;\land\;
K_c(x,p^+)>K_c(x,p^-)
$$

따라서 positive support 조건이 실제로 필요한지 확인해야 한다. 이를 위해 다음 두 score를 비교한다.

$$
R_c(x)
=
\Pr\left[K_c(x,p^+)>K_c(x,p^-)\right]
$$

$$
V_c(x)
=
\Pr\left[K_c(x,p^+)>0\;\land\;K_c(x,p^+)>K_c(x,p^-)\right]
$$

특히 $R_c(x)$는 높지만 same-class agreement가 negative인 samples를 분석한다. 이러한 samples가 OOD에 많이 나타난다면, signed-support condition은 method의 핵심 요소로 정당화된다.

### Gate 7. Reference label shuffle test

Reference validation score가 labeled ID objectives에 의존한다면, reference labels를 shuffle했을 때 score가 무너져야 한다. 즉, $P_c$의 class label structure를 깨뜨리면 $V_c(x)$와 최종 OOD 성능이 크게 감소해야 한다.

이 test는 제안 방법이 단순히 reference pool과의 유사도를 보는 것이 아니라, labeled reference objectives와의 class-conditional agreement를 보고 있음을 확인하기 위한 sanity check이다.

### Gate 8. Reference size and sampling stability

Main formulation에서는 class $c$의 correctly classified clean ID samples 중에서 class-balanced uniform random sampling으로 reference set $P_c$를 구성한다. 따라서 reference size와 random seed에 대해 score가 얼마나 안정적인지 확인해야 한다.

Reference size는 다음과 같이 ablation한다.

$$
N\in\{4,8,16,32,64\}
$$

각 $N$에 대해 $V_c(x)$, $E_c(x)$, 그리고 OOD AUROC가 얼마나 안정적인지 본다. 작은 $N$에서도 성능과 ordering이 유지된다면, 제안 방법은 small class-wise reference set만으로도 작동한다는 실용적 장점을 주장할 수 있다.

### Gate 9. Gradient-space ablation

Step 7에서 논의한 gradient space 선택을 실제로 비교한다.

$$
\theta_\ell=\text{FC head}
$$

$$
\theta_\ell=\text{last shared block}
$$

$$
\theta_\ell=\text{multiple high-level blocks}
$$

각 gradient space에서 $V_c(x)$, $E_c(x)$, final OOD metric, 그리고 diagnostic gate 결과를 비교한다. classifier는 효율적인 기본 설정이고, last shared block과 all은 더 넓은 parameter compatibility를 보는 설정이다.

### Gate 10. Final score ablation

마지막으로 최종 score의 각 구성요소가 필요한지 확인한다.

Confidence only:

$$
q_{\max}(x)=\max_c q_c(x)
$$

Reference validation only:

$$
\max_c V_c(x)
$$

Joint evidence:

$$
\max_c q_c(x)V_c(x)
$$

Final OOD score:

$$
S_{\mathrm{OOD}}(x)
=
-
\log\left(\max_c q_c(x)V_c(x)+\epsilon\right)
$$

이 ablation은 $q_c(x)$와 $V_c(x)$가 각각 어떤 역할을 하는지 보여준다. 특히 high-confidence near-OOD에서 $V_c(x)$가 낮아지고, low-confidence but semantically ID-like csID에서 $V_c(x)$가 유지된다면, 제안 score의 핵심 claim을 뒷받침할 수 있다.

정리하면, Step 8의 diagnostic tests는 최종 benchmark 성능을 보기 전에 다음 세 가지를 검증하기 위한 것이다.

$$
\text{not merely confidence}
$$

$$
\text{actually reference-validated class hypothesis}
$$

이 gate들을 통과해야 제안 방법을 단순 handcrafted gradient score가 아니라, class hypothesis가 labeled ID reference objectives에 의해 first-order로 검증되는지를 측정하는 method로 주장할 수 있다.

---

## Step 1 보충설명

log-odds margin gradient를 더 전개하면 acceptance direction의 의미를 더 명확히 볼 수 있다. 

$$
m_c(x;\theta)
=
\log\sum_{j\neq c}\exp z_j(x;\theta)
-
z_c(x;\theta)
$$

따라서,

$$
\nabla_\theta m_c(x;\theta)
=
\nabla_\theta
\log\sum_{j\neq c}\exp z_j(x;\theta)
-
\nabla_\theta z_c(x;\theta)
$$

첫 번째 항은 다음과 같이 계산된다.

$$
\nabla_\theta
\log\sum_{j\neq c}\exp z_j(x;\theta)
=
\frac{
\sum_{j\neq c}
\exp z_j(x;\theta)
\nabla_\theta z_j(x;\theta)
}{
\sum_{j\neq c}\exp z_j(x;\theta)
}
$$

따라서,

$$
\nabla_\theta m_c(x;\theta)
=
\sum_{j\neq c}
\frac{
\exp z_j(x;\theta)
}{
\sum_{r\neq c}\exp z_r(x;\theta)
}
\nabla_\theta z_j(x;\theta)
-
\nabla_\theta z_c(x;\theta)
$$

Softmax probability를 이용하면, $j\neq c$에 대해

$$
\frac{
\exp z_j(x;\theta)
}{
\sum_{r\neq c}\exp z_r(x;\theta)
}
=
\frac{
q_j(x)
}{
1-q_c(x)
}
$$

따라서,

$$
\nabla_\theta m_c(x;\theta)
=
\sum_{j\neq c}
\frac{q_j(x)}{1-q_c(x)}
\nabla_\theta z_j(x;\theta)
-
\nabla_\theta z_c(x;\theta)
$$

이 식은 target $x$를 class $c$로 받아들이는 방향이 단순히 $z_c(x)$를 증가시키는 방향이 아님을 보여준다. 그것은 $z_c(x)$를 현재 competing classes의 logit mass에 비해 상대적으로 증가시키는 방향이다.

여기서 competing class $j$의 가중치는

$$
r_j^{(c)}(x)
=
\frac{q_j(x)}{1-q_c(x)},
\qquad
j\neq c
$$

이다. 즉, class $c$ 이외의 probability mass 안에서 class $j$가 차지하는 비율이 클수록, class $j$는 class $c$ hypothesis의 더 중요한 competitor로 작용한다.

따라서 first-order acceptance direction $\rho_c(x)$는 class-conditional하면서도 confusion-aware한 방향이다. 이는 현재 모델이 target $x$를 class $c$로 더 강하게 설명하기 위해, 어떤 competing classes를 상대적으로 낮추고 class $c$를 상대적으로 높여야 하는지를 반영한다.
