---
date: 2021-10-05
title: "Chapter 2. Probability Distribution (4) - 지수족"
categories: 
  - 머신러닝과 패턴인식
tags: 
  - 머신러닝
  - 패턴인식
  - 확률
toc: true  
toc_sticky: true 
---
*본 글은 책 '패턴 인식과 머신 러닝'의 국내 출판본을 바탕으로 작성되었습니다.*

# 2.4 지수족

## 지수족의 성질

지금까지 살펴본 확률 분포들은 **지수족** *exponential family*라 불리는 더 넓은 분포 부류의 집합에 해당합니다. 지수족에 포함되는 분포들의 성질에 대해 알아보겠습니다.

### 형태
$\mathbf{x}$에 대한 지수족의 분포들은 매개변수 $\boldsymbol\eta$가 주어졌을 때 다음의 형태로 정의됩니다.

$$p(\mathbf{x}\vert\boldsymbol\eta) = h(\mathbf{x})g(\boldsymbol\eta)\exp\{\boldsymbol\eta^\text{T}\mathbf{u}(\mathbf{x})\}$$

여기서 $\mathbf{x}$는 스칼라, 벡터, 이산 변수, 연속 변수 등의 형태가 가능하며 $\boldsymbol\eta$는 분포의 **자연 매개변수** *natural parameter*라고 불립니다. $\mathbf{u}(\mathbf{x})$는 $\mathbf{x}$에 대한 어떤 함수입니다. $g(\boldsymbol\eta)$는 분포가 정규화되어 있도록 해주는 계수이며 다음을 만족합니다. 

$$g(\boldsymbol\eta)\int h(\mathbf{x})\exp\{\boldsymbol\eta^\text{T}\mathbf{u}\}d\mathbf{x} = 1$$

이제 앞서 배웠던 분포들을 지수족의 형태로 나타낼 수 있는지 하나씩 알아보겠습니다.

#### 베르누이 분포

$$p(x\vert\mu) = \text{Bern}(x\vert\mu) = \mu^x(1-\mu)^{1-x}$$

여기서 오른쪽 변을 로그의 지수로 표현하면 다음을 얻습니다.

$$p(x\vert\mu) = (1-\mu)\exp\left\{\ln\left(\frac{\mu}{1-\mu}\right)x\right\}$$

이 식을 지수족의 형태와 비교하면 다음을 확인할 수 있습니다.

$$\eta = \ln\left(\frac{\mu}{1-\mu}\right)$$

이 식을 $\mu$에 대해 풀면 $\mu = \sigma(\eta)$를 얻게 됩니다. 

$$\sigma(\eta) = \frac{1}{1+\exp(-\eta)}$${: .notice}

이 함수를 **로지스틱 시그모이드** *logistic sigmoid* 함수라고 부릅니다. 따라서 베르누이 분포를 지수족 표준 표현법의 형태로 표현하면 다음과 같습니다.

$$p(x\vert\eta) = \sigma(-\eta)\exp(\eta x)$$

정리하면 다음과 같습니다.

$$\begin{aligned}
u(x) &= x \\ h(x) &= 1 \\ g(\eta) &= \sigma(-\eta)
\end{aligned}$$

#### 다항 분포

다항 분포는 단일 관측 변수 $\mathbf{x}$에 대해서 다음 형태를 띱니다.

$$p(\mathbf{x}\vert\boldsymbol\mu) = \prod^M_{k=1}\mu_k^{x_k}=\exp\left\{\sum^M_{k=1}x_k\ln\mu_k\right\}$$

여기서 $\mathbf{x} = (x_1, ..., x_M)^\text{T}$ 입니다. 이를 지수족 분포의 형태로 표현하면 다음과 같습니다.

$$p(\mathbf{x}\vert\boldsymbol\eta) = \exp(\boldsymbol\eta^\text{T}\mathbf{x})$$

여기서 $\eta_k = \ln\mu_k$이며 $\boldsymbol\eta = (\eta_1, ..., \eta_M)^\text{T}$ 입니다. 역시 다음을 얻을 수 있습니다.

$$\begin{aligned}
\mathbf{u}(\mathbf{x}) &= \mathbf{x} \\ h(\mathbf{x}) &= 1 \\ g(\boldsymbol\eta) &= 1
\end{aligned}$$

여기서 매개변수 $\eta_k$들은 $\sum^M_{k=1
}\mu_k = 1$이기 때문에 독립적이지 않습니다. 이 제약 조건에 따라서 $M-1$개의 매개변수 $\mu_k$가 정해지면 나머지 한 매개변숫값은 고정적이게 됩니다. 따라서 이 분포를 $M-1$개의 매개변수로 표현해서 이 제약 조건을 제거해보겠습니다. 이 제약 조건을 이용하면 다항 분포를 다음과 같이 표현할 수 있습니다.

$$\exp\left\{\sum^M_{k=1}x_k\ln\mu_k\right\} = \exp\left\{\sum^{M-1}_{k=1}x_k\ln\left(\frac{\mu_k}{1-\sum^{M-1}_{j=1}\mu_j}\right) +\ln\left(1-\sum^{M-1}_{k=1}\mu_k\right)\right\}$$

이에 따라 다음을 확인할 수 있습니다.

$$\ln\left(\frac{\mu_k}{1-\sum_j\mu_j}\right) = \eta_k$$

$$\mu_k = \frac{\exp(\eta_k)}{1+\sum_j\exp(\eta_j)}$${: .notice}

위 식을 **소프트맥스** *softmax* 함수, 혹은 **정규화된 지수 함수** *normalized exponential function* 이라고 부릅니다. 이 표현법을 사용하면 다항 분포는 다음과 같습니다.

$$p(\mathbf{x}\vert\boldsymbol\eta)=\left(1+\sum^{M-1}_{k=1}\exp(\eta_k)\right)^{-1}\exp(\boldsymbol\eta^\text{T}\mathbf{x})$$

여기서 매개변수 벡터 $\boldsymbol\eta = (\eta_1, ..., \eta_{M-1})^\text{T}$ 이며 위 식은 지수족의 표준 형태입니다. 따라서 다음을 얻을 수 있습니다.

$$\begin{aligned}
\mathbf{u}(\mathbf{x}) &= \mathbf{x} \\ h(\mathbf{x}) &= 1 \\ g(\boldsymbol\eta) &= \left(1+\sum^{M-1}_{k=1}\exp(\eta_k)\right)^{-1}
\end{aligned}$$

#### 가우시안 분포

$$p(x\vert\mu,\sigma^2) = \frac1{(2\pi\sigma^2)^{1/2}}\exp\left\{-\frac{1}{2\sigma^2}x^2 + \frac\mu{\sigma^2}x-\frac{1}{2\sigma^2}\mu^2\right\}$$

단변량 가우시안의 경우는 위와 같습니다. 마찬가지로 표준 지수족의 형태로 바꿀 수 있으며 다음과 같습니다.

$$\begin{aligned}
\boldsymbol\eta &= \binom{\mu/\sigma^2}{-1/2\sigma^2}\\
\mathbf{u}(x) &= \binom{x}{x^2} \\ 
h(x) &= (2\pi)^{-1/2} \\ 
g(\boldsymbol\eta) &= (-2\eta_2)^{1/2}\exp\left(\frac{\eta^2_1}{4\eta_2}\right)
\end{aligned}$$

## 최대 가능도와 충분 통계량

최대 가능도 방법을 이용해서 지수족 분포에서 매개변수 벡터 $\boldsymbol\eta$를 추정할 수 있습니다. 앞서 지수족 분포의 정규화 계수에 대한 식에서 양변에 $\boldsymbol\eta$에 대해 기울기를 취하면 다음과 같습니다.

$$\nabla g(\boldsymbol\eta)\int h(\mathbf{x})\exp\{\boldsymbol\eta^\text{T}\mathbf{u}(\mathbf{x})\}d\mathbf{x}
\\ +  g(\boldsymbol\eta)\int h(\mathbf{x})\exp\{\boldsymbol\eta^\text{T}\mathbf{u}(\mathbf{x})\}\mathbf{u}(\mathbf{x})d\mathbf{x} = 0$$

이를 재배열하면 다음과 같습니다.

$$-\frac{1}{g(\boldsymbol\eta)}\nabla g(\boldsymbol\eta)=g(\boldsymbol\eta)\int h(\mathbf{x})\exp\{\boldsymbol\eta^\text{T}\mathbf{u}(\mathbf{x})\}\mathbf{u}(\mathbf{x})d\mathbf{x}=\mathbb{E}[\mathbf{u}(\mathbf{x})]$$

따라서 다음과 같습니다.

$$-\nabla\ln g(\boldsymbol\eta) = \mathbb{E}[\mathbf{u}(\mathbf{x})]$$

이제 독립적이고 동일하게 분포된 데이터 집합 $\mathbf{X} = \{\mathbf{x}_1, ..., \mathbf{x}_N\}$에 대한 가능도 함수를 살펴보겠습니다.

$$p(\mathbf{X}\vert\boldsymbol\eta) = \left(\prod^N_{n=1}h(\mathbf{x}_n)\right)g(\boldsymbol\eta)^N\exp\left\{\boldsymbol\eta^\text{T}\sum^N_{n=1}\mathbf{u}(\mathbf{x}_n)\right\}$$

$\ln p(\mathbf{X}\vert\boldsymbol\eta)$의 $\boldsymbol\eta$에 대한 기울기를 0으로 놓으면 최대 가능도 추정값 $\boldsymbol\eta_\text{ML}$에 대해서 다음의 조건이 만족됩니다. 

$$-\nabla\ln g(\boldsymbol\eta_{\text{ML}}) = \frac1N\sum^N_{n=1}\mathbf{u}(\mathbf{x}_n)$$

위 식을 풀어서 $\boldsymbol\eta_\text{ML}$을 구할 수 있습니다. 최대 가능도 추정값에 대한 해는 $\sum_n\mathbf{u}(\mathbf{x}_n)$을 통해서만 데이터와 연관되어 있습니다. 따라서 이를 **충분 통계량** 이라고 합니다. 예를 들면 베르누이 분포의 경우 $\mathbf{u}(x)$는 $\mathbf{x}$만 있으면 주어지기 때문에 $\{x_n\}$의 합만 가지고 있으면 됩니다.

또한 $N \rightarrow \infty$의 경우 오른쪽 변이 $\mathbb{E}[\mathbf{u}(\mathbf{x})]$가 되므로 $\boldsymbol\eta_\text{ML}$의 값이 $\boldsymbol\eta$와 같음을 알 수 있습니다.

## 켤례 사전 분포

앞서 살펴봤듯이 확률 분포 $p(\mathbf{x}\vert\boldsymbol\eta)$에 대해서 가능도 함수에 대해 켤레 사전 분포 $p(\boldsymbol\eta)$를 찾는 것이 가능하며, 그 결과 사후 분포 역시 사전 분포와 같은 함수적 형태를 가집니다. 지수족 분포의 일반적 형태에 대해서도 켤레 사전 분포가 존재합니다.

$$p(\boldsymbol\eta\vert\boldsymbol\chi,\nu) = f(\boldsymbol\chi, \nu)g(\boldsymbol\eta)^\nu\exp\{\nu\boldsymbol\eta^\text{T}\boldsymbol\chi\}$$

여기서 $f(\boldsymbol\chi, \nu)$는 정규화 계수입니다. 이 형태가 켤레라는 것을 확인하기 위해 이 식을 위의 $\mathbf{X}$에 대한 가능도 함수에 곱해보겠습니다.

$$p(\boldsymbol\eta\vert\mathbf{X}, \boldsymbol\chi, \nu) \propto g(\boldsymbol\eta)^{\nu+N}\exp\left\{\boldsymbol\eta^\text{T}\left(\sum^N_{n=1}\mathbf{u}(\mathbf{x}_n)+\nu\boldsymbol\chi\right)\right\}$$

위 식이 사전 분포식과 같은 함수적 형태를 가짐을 알 수 있습니다. 따라서 켤레 성질을 가지고 있습니다. 여기서의 매개변수 $\nu$는 사전 분포에서의 가상의 관측값 개수라고 해석할 수 있습니다. 이 관측값 각각은 충분 통계량 $\mathbf{u}(\mathbf{x})$에 해당하는 값으로 $\boldsymbol\chi$를 가지게 됩니다.

## 무정보적 사전 분포

사전 정보가 어떤 형태의 분포로 표현되어야 하는지 알기 어려운 경우가 있습니다. 이런 경우에는 **무정보적 사전 분포** *noninformative prior*가 사용됩니다. 이를 통해 사후 분포에 대한 사전 분포의 영햐력을 최소화 할 수 있습니다.

이산 변수라면 각 상태의 사전 확률을 같게 만드는 방법을 사용할 수 있습니다. 하지만 연속 변수라면 이러한 방법을 사용할 수 없는 두 가지 이유가 있습니다. 

첫 번째로 정의역이 무한하므로 적분하면 값이 발산해버립니다. 이런 경우를 **부적합** *improper* 분포라고 합니다. 이런 경우 사후 분포가 **적합** *proper* 분포라는 조건 하에 사용이 가능합니다. 가령 가우시안 분포의 평균에 대한 사전 분포로 균일 분포를 사용한다면 최소 하나의 데이터 포인터가 주어진 후부터는 사후 분포가 적합 분포가 됩니다.

두 번째로 비선형 변수 변환을 시행할 때 문제가 될 수 있습니다. 만약 $h(\lambda)$가 상수고 $\lambda = \eta^2$의 변수 변환을 한다면 $\widehat{h}(\eta)=h(\eta^2)$ 역시 상수일 것입니다. 하지만 밀도 $p_\lambda(\lambda)$가 상수가 되도록 선택하면 $\eta$의 밀도는 다음과 같습니다.

$$p_\eta(\eta) = p_\lambda(\lambda)\left\vert\frac{d\lambda}{d\eta}\right\vert = p_\lambda(\eta^2)2\eta \propto \eta$$

따라서 $\eta$에 대한 밀도는 상수가 아니게 됩니다. 최대 가능도 방법을 사용할 때는 가능도 함수 $p(x\vert\lambda)$가 $\lambda$에 대한 단순한 함수이기 때문에 매개변수화를 자유롭게 사용할 수 있어서 문제가 되지 않지만, 사전 분포로 상수를 선택할 경우에는 매개변수를 적절히 표현해야 합니다.

무정보적 사전 분포에 대한 두 가지 간단한 예를 살펴보겠습니다.

### 위치 매개변수

$$p(x\vert\mu)=f(x-\mu)$$

밀도가 위와 같은 형태를 가질 경우 매개변수 $\mu$를 **위치 매개변수** *location parameter*라고 합니다. 이러한 형태를 지닌 밀도족들은 $x$를 $\widehat{x}=x+c$와 같이 상수만큼 이동시킬 수 있습니다. 이를 **이동 불변성** *translation invariance*이라고 합니다.

$$p(\widehat{x}\vert\widehat{\mu}) = f(\widehat{x}-\widehat{\mu})$$

밀도가 새로운 변수하에서도 원래 밀도와 같은 형태를 띠게 되기 때문에 어떤 원점을 선택하느냐와 독립적입니다. 이런 경우 이동 불변성을 표현할 수 있는 분포를 사전 분포로 선택해야 합니다.

$$\int^B_Ap(\mu)d\mu = \int^{B-c}_{A-c}p(\mu)d\mu = \int^B_Ap(\mu-c)d\mu$$

이 성질은 모든 A와 B에 대해 만족해야 합니다. 따라서 다음을 얻게 됩니다.

$$p(\mu-c) = p(\mu)$$

이것은 $p(\mu)$가 상수라는 것을 의미합니다. 가우시안 분포의 평균 $\mu$도 위치 매개변수의 예 입니다. 

### 척도 매개변수

$$p(x\vert\sigma)=\frac{1}{\sigma}f(\frac{x}\sigma)$$

여기서 $\sigma > 0$ 입니다. $f(x)$가 올바르게 정규화되었다는 가정하에 위 식은 정규화된 밀도입니다. 여기서 $\sigma$는 **척도 매개변수** *scale parameter*라고 불립니다. 이 밀도는 **크기 불변성** *scale invariance*를 가집니다. 

$$p(\widehat{x}\vert\widehat{\sigma}) = \frac1{\widehat{\sigma}}f\left(\frac{\widehat{x}}{\widehat{\sigma}}\right)$$

이 변환은 크기 변환에 해당합니다. 예를 들어 $x$가 미터일 경우 킬로미터로 변환하는 것이 이에 해당합니다.

$$\int^B_Ap(\sigma)d\sigma = \int^{B/c}_{A/c}p(\sigma)d\sigma = \int^B_Ap\left(\frac1c\sigma\right)\frac1cd\sigma$$

모든 $A$와 $B$에 대해서 위 성질이 만족되어야 하므로 다음과 같습니다.

$$p(\sigma)=p\left(\frac1c\sigma\right)\frac1c$$

따라서 $p(\sigma) \propto 1/\sigma$여야 합니다. 또한 $0 \leq \sigma \leq \infty$ 이기 때문에 이 분포는 부적합 분포입니다. 척도 매개변수의 예시로는 위치 매개변수 $\mu$를 고려한 후의 가우시안 분포의 표준 편차 $\sigma$가 있습니다.


