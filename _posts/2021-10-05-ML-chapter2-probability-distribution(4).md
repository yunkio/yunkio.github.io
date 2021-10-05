---
date: 2021-10-05
title: "Chapter 2. Probability Distribution (4) - 지수족, 비매개변수적 방법"
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

# 2.5 비매개변수적 방법

앞서 살펴본 분포들은 데이터 집합에 의해 결정되는 적은 수의 매개변수에 의해 조절되는 **매개변수적** 방법입니다. 이 방법의 중요한 한계는 관측된 데이터를 만들어낸 분포를 표현하기에 적절하지 않을 수도 있다는 점입니다. 

앞으로는 분포의 형태에 대해 적은 수의 가정만 하는 **비매개변수적** 방법인 밀도 추정 방법에 대해 살펴볼 것입니다. 

## 히스토그램 밀도 추정

단일 연속 변수 $x$에 초점을 맞춰 히스토그램 밀도 모델의 성질을 살펴보겠습니다. 표준 히스토그램 방법은 $x$를 너비 $\Delta_i$를 가진 계급 구간들로 나누고 구간 $i$에 속한 $x$의 숫자 $n_i$를 세는 것입니다.

$$p_i=\frac{n_i}{N\Delta_i}$$

여기서 $\int p(x)dx = 1$ 임을 쉽게 증명할 수 있습니다. 이를 통해 각 계급 구간의 너비에 대해서 사수인 밀도 $p(x)$가 주어집니다.  

![image](/assets/images/ml/Figure2.24.png){: width="400"}{: .align-center} 
Figure 2.24 히스토그램 밀도 추정의 예시
{: style="text-align: center; font-size:0.7em;"}

맨 위 그림에서 볼 수 있는 것처럼 $\Delta$가 아주 작을 경우에는 밀도 모델이 매우 뾰족하며 원 분포에는 포함되어 있지 않은 구조가 있습니다. 반면 $\Delta$가 클 경우 밀도 모델이 매끈하여 녹색 곡선의 양봉 형태를 표현하지 못 합니다. 

이와 같은 방법은 원 데이터 집합을 저장할 필요가 없으며, 데이터가 순차적으로 입력될 경우에도 쉽게 적용할 수 있다는 장점이 있습니다. 일/이차원의 데이터들을 빠르게 도식화하여 확인하는데 유용합니다.

하지만 대부분의 경우 좋은 방법은 아닙니다. 계급 구간의 가장자리로 인해 원 분포와는 상관없는 불연속면이 생긴다는 문제가 있습니다. 또 고차원 데이터를 다룰 경우 $D$차원 공간상의 각각의 변수들을 $M$개의 계급으로 나눌 경우 총 계급 구간의 숫자가 $M^D$개가 되는데, 차원의 저주의 한 예시가 됩니다.

치명적인 단점들에도 불구하고 두 가지 중요한 시사점이 있습니다. 첫 번째로 특정 위치상의 확률 밀도를 추정하기 위해서는 그 위치의 지역적 이웃 구간들에 존재하는 데이터를 고려해야 한다는 것입니다. 계급 구간의 너비라는 매개변수를 사용했는데 이는 지역적 구간의 공간적 크기를 결정짓는 데 활용되는 평활 매개변수에 해당합니다. 두 번째로는 좋은 결과를 얻기 위해서는 적절한 평활 매개변숫값을 정해야 합니다. 이 두 가지 시사점을 바탕으로 비매개변수적 밀도 추정법 두 가지를 살펴보겠습니다.

## 커널 밀도 추정

유클리디안 $D$차원 공간의 확률 밀도 $p(\mathbf{x})$로부터 관측값들을 추출하여 $p(\mathbf{x})$를 추정한다고 하겠습니다. 지역성에 대한 앞에서의 논의를 바탕으로 $\mathbf{x}$를 포함하는 작은 구역 $\text{R}$에 대해 고려해보겠습니다. 

$$P = \int_\mathcal{R}p(\mathbf{x})d\mathbf{x}$$

$p(\mathbf{x})$로부터 추출한 $N$개의 관측 데이터를 고려하면 각각의 데이터는 구역 $\text{R}$에 포함될 확률 $P$를 가지고 있습니다. 따라서 $N$개의 데이터 중 총 $K$개의 데이터들이 구역 $\text{R}$에 존재할 확률을 다음의 이항 분포로 구할 수 있습니다.

$$\text{Bin}(K\vert N,P) = \frac{N!}{K!(N-K)!}P^K(1-P)^{N-K}$$ 

여기서 데이터의 일부가 구역 $\text{R}$에 존재할 확률의 기댓값이 $\mathbb{E}[K/N] = P$임을 알 수 있습니다. 마찬가지로 이 평균값에 대한 분산은 $\text{var}[K/N] = P(1-P)/N$입니다. 이 분포는 큰 $N$값에 대해서 평균을 중심으로 날카롭고 뾰족한 모양을 그리므로 $K \simeq NP$ 입니다. 

구역 $\text{R}$이 충분히 작아서 확률 밀도 $p(\mathbf{x})$가 한 구역 내에서는 대략 상수라고 가정해 보면 $P \simeq p(\mathbf{x})V$를 얻게 됩니다. 여기서 $V$는 $\text{R}$의 부피입니다. 여기서 다음 형태의 밀도 추정식을 얻을 수 있습니다.

$$p(\mathbf{x}) = \frac{K}{NV}$$

위 식은 구역 $\text{R}$에 대한 두 가지의 서로 모순되는 가정을 지니고 있습니다. 구역 내에서 밀도가 대략 사수일 정도로 구역 $\text{R}$이 충분히 작다는 가정과 구역 내의 데이터 $K$가 뾰족한 이항 분포를 이룰 정도로 구역 $\text{R}$의 크기가 충분히 크다는 가정입니다.

위 식은 두 가지 방법으로 이용 가능합니다. $K$를 고정시키고 $V$의 값을 데이터로부터 구하는 방법이 그 중 하나이며, 이를 바탕으로 **K 최근접 이웃** *K nearest neighbour*방법을 도출해 낼 수 있습니다. 다른 한 가지는 $V$의 값을 고정시키고 데이터로부터 $K$를 구하는 것입니다. 이를 바탕으로 **커널** *kernel* 기반의 방법을 도출해 낼 수 있습니다. $N$이 증가함에 따라서 $V$가 적당히 감소하고 $K$가 증가한다고 가정하면 두 방법 모두 $N \rightarrow \infty$일 경우 실제 확률 밀도로 수렴합니다.

먼저 커널 방법에 대해 더 살펴보겠습니다. $\text{R}$ 구역이 우리가 확률 밀도를 구하고 싶은 포인트 $\mathbf{x}$ 주변의 작은 초입방체일 경우 이 구역에 포함되는 포인트의 수 $K$를 세기 위해 다음과 같은 함수를 정의하겠습니다.

$$k(\mathbf{u}) = \begin{cases} 
1, &\vert u_i\vert \leq 1/2, & i = 1, ..., D. \\
0, & \text{if else}
\end{cases}$$

위 식은 원점 주변의 단위 입방체를 의미합니다. 함수 $k(\mathbf{u})$는 **커널 함수** *kernel function*의 예 입니다. 현재 맥락에서는 **파젠 윈도우** *Parzen window* 라고 부르기도 합니다. 위 식으로부터 $\mathbf{x}$를 중심으로 한 변의 길이가 $h$인 입방체 안에 데이터 $\mathbf{x}_n$이 존재할 경우 $k((\mathbf{x} - \mathbf{x}_n)/h)$의 값이 1이고 아닐 경우에는 0이라는 사시를 알 수 있습니다. 따라서 이 입방체 안에 존재하는 총 데이터의 숫자는 다음과 같습니다.

$$K = \sum^N_{n=1}k\left(\frac{\mathbf{x}-\mathbf{x}_n}{h}\right)$$

이 식을 위의 밀도 추정식에 대입하면 $\mathbf{x}$에서의 밀도식을 구할 수 있습니다.

$$p(\mathbf{x}) = \frac1N\sum^N_{n=1}\frac1{h^D}k\left(\frac{\mathbf{x}-\mathbf{x}_n}{h}\right)$$

여기서 한 변의 길이가 $h$인 $D$차원상의 초입방체의 부피는 $V=h^D$라는 것을 이용하였습니다. $k(\mathbf{u})$의 대칭성을 이용하면 $\mathbf{x}$를 중심으로 한 하나의 입방체에 대해서가 아니라 $N$개의 데이터 포인트 $\mathbf{x}_n$들을 중심으로 한 $N$개의 입방체에 대한 합으로 이 식을 다시 해석할 수 있습니다.

위 커널 밀도 추정은 인공적인 불연속면이 생긴다는 문제점을 가집니다. 따라서 더 매끄러운 커널 함수를 이용해 매끄러운 모델을 구할 수 있습니다. 일반적으로는 가우시안 함수가 사용됩니다.

$$p(\mathbf{x})=\frac1N\sum^N_{n=1}\frac1{(2\pi h^2)^{D/2}}\exp\left\{-\frac{\Vert\mathbf{x}-\mathbf{x}_n\Vert^2}{2h^2}\right\}$$

여기서 $h$는 가우시안 성분의 표준 편차입니다. 이 모델은 각각의 데이터에 가우시안을 위치시키고 각자의 기여 정도를 전체 데이터 집합에 대해 합한 후 $N$으로 나누어 정규화한 것입니다. 

![image](/assets/images/ml/Figure2.25.png){: width="400"}{: .align-center} 
Figure 2.25 커널 밀도 모델을 히스토그램에 적용한 결과
{: style="text-align: center; font-size:0.7em;"}

위 그림에서 매개변수 $h$가 평활 매개변수로 작동하는 것을 확인할 수 있습니다. $h$를 최적화하는 것은 모델 복잡도를 결정하는 문제에 해당합니다. 결과로 구해지는 확률 분포가 0 이상의 값을 가지며 적분하였을 경우 1이 된다는 조건을 만족하면 어떤 함수든 위의 $\mathbf{x}$의 분포식의 커널 함수 $k(\mathbf{u})$로 사용 가능합니다. 이러한 밀도 모델들을 커널 밀도 추정, 혹은 **파젠** *Parzen* 추정이라고 부릅니다.

## 최근접 이웃 방법론

커널 밀도 추정의 문제점은 커널을 규정하는 매개변수 $h$가 모든 커널에 대해 동일하다는 것입니다. 최적의 $h$값은 데이터 공간상에서의 위치에 대해 종속적일 수 있습니다. 이 문제를 해결하는 것이 최근접 이웃 밀도 추정 방법론입니다. 

고정된 $K$값을 사용하고 데이터로부터 $V$ 값을 찾아낼 것이기 때문에 포인트 $\mathbf{x}$ 주변의 작은 구에서의 밀도 $p(\mathbf{x})$를 추정해야 합니다. 구가 정확하게 $K$개의 데이터를 포함할 때까지 반지름을 늘릴 것입니다. 밀도 추정식의 $V$를 이 결과에 해당하는 구의 부피로 설정하면 밀도 $p(\mathbf{x})$에 대한 추정값을 구할 수 있으며 이를 **K 최근접 이웃** *K nearest neighbour*방법이라고 합니다. 다양한 매개변수 $K$에 대해 최근접 이웃 방법론을 적용한 결과는 다음과 같습니다.

![image](/assets/images/ml/Figure2.26.png){: width="400"}{: .align-center} 
Figure 2.26 Figure 2.24와 2.25에 사용한 데이터에 KNN을 적용한 결과
{: style="text-align: center; font-size:0.7em;"}

$K$값이 평활화의 정도를 결정하게 됩니다. 이 모델은 모든 공간에 대해 적분을 취할 경우 발산하기 때문에 밀도 모델은 아닙니다.

이 방법은 밀도 추정 뿐만 아니라 분류 문제에도 사용될 수 있습니다. 각각의 클래스에 따로 K 최근접 이웃 밀도 추정법을 적용한 후 베이지안 정리를 사용할 것입니다. 각각의 클래스 $\text{C}_k$에 대해 $N_k$개의 데이터를 가지는 데이터 집합을 가정하겠습니다. 이때 전체 데이터의 수는 $N$ 입니다. 이 상황에서 새로운 포인트 $\mathbf{x}$를 분류하고 싶은 경우를 생각해보겠습니다.

첫 번째로, 클래스에 상관없이 정확히 $K$개의 데이터 포인트들을 포함하는 $\mathbf{x}$를 중심으로 한 구를 그립니다. 그 결과 이 구는 부피 $V$를 가지며 각 클래스 $\text{C}_k$로부터 각각 $K_k$만큼의 포인트를 포함하게 되었다고 하겠습니다. 그러면 각 클래스에 대한 밀도 추정을 할 수 있습니다.

$$p(\mathbf{x}\vert\mathcal{C}_k) = \frac{K_k}{N_kV}$$

이와 비슷하게 밀도는 다음처럼 주어집니다.

$$p(\mathbf{x})=\frac{K}{NV}$$

각 클래스의 사전 밀도는 다음과 같습니다.

$$p(\mathcal{C}_k)=\frac{N_k}{N}$$

베이지안 정리를 이용해서 위 세개의 식을 합치면 어떤 클래스에 속하는지에 대한 사후 확률을 구할 수 있습니다.

$$p(\mathcal{C_k}\vert\mathbf{x}) = \frac{p(\mathbf{x}\vert\mathcal{C}_k)p(\mathcal{C}_k)}{p(\mathbf{x})} = \frac{K_k}{K}$$

오분류의 확률을 최소화하고 싶다면 시험 데이터 $\mathbf{x}$를 가장 큰 사후 확률값 $K_k/K$를 가진 클래스에 포함시키면 됩니다. 새 데이터를 분류할 때는 우선 훈련 집합에서 새 데이터로부터 가장 가까운 $K$개의 데이터를 찾아낸 후, $K$개의 데이터들 중 가장 많은 데이터가 속해 있는 클래스에 데이터를 할당합니다. 

<div>
 <img src="/assets/images/ml/Figure2.28a.png" width="250" alt=""  /> 
 <img src="/assets/images/ml/Figure2.28b.png" width="250" alt=""  />
 <img src="/assets/images/ml/Figure2.28c.png" width="250" alt="" />
</div>
Figure 2.28 석유 흐름 데이터의 산포도 - 다양한 $K$값에 대한 K 최근접 이웃 알고리즘 적용
{: style="text-align: center; font-size:0.7em;"}

위 그림에서 볼 수 있듯이 $K$가 평활화 정도를 조절합니다. $K = 1$일 경우 **최근접 이웃** 방법이 되며 $N \rightarrow \infty$의 경우에 오차율이 최적 분류기를 통해서 얻을 수 있는 최소 가능 오차의 두 배가 넘지 않는다는 특징이 있습니다.

매개변수적 모델과 비매개변수적 모델을 살펴봄으로써 우리는 충분히 유연하면서도 모델의 복잡도가 훈련 집합의 크기와 독립적으로 조절될 수 있는 모델이 필요하다는 것을 알 수 있었습니다. 이후에는 이를 어떻게 달성할 지를 살펴보겠습니다.