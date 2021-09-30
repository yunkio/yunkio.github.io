---
date: 2021-09-30
title: "Chapter 2. Probability Distribution (1) - 이산 확률 변수"
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

# 2.0 프롤로그

앞 글에서 확률론이 패턴 인식에서 굉장히 중요한 위치를 차지하고 있다는 점을 알 수 있었습니다. 이번 장에서는 중요한 역할을 차지하는 다양한 확률 분포에 대해 살펴볼 것입니다. 더해서, 베이지안 추론 등의 중요한 통계적 개념도 살펴보겠습니다.

분포의 중요한 역할 중 하나는 **밀도 추정** 문제입니다. 밀도 추정이란 한정된 수의 관찰 집합 $\mathbf{x}_1, ..., \mathbf{x}_N$ 이 주어졌을 때 확률 변수 $\mathbf{x}$의 확률 분포 $p(\mathbf{x})$를 모델링 하는 것입니다.

이 장에서는 이산 확률 변수의 이항 분포와 다항 분포에 대해 살펴보고, 연속 확률 변수의 가우시안 분포에 대해서도 살펴보겠습니다. 이 분포들은 **매개변수적** *parametric* 분포의 예 입니다. 예를 들면 가우시안 분포는 평균과 분산과 같은 적은 수의 조절 가능한 매개변수에 의해 분포가 결정됩니다. 빈도적 관점에서는 가능도 함수 등의 최적화 기준을 활용해서 매개변수를 찾습니다. 반면, 베이지안 관점에서는 베이지안 정리를 활용해 매개변수에 대한 사전 분포를 바탕으로 관측치가 주어졌을 때의 해당 사후 분포를 계산합니다. 

또한 **켤레** *conjugate* 사전 확률에 대해서도 알아볼 것입니다. 사후 확류이 사전 확률과 같은 함수적 형태를 띠도록 만들어 줍니다. 예를 들면 다항 분포 매개변수의 켤레 사전 확률은 **디리클레 분포** *Dirichlet distribution* 입니다. 이 분포들은 **지수족** *exponential family*에 속하는데, 지수족 분포에 대한 중요한 성질들도 살펴볼 것입니다.

매개변수적인 접근법의 한계 중 하나는 분포가 특정 함수의 형태를 띠고 있다고 가정한다는 것입니다. 실제 사례에서는 이 가정이 적절하지 않은 경우가 많습니다. 이런 경우에는 **비매개변수적** *nonparametric* 밀도 추정 방식이 대안으로 활용될 수 있습니다. 이러한 방식은 여전히 매개변수를 가지고 있긴 하지만 분포 형태가 아니라 모델의 복잡도에 영향을 줍니다. 

# 2.1 이산 확률 변수

## 베르누이 분포

우선 하나의 이진 확률 변수 $x \in \{0, 1\}$을 가정해보겠습니다. 가령 동전을 던졌을 때 앞면이 나오는 경우를 ${x = 1}$, 뒷면이 나오는 경우를 ${x = 0}$으로 나타내겠습니다. 이때 동전의 형태가 망가져 앞면과 뒷면이 나올 확률이 다르다고 하겠습니다. ${x=1}$일 확률을 다음과 같이 나타낼 수 있습니다.

$$ p(x=1\vert\mu)=\mu$$

여기서 $0\leq\mu\leq1$ 이며 $p(x=0\vert\mu) = 1 - \mu$ 가 됩니다. 따라서 $x$에 대한 확률 분포를 다음과 같이 적습니다.

$$\text{Bern}(x\vert\mu)=\mu^x(1-\mu)^{1-x}$$

이것을 **베르누이 분포** *Bernoulli distribution* 이라고 합니다. 정규화되어 있으며 이때 평균은 $\mathbb{E}[x] = \mu$, 분산은 $\text{var}[x] = \mu(1-\mu)$ 입니다. 	
 
 $x$의 관측값 데이터 집합 $\mathcal{D} = \{x_1,...,x_N\}$이 주어졌다고 했을 때 관측값들이 $p(x\vert\mu)$에서 독립적으로 추출되었다는 가정하에 $\mu$의 함수로써 가능도 함수를 구성할 수 있습니다.  또한 빈도적 관점에서 보면 가능도 함수를 최대화하는 $\mu$를 추정할 수 있습니다.  베르누이 분포의 로그 가능도 함수는 다음과 같습니다.

$$p(\mathcal{D}\vert\mu) = \prod^N_{n=1}p(x_n\vert\mu)=\prod^N_{n=1}\mu^{x_n}(1-\mu)^{1-{x_n}}
\\ \ln p(\mathcal{D}\vert\mu)=\sum^N_{n=1}\ln p(x_n\vert\mu) = \sum^N_{n=1}\left\{x_n\ln\mu+(1-x_n)\ln(1-\mu)\right\}$$

식을 보면 로그 가능도 함수는 오직 관측값들의 합인 $\sum_nx_n$을 통해서만 $N$개의 관측값 $x_n$과 연관됩니다. 이 합은 **충분 통계량** *sufficient statistic* 의 예시입니다.  베르누이 분포의 로그 가능도 함수 $\ln p(\mathcal{D}\vert\mu)$을 $\mu$에 대해 미분하고 이를 0과 같다고 놓으면 다음과 같은 최대 가능도 추정값을 구할 수 있습니다.

$$\mu_\text{ML}=\frac1N\sum^N_{n=1}x_n$$

위 식은 **표본 평균** 이라고 부릅니다. 데이터에서 $x=1$인 관찰값의 수를 $m$이라고 하면 위 식을 다음의 형태로 다시 적을 수 있습니다. 

$$ \mu_\text{ML}=\frac mN$$

이 식을 예로 들어 설명해보면, 동전을 세 번 던져서 세 번 다 앞면이 나왔다면 $N=m=3$ 이므로 $\mu_\text{ML}=1$입니다.  즉 무조건 앞면이 나온다는 의미가 되며, 과적합의 사례라고 볼 수 있습니다. $\mu$에 대한 사전 분포를 활용하여 더 나은 결과를 도출할 수 있습니다. 이에 대해서는 뒤에서 더 살펴보도록 하겠습니다.

### 이항 분포

크기 $N$의 데이터가 주어졌을 때 $x=1$인 관측값의 수 $m$에 대한 분포를 생각해 볼 수 있고 이를 **이항 분포** *binomial distribution* 이라고 합니다. 이항 분포는 $\mu^m(1-\mu)^{N-m}$에 비례합니다. 정규화 계수를 구하기 위해서는 동전 던지기를 $N$번 했을 때 앞면이 $m$번 나올 수 있는 모든 가짓수를 구해야 합니다.

$$\text{Bin}(m\vert N,\mu) = \binom{N}{m}\mu^m(1-\mu)^{N-m}$$

$$\binom{N}{m} = \frac{N!}{(N-m)!m!}$$

![image](/assets/images/ml/Figure2.1.png){: width="400"}{: .align-center} 
Figure 2.1 $N = 10$, $\mu=0.25$의 이항 분포 히스토그램
{: style="text-align: center; font-size:0.7em;"}

$\binom{N}{m}$은 $N$개의 물체 중 $m$개의 물체를 선별하는 가짓수를 의미합니다. 사건들이 서로 독립일 경우 사건들의 합의 평균값은 평균값들의 합과 같으며 사건들의 합의 분산은 분산들의 합과 같습니다. 따라서 $m=x_1 + ... + x_N$이기 때문에 각 관측값의 평균과 분산은 다음과 같습니다.

$$ \begin{aligned}
\mathbb{E}[m] \equiv \sum^N_{m=0}m\text{Bin}(m\vert N,\mu) &= N\mu
\\ \text{var}[m] \equiv \sum^N_{m=0}(m-\mathbb{E}[m])^2\text{Bin}(m\vert N,\mu) &= N\mu(1-\mu) 
\end{aligned}$$

## 베타 분포

앞서 살펴본 베르누이 분포 및 이항 분포에서는 데이터의 수가 적을 때 과적합이 일어나기 쉽습니다. 이 문제에 대해 베이지안적으로 접근하여 매개변수 $\mu$에 대한 사전 분포 $p(\mu)$를 도입해보겠습니다. 

가능도 함수가 $\mu^x(1-\mu)^{1-x}$의 형태를 가지는 인자들의 곱의 형태를 띄고 있으므로 $\mu$와 $(1-\mu)$의 거듭제곱에 비례하는 형태를 사전 분포로 선택한다면 사전 확률과 가능도 함수의 곱에 비례하는 사후 분포 역시 사전 분포와 같은 함수적 형태를 가지게 됩니다. 이러한 성질을 **켤레성** *counjugacy*라고 합니다. 우리는 사전 분포로 **베타 분포**를 사용해보겠습니다.

$$\begin{aligned}\text{Beta}(\mu\vert a,b) &= \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}\mu^{a-1}(1-\mu)^{b-1} \\
\Gamma(x) &= \int^\infty_0u^{x-1}e^{-u}du\end{aligned}$$

위 식의 계수들은 베타 분포가 정규화되도록 합니다. 

$$\int^1_0\text{Beta}(\mu\vert a,b)d\mu = 1$$

또한 베타 분포의 평균과 분산은 다음과 같습니다.

$$
\begin{aligned}
\mathbb{E}[\mu] &= \frac{a}{a+b} \\
\text{var}[\mu] &= \frac{ab}{(a+b)^2(a+b+1)}
\end{aligned}
$$

<figure class="half">
  <a href="/assets/images/ml/Figure2.2a.png">
  <img src="/assets/images/ml/Figure2.2a.png" width="200"></a>

  <a href="/assets/images/ml/Figure2.2b.png">
  <img src="/assets/images/ml/Figure2.2b.png" width="200"></a>
</figure>
<figure class="half">
  <a href="/assets/images/ml/Figure2.2c.png">
  <img src="/assets/images/ml/Figure2.2c.png" width="200"></a>

  <a href="/assets/images/ml/Figure2.2d.png">
  <img src="/assets/images/ml/Figure2.2d.png" width="200"></a>
</figure>

Figure 2.2 초매개변수 $a$와 $b$에 따른 베타 분포 $\text{Beta}(\mu\vert a,b)$의 그래프 
{: style="text-align: center; font-size:0.7em;"}

여기서 $a$와 $b$가 매개변수 $\mu$의 분포를 조절하기 때문에 이 변수들은 **초매개변수** *hyperparemeter* 라고 불립니다. 

이제 베타 사전 분포와 이항 가늠도 함수를 곱한 후 정규화를 시행하면 $\mu$의 사후 분포를 구할 수 있습니다. $\mu$와 관련된 인자만 남겨서 사후 분포의 형태를 알아보겠습니다.

$$p(\mu\vert m,l,a,b) \propto \mu^{m+a-1}(1-\mu)^{l+b-1}$$

여기서 $l = N-m$으로 동전 던지기 예시에서는 '뒷면'을 의미합니다. 위의 사후 분포 식은 사전 분포와 $\mu$에 대해서 같은 함수적 종속성을 가집니다.  즉 가능도 함수에 대해서 사전 분포가 켤레적인 성질을 가집니다. 실제로 사후 분포는 또 다른 베타 분포라고 볼 수 있습니다. 이 사후 분포 식을 베타 분포 식이랑 비교해서 정규화 계수를 얻게 됩니다.

$$p(\mu\vert m,l,a,b) = \frac{\Gamma(m+a+l+b)}{\Gamma(m+a)\Gamma(l+b)}\mu^{m+a-1}(1-\mu)^{l+b-1}$$

이 식을 해석해보겠습니다.  $x=1$인 값이 $m$개 있고 $x=0$인 값이 $l$개 있는 데이터 집합을 관찰한 결과 사전 분포와 비교했을 때 사후 분포에는 $a$의 값이 $m$ 만큼, $b$의 값이 $l$만큼 증가했습니다. 이 사실로부터 사전 분포의 $a$와 $b$를 각각 $x=1$, $x=0$에 경우에 대한 **유효 관찰수** *effective number of observations*로 해석할 수 있습니다. 

만약 우리가 추가적으로 관측치를 얻으면 지금의 사후 분포를 새로운 사전 분포로 사용할 수도 있습니다. 이를 확인하기 위해 관측치를 한 번에 하나씩 받아서 사후 분포를 업데이트하는 방식을 생각해 보겠습니다. 매 업데이트 단계에서 새로운 관측치에 해당하는 가능도 함수를 곱하고 그 다음에 정규화를 시행해서 새로운 사후 분포를 얻는 과정으로 진행됩니다. 각 단계에서 사후 분포는 $x=1$과 $x=0$에 해당하는 관측치의 숫자가 새로운 $a$와 $b$로 주어지는 베타 분포에 해당하므로, 관측치의 값에 따라 $a$ 또는 $b$를 1씩 증가시키면 됩니다.

<div>
 <img src="/assets/images/ml/Figure2.3a.png" width="250" alt=""  /> 
 <img src="/assets/images/ml/Figure2.3b.png" width="250" alt=""  />
 <img src="/assets/images/ml/Figure2.3c.png" width="250" alt="" />
</div>

Figure 2.3 베이지안 추론의 시각화 
{: style="text-align: center; font-size:0.7em;"}

위 그림에서 사전 분포는 $a=2$, $b=2$인 베타 분포로 주어지며 관측치로 $x=1$인 하나의 관측값에 해당하는 가능도 함수를 받아 $a=3$, $b=2$인 새로운 베타 분포를 얻는다.

# 2.2 다항 변수
