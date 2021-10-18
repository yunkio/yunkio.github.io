---
date: 2021-10-18
title: "Chapter 3. Linear Regression (2) - 편향 분산 분해, 베이지안 선형 회귀"
categories: 
  - 머신러닝과 패턴인식
tags: 
  - 머신러닝
  - 패턴인식
  - 회귀
toc: true  
toc_sticky: true 
---
*본 글은 책 '패턴 인식과 머신 러닝'의 국내 출판본을 바탕으로 작성되었습니다.*

# 3.2 편향 분산 분해

앞에서는 기저 함수들의 형태와 종류가 둘 다 고정되어 있다고 가정하였습니다. 앞서서 과적합 문제를 피하기 위해 정규화 항을 사용할 때 발생하는 트레이드 오프를 살펴보았습니다. 과적합 문제를 피하기 위해 기저 함수의 수를 제한하면 모델의 유연성에 제약을 가하게 되고, 정규화항을 사용하면 정규화 계수 $\lambda$를 정하는 까다로운 문제가 추가됩니다. 최대 가능도 함수를 사용할 경우 이처럼 과적합 문제를 해결하기 어렵습니다. 

베이지안 방법론을 바탕으로 각각의 매개변수들을 주변화할 경우에는 이러한 문제가 발생하지 않습니다. 뒤에서 베이지안 관점에서의 모델 복잡도에 대해 더 깊이 살펴볼 것입니다. 하지만 그 전에 빈도주의 관점의 모델 복잡도에 대해 살펴보겠습니다. 이를 **편향 분산 트레이드 오프** *bias variance trade off*라고 합니다. 

1장에서 조건부 분포 $p(t\vert\mathbf{x})$가 주어졌을 경우 최적 예측값에 도달하도록 하는 다양한 오류 함수들에 대해 살펴보았으며, 가장 많이 사용되는 오류 함수는 제곱 오류 함수였습니다. 이 경우 최적의 예측치 $h(\mathbf{x})$는 다음과 같은 조건부 기댓값으로 주어집니다.

$$h(\mathbf{x})=\mathbb{E}[t\vert\mathbf{x}]=\int tp(t\vert\mathbf{x})dt$$

기대 제곱 오류는 다음의 형태로 적을 수 있었습니다.

$$\mathbb{E}[L] = \int\{y(\mathbf{x})-h(\mathbf{x})\}^2p(\mathbf{x})d\mathbf{x} +\int\int\{h(\mathbf{x})-t\}^2p(\mathbf{x},t)d\mathbf{x}dt$$

두 번째 항은 데이터의 내재적인 노이즈라고 볼 수 있으며 $y(\mathbf{x})$와는 독립적입니다. 첫 번째 항은 함수 $y(\mathbf{x})$로 어떤 것을 선택하느냐에 따라 결정되므로 우리는 이 값을 최소화하는 $y(\mathbf{x})$를 찾아야 합니다.  데이터가 무수히 많고 계산 자원에 제한이 없다면 최적의 함수를 찾는 것이 가능하지만, 현실적으로는 회귀 함수 $h(\mathbf{x})$를 정확히 알 수 없습니다. 

위 식의 첫 번째 항을 특정 데이터 집합 $\mathcal{D}$에 대해서 나타내면 다음과 같습니다.

$$\{y(\mathbf{x};\mathcal{D})-h(\mathbf{x})\}^2$$

주어진 데이터 집합$\mathcal{D}$에 대해서 학습 알고리즘을 실행해 예측함수 $y(\mathbf{x};\mathcal{D})$를 구할 수 있습니다. 이 값은 특정 데이터 집합 $\mathcal{D}$에 대해 종속적입니다.  따라서 각 데이터 집합으로부터 구한 값들을 평균내어 사용할 수 있습니다. 괄호 안에 $\mathbb{E}_\mathcal{D}[y(\mathbf{x};\mathcal{D})]$를 더하고 뺀 후 전개하면 다음과 같습니다.

$$\mathbb{E}_\mathcal{D}[\{y(\mathbf{x};\mathcal{D})-h(\mathbf{x})\}^2] = \\\{\mathbb{E}_\mathcal{D}[y(\mathbf{x};\mathcal{D})]-h(\mathbf{x})\}^2 + \mathbb{E}_\mathcal{D}[\{y(\mathbf{x};\mathcal{D})-\mathbb{E}_\mathcal{D}[y(\mathbf{x};\mathcal{D}\}^2]$$

위 식에서 첫 번째 항은 **편향**의 제곱이며 전체 데이터 집합들에 대해 평균 예측이 회귀 함수와 얼마나 차이나는지를 표현한 것입니다. 두 번째 항은 **분산** 입니다. 각각의 데이터 집합에서의 해가 전체 평균에서 얼마나 차이가 나는지를 표현한 것입니다. 어떤 데이터 집합을 선택하는지에 대한 함수 $y(\mathbf{x};\mathcal{D})$의 민감도이기도 합니다. 기대 제곱 오류는 다음과 같이 분해될 수 있습니다.

$$기대 오류 = (편향)^2 + 분산 + 노이즈$$

여기서 각각은 다음과 같습니다.

$$
\begin{aligned}
(편향)^2 &= \int\{\mathbb{E}_\mathcal{D}[y(\mathbf{x};\mathcal{D})]-h(\mathbf{x})\}^2p(\mathbf{x})d\mathbf{x} \\
분산 &=  \int\mathbb{E}_\mathcal{D}[\{y(\mathbf{x};\mathcal{D})-\mathbb{E}_\mathcal{D}[y(\mathbf{x};\mathcal{D}\}^2]p(\mathbf{x})d\mathbf{x} \\
노이즈 &=  \int\int\{h(\mathbf{x})-t\}^2p(\mathbf{x},t)d\mathbf{x}dt
\end{aligned}
$$

편향과 모델은 트레이드 오프 관계입니다. 유연한 모델은 낮은 편향값과 높은 분산값을 가지며, 엄격한 모델은 그 반대입니다. 이 밸런스를 좋게 가지는 모델이 최적의 예측치를 내게 됩니다. 

<figure class="half">
  <a href="/assets/images/ml/Figure3.5a.png">
  <img src="/assets/images/ml/Figure3.5a.png" width="150"></a>

  <a href="/assets/images/ml/Figure3.5b.png">
  <img src="/assets/images/ml/Figure3.5b.png" width="150"></a>
</figure>
<figure class="half">
  <a href="/assets/images/ml/Figure3.5c.png">
  <img src="/assets/images/ml/Figure3.5c.png" width="150"></a>

  <a href="/assets/images/ml/Figure3.5d.png">
  <img src="/assets/images/ml/Figure3.5d.png" width="150"></a>
</figure>
<figure class="half">
  <a href="/assets/images/ml/Figure3.5e.png">
  <img src="/assets/images/ml/Figure3.5e.png" width="150"></a>

  <a href="/assets/images/ml/Figure3.5f.png">
  <img src="/assets/images/ml/Figure3.5f.png" width="150"></a>
</figure>

Figure 3.5 정규화 매개변수 $\lambda$에 의해 결정되는 모델 복잡도가 편향과 분산에 미치는 영향
{: style="text-align: center; font-size:0.7em;"}

위 그림은 임의의 함수로부터 각각 25개의 데이터를 가지는 100개의 데이터 집합을 만든 후 모델을 피팅한 결과입니다. 맨 위의 행은 큰 정규화 계수 $\lambda$를 사용했을 때 낮은 분산과 높은 편향을 가지는 경우를 보여주며, 밑은 반대입니다. 복잡한 모델의 여러 결과들을 평균 내는 것이 회귀 함수에 대해 좋은 근사를 보여줌을 확인할 수 있습니다.

이 예시에 대해서 편향 분산 트레이드 오프를 수량적으로 확인할 수 있습니다. 평균 예측치는 다음으로부터 얻을 수 있습니다.

$$\overline{y}(x) = \frac1L\sum^L_{l=1}y^{(l)}(x)$$

또한 적분된 제곱 편향값과 적분된 분산값은 다음과 같습니다.

$$\begin{aligned}
(편향)^2 &= \frac1N\sum^N_{n=1}\{\overline{y}(x_n)-h(x_n)\}^2 \\
분산 &= \frac1N\sum^N_{n=1}\frac1L\sum^{L}_{l=1}\{y^{(l)}(x_n)-\overline{y}(x_n)\}^2
\end{aligned}$$

분포 $p(x)$를 이용해 가중한 후 $x$에 대해 적분하는 것을 해당 분포로부터 추출한 데이터들에 대한 유한 합산을 이용해 근사하였습니다. 해당 값들과 해당 값들의 합들을 시각화하면 다음과 같습니다.

![image](/assets/images/ml/Figure3.6.png){: width="400"}{: .align-center} 
Figure 3.6 편향, 분산, 편향과 분산의 합
{: style="text-align: center; font-size:0.7em;"}

작은 $\lambda$를 사용하면 각각의 노이즈들에 따라 세밀하게 조절되어 분산이 커지며, 큰 $\lambda$를 사용하면 가중 매개변수가 0에 가까워져 편향은 커집니다. 

이러한 편향 분산 분해는 모델 복잡도의 문제에 대해 흥미로운 통찰을 제공하지만 실용성은 적습니다. 여러 데이터 집합들의 모임에 대한 평균을 바탕으로 했지만 실제로는 단 하나의 데이터 관측 집합이 주어지는 경우가 대부분입니다. 만약 훈련 집합이 여러 개가 있다면 하나로 합쳐서 훈련에 사용하는 것이 더 효과적이며, 과적합을 줄이는 데도 더 좋습니다.

# 3.3 베이지안 선형 회귀

앞서 기저 함수의 숫자로 결정되는 모델의 복잡도는 데이터 집합의 크기에 따라 조절되어야 한다는 것을 살펴보았습니다. 정규화항을 추가하여 계숫값에 따라 모델의 복잡도를 조절할 수도 있습니다. 특정 문제에 대해 적당한 모델 복잡도를 결정하는 것은 중요한 문제입니다. 단지 가능도 함수만을 최대화하는 방식을 사용하면 언제나 과적합에 해당하는 아주 복잡한 모델을 선택하게 됩니다. 베이지안 방법론을 바탕으로 선형 회귀를 시행하면 과적합 문제를 피하면서도 훈련 데이터만 가지고 모델의 복잡도를 결정할 수 있습니다.

## 매개변수 분포

모델 매개변수 $\mathbf{w}$에 대한 사전 확률 분포를 도입해보겠습니다. 노이즈 정밀도 매개변수 $\beta$가 알려져 있는 상수라고 가정하겠습니다. 앞서 가능도 함수 $p(\mathbf{t}\vert\mathbf{w})$는 $\mathbf{w}$의 이차 함수의 지수 함수로 정의됨을 알아보았습니다. 따라서 이에 해당하는 켤레 사전 분포는 다음과 같습니다.

$$p(\mathbf{w})=\mathcal{N}(\mathbf{w}\vert\mathbf{m}_0,\mathbf{S}_0)$$

여기서 $\mathbf{m}_0$는 평균, $\mathbf{S}_0$은 공분산입니다. 사후 분포는 다음과 같습니다.

$$p(\mathbf{w}\vert\mathbf{t}) = \mathcal{N}(\mathbf{w}\vert\mathbf{m}_N, \mathbf{S}_N)$$

여기서 다음과 같습니다.

$$\begin{aligned}\mathbf{m}_N &= \mathbf{S}_N(\mathbf{S}_0^{-1}\mathbf{m}_0+\beta\boldsymbol\Phi^\text{T}\mathbf{t}) \\
\mathbf{S}^{-1}_N &= \mathbf{S}^{-1}_0+\beta\boldsymbol\Phi^\text{T}\boldsymbol\Phi
\end{aligned}$$

사후 분포가 가우시안 분포이기 때문에 최빈값과 평균값은 일치합니다. 따라서 최대 사후 가중 벡터는 $\mathbf{w}_\text{MAP}=\mathbf{m}_N$입니다. 처리 과정을 단순화하기 위해 0을 평균으로 가지고 단일 정밀도 매개변수 $\alpha$에 의해 결정되는 등방 가우시안 분포를 사용하도록 하겠습니다. 식은 다음과 같습니다.

$$p(\mathbf{w}\vert\alpha)=\mathcal{N}(\mathbf{w}\vert0,\alpha^{-1}\mathbf{I})$$

이에 해당하는 $\mathbf{w}$에 대한 사후 분포는 다음과 같습니다.

$$\begin{aligned}
\mathbf{m}_N&=\beta\mathbf{S}_N\boldsymbol\Phi^\text{T}\mathbf{t}\\
\mathbf{S}_N^{-1}&=\alpha\mathbf{I}+\beta\boldsymbol\Phi^\text{T}\boldsymbol\Phi
\end{aligned}$$

로그 사후 분포는 로그 가능도와 로그 사전 분포의 합으로 나타낼 수 있습니다. 이를 $\mathbf{w}$에 대한 함수로 적으면 다음과 같습니다.

$$\ln p(\mathbf{w}\vert\mathbf{t})=-\frac\beta2\sum^N_{n=1}\{t_n-\mathbf{w}^\text{T}\boldsymbol\phi(\mathbf{x}_n)\}^2-\frac\alpha2\mathbf{w}^\text{T}\mathbf{w} + \text{const}$$

이 사후 분포를 $\mathbf{w}$에 대해 최대화하는 것은 제곱 정규화항을 포함한 제곱합 오류 함수를 극소화하는 것과 같습니다.

![image](/assets/images/ml/Figure3.7.png){: width="500"}{: .align-center}

Figure 3.7 $y(x,\mathbf{w}) = w_0 + w_1x$ 형태의 선형 모델에 대한 순차적 베이지안 학습 과정
{: style="text-align: center; font-size:0.7em;"}

## 예측 분포

## 등가 커널
