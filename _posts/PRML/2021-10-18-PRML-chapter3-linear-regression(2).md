---
date: 2021-10-18
title: "[PRML] Chapter 3. Linear Regression (2) - 편향 분산 분해, 베이지안 선형 회귀"
categories: 
  - 패턴인식과 머신러닝
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

위 그림은 선형 기저 함수 모델의 베이지안 학습과 사후 분포의 순차적 업데이트 방식을 시각화 한 그림입니다. 매개변수 $\alpha$와 정밀도 매개변수 $\beta$가 주어진 상황에서 데이터 집합 크기가 커짐에 따라 베이지안 학습이 어떻게 되는지를 확인할 수 있습니다. 또, 현재의 사후 분포가 새로운 데이터가 관측된 후에 새로운 사전 분포가 됨을 알 수 있습니다. 분포의 흰색 십자가가 데이터를 만들 때 사용한 매개변수를 의미하며, 오른쪽 열의 파란색 동그라미는 관측된 데이터를 의미합니다. 이렇게 만들어진 샘플 회귀 함수는 오른쪽에 빨간색 직선으로 나타납니다.

## 예측 분포

실제 응용 사례에서는 $\mathbf{w}$값을 알아내는 것보다 새로운 $\mathbf{x}$에 대한 $t$값을 예측하는 것이 더 중요할 수 있습니다. 이를 위해 **예측 분포** *predictive distribution*을 고려할 필요가 있습니다.

$$p(t\vert\mathbf{t},\alpha,\beta) = \int p(t\vert\mathbf{w},\beta)p(\mathbf{w}\vert\mathbf{t},\alpha,\beta)d\mathbf{w}$$

여기서 $\mathbf{t}$는 표적값들의 벡터입니다. 위 식은 두 가우시안 분포의 콘볼루션을 포함하고 있으므로 예측 분포는 다음과 같은 형태를 지닙니다.

$$p(t\vert\mathbf{x},\mathbf{t},\alpha,\beta) = \mathcal{N}(t\vert\mathbf{m}^\text{T}_N\boldsymbol{\phi}(\mathbf{x}),\sigma^2_N(\mathbf{x}))$$

여기서 분산 $\sigma^2_N(\mathbf{x})$는 다음과 같습니다.

$$\sigma^2_N(\mathbf{x}) = \frac1\beta+\boldsymbol{\phi}(\mathbf{x})^\text{T}\mathbf{S}_N\boldsymbol{\phi}(\mathbf{x})$$

위 식의 첫 번째 항은 데이터의 노이즈를 표현하며, 두 번째 항은 매개변수 $\mathbf{w}$에 대한 불확실성을 표현합니다. 노이즈와 $\mathbf{w}$에 대한 분포를 처리하는 것은 독립적인 가우시안 분포이기 때문에 분산들을 합산할 수 있습니다. 추가적인 데이터가 관측된다면 사후 분포는 더 좁아지며 $\lim N \rightarrow \infty$일 경우 두 번째 항이 0이 되어 예측 분포의 분산은 데이터의 노이즈만을 포함하게 됩니다.

<figure class="half">
  <a href="/assets/images/ml/Figure3.8a.png">
  <img src="/assets/images/ml/Figure3.8a.png" width="200"></a>

  <a href="/assets/images/ml/Figure3.8b.png">
  <img src="/assets/images/ml/Figure3.8b.png" width="200"></a>
</figure>
<figure class="half">
  <a href="/assets/images/ml/Figure3.8c.png">
  <img src="/assets/images/ml/Figure3.8c.png" width="200"></a>

  <a href="/assets/images/ml/Figure3.8d.png">
  <img src="/assets/images/ml/Figure3.8d.png" width="200"></a>
</figure>

Figure 3.8 예측 분포의 예시
{: style="text-align: center; font-size:0.7em;"}

위 그림에서 녹색 곡선은 노이즈가 추가된 데이터가 만들어진 원본 함수이며, 데이터 집합의 크기가 각각 $N=1, 2, 4, 25$인 경우에 대한 함수 도식이 그려져 있습니다. 빨간색 곡선은 예측 분포들의 평균이며 빨간색 구간은 1 표준 편차만큼 표현한 것입니다. 예측값의 불확실성은 $x$에 종속적이며 데이터 포인트들의 주변에서 불확실성이 가장 작습니다. 

위 그림은 점에 대한 에측 분산만을 $x$에 대한 함수로 보여 주고 있으므로 서로 다른 $x$의 예측값들에 대한 공분산을 살펴보기 위해 $\mathbf{w}$에 대한 사전 분포로부터 샘플들을 추출한 후 그에 대한 함수 $y(x, \mathbf{w})$를 그려보겠습니다.

<figure class="half">
  <a href="/assets/images/ml/Figure3.9a.png">
  <img src="/assets/images/ml/Figure3.9a.png" width="200"></a>

  <a href="/assets/images/ml/Figure3.9b.png">
  <img src="/assets/images/ml/Figure3.9b.png" width="200"></a>
</figure>
<figure class="half">
  <a href="/assets/images/ml/Figure3.9c.png">
  <img src="/assets/images/ml/Figure3.9c.png" width="200"></a>

  <a href="/assets/images/ml/Figure3.9d.png">
  <img src="/assets/images/ml/Figure3.9d.png" width="200"></a>
</figure>

Figure 3.9 $\mathbf{w}$의 사후 분포들에서 추출한 샘플들을 사용한 $y(w,\mathbf{w})$의 도식
{: style="text-align: center; font-size:0.7em;"}

가우시안 같은 지역적인 기저 함수를 사용한다면 기저 함수의 중심으로부터 떨어진 구간에서는 위 식의 두 번째 항의 기여도가 0이 됩니다. 즉 기저 함수에 의해 포함되는 지역의 바깥에 대해서 예측할 경우 모델의 신뢰도가 높아진다는 겨로가가 나오게 됩니다. 이는 옳지 않은 결과입니다. **가우시안 과정** *Gaussian process* 라고 알려져 있는 기법을 활용하여 이 문제를 피할 수 있습니다.

## 등가 커널

선형 기저 함수 모델에서의 사후 해를 다른 방식으로 해석할 수 있습니다. 가우시안 과정을 포함한 커널 방법론에 대해 살펴보는 첫 단계입니다. 예측 평균을 다음과 같이 적을 수 있습니다.

$$y(\mathbf{x}, \mathbf{m}_N) = \mathbf{m}^\text{T}_N\boldsymbol\phi(\mathbf{x}) = \beta\boldsymbol\phi(\mathbf{x})^\text{T}\mathbf{S}_N\boldsymbol\Phi^\text{T}\mathbf{t} = \sum^N_{n=1}\beta\boldsymbol\phi(\mathbf{x})^\text{T}\mathbf{S}_N\boldsymbol\phi(\mathbf{x}_n)t_n$$

이 식은 다음과 같습니다.

$$\begin{aligned}
y(\mathbf{x},\mathbf{m}_N)&=\sum^N_{n=1}k(\mathbf{x},\mathbf{x}_n)t_n \\
k(\mathbf{x},\mathbf{x}') &= \beta\boldsymbol\phi(\mathbf{x})^\text{T}\mathbf{S}_N\boldsymbol\phi(\mathbf{x}')
\end{aligned}$$

위 식 $k(\mathbf{x},\mathbf{x}')$는 **평활 행렬** *smoother matrix*, **등가 커널** *equivalent kernel* 이라고 알려져 있습니다. 훈련 집합 표적값들의 선형 결합을 입력받아서 예측을 내는 이러한 회귀 함수는 **선형 평활기** *linear smoother* 라고 부릅니다. $\mathbf{S}_N$의 정의에 $\mathbf{x}_n$이 포함되어 있기 때문에 등가 커널은 입력값 $\mathbf{x}_n$에 종속적입니다. 

![image](/assets/images/ml/Figure3.10.png){: width="500"}{: .align-center}

Figure 3.10 가우시안 기저 함수에 대한 등가커널 $k(x, x')$
{: style="text-align: center; font-size:0.7em;"}

위 그림은 기저 함수가 가우시안인 경우의 등가 커널에 대한 그림입니다. 커널 함수 $k(x, x')$를 세 개의 서로 다른 $x$값들에 대해서 $x'$의 함수로 그렸습니다. 각각은 $x$ 주변에서 지역화되어 있습니다. $y(x, \mathbf{m}_N)$으로 주어지는 $x$에서의 예측 분포는 표적값들의 가중 조합을 통해 구해지게 되는데 $x$값에 근접할수록 더 높은 가중치, $x$값으로부터 멀리 떨어질수록 낮은 가중치를 지니게 됩니다. 이러한 지역화 성질은 비지역적인 다항 기저 함수와 시그모이드 기저 함수의 경우에도 적용됩니다.

$y(\mathbf{x})$와 $y(\mathbf{x'})$ 간의 공분산에 대해 고려해보면 등가 커널의 역할에 대한 더 깊은 의미를 생각해볼 수 있습니다.

$$\text{cov}[y(\mathbf{x}),y(\mathbf{x}')] = \beta^{-1}k(\mathbf{x},\mathbf{x}')$$

등가 커널의 형태로부터 서로 근처에 있는 포인트들의 예측 평균들은 상관성이 크며 서로 멀리 떨어져 있는 포인트들의 예측 평균은 상관성이 작다는 것을 알 수 있습니다. 

선형 회귀를 커널 함수로 표현하는 것을 바탕으로 다음과 같은 대체적인 회귀 방법론을 생각해볼 수 있습니다. 기저 함수를 직접 도입하여 사용하는 대신 지역화된 커널을 직접 정의하고 이를 바탕으로 새 입력 벡터 $\mathbf{x}$에 대한 예측값을 주어진 관측된 훈련 집합으로부터 구할 수 있습니다. 이를 바탕으로 유도되는 방법론이 **가우시안 과정** *Gaussian process* 입니다. 이에 대해서는 나중에 더 자세히 살펴보겠습니다.

등가 커널이 가중치들을 결정하며 이 가중치들을 바탕으로 훈련 집합의 타깃 변수들이 합쳐져서 새로운 $\mathbf{x}$에 대한 예측을 한다는 것을 살펴보았습니다. 이 가중치들을 모든 $\mathbf{x}$에 대해 합산하면 1이 됩니다.

$$\sum^N_{n=1}k(\mathbf{x},\mathbf{x}_n) = 1$$

커널 함수는 양의 값을 가질 수도, 음의 값을 가질 수도 있습니다. 
