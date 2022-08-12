---
date: 2022-08-11
title: "[Paper Review] USAD: UnSupervised Anomaly Detection on Multivariate Time Series"
categories: 
  - Paper Review
tags: 
  - Time Series
  - Anomaly Detection
  - Unsupervised Learning
toc: true  
toc_sticky: true 
---
# Paper contents

Usad: Unsupervised anomaly detection on multivariate time series

Audibert, J., Michiardi, P., Guyard, F., Marti, S., & Zuluaga, M. A.

Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (2020)

https://dl.acm.org/doi/abs/10.1145/3394486.3403392

## 0. Abstract

센서 등의 기술의 발달으로 데이터의 크기 및 복잡성이 늘어나면서 기존의 전통적인 방법은 정상 및 이상을 식별하는 데에 한계가 있는 상황입니다. 이러한 문제를 극복하기 위해 오토인코더를 활용한 빠르고 안정적인 방법인 **UnSupervised Anomaly Detection for multivariate time series (USAD) ** 를 제안합니다. 오토인코더를 활용하여 비지도 학습 방법으로 학습이 가능합니다. 또한 적대적 학습 방법을 이용해 이상을 빠르게 고립시키면서도 빠른 훈련이 가능합니다. 여러 공개된 데이터셋을 활용하여 실험을 진행하여 제안된 모델의 강건함과 속도, 그리고 높은 이상 탐지 성능을 보였습니다. 

## 1. Introduction

다변량 시계열 이상 탐지는 최근 활발한 연구가 진행되고 있습니다. 기존에 사용되었던 K-means 혹인 OC-SVM와 같은 거리 기반 방법론은 최근의 복잡한 데이터는 적합하지 않습니다. 최근에는 그보다는 딥러닝 기반의 비지도 이상탐지 기법들이 주목을 받고 있습니다. 특히 RNN 계열의 모델을 사용하는 경우가 대중적입니다. 하지만 RNN는 계산적으로 비효율적이며 학습에 오랜 시간이 소요된다고 알려져 있습니다. 이는 곧 많은 전력 소모로 이어집니다. 이외에도 최근에는 생성적 적대 신경망 *(GAN)*을 바탕으로 한 연구도 진행되고 있습니다. 하지만 GAN은 mode-collapse 혹은 non-convergence와 같은 문제들로 인해 학습이 어렵습니다. 이러한 불안정성은 실제 적용에 큰 장애물이 됩니다. 
이 논문에서는, 오토인코더 구조를 바탕으로 한 **UnSupervised Anomaly Detection for multivariate time series (USAD)**를 제안합니다. 인코더-디코더 구조의 적대적 학습을 통해 이상 징후를 포함한 재구성 오차를 증폭하는 방법을 학습하며, GAN에 비해 안정적입니다.  이 논문의 기여는 다음과 같습니다.

* 적대적 훈련 프레임워크에 인코더-디코더 구조를 제안하여 오토인코더와 적대적 학습 각각의 단점을 극복하면서 장점을 활용했습니다.
* 공개된 데이터 셋을 활용해 경험적인 연구를 진행하여 제안된 방법론의 강건함, 훈련 속도 및 성능을 입증했습니다.

## 2. Method

### 2.1 Problem Formulation

단변량 시계열은 데이터 포인트들의 연속이며, 각각은 특정 시점 $t$로부터 얻어진 값입니다. 단변량 시계열은 각각의 시점마다 하나의 변수를 가지고 있으며, 다변량 시계열은 두개 이상의 값을 가지고 있습니다. 

$${\tau} = \{\mathbf{x}_1, ..., \mathbf{x}_T\}, \mathbf{x} \in \mathbb{R}^m$$

비지도 학습 기반 이상탐지란, 주어진 $\tau$가 정상 데이터만으로 구성되어 있다고 가정할 때, $\tau$와 알려지지 않은 관측치 $\hat{\mathbf{x}}_t, t>T$가 충분한 차이를 보일 때 이를 식별하는 것입니다. 이때 $\hat{\mathbf{x}}_t$와 $\tau$가 다른 정도는 이상치 점수를 통해 측정되며, 이상치 점수가 임계치를 넘을 경우에 이상으로 판단합니다.  주어진 시점 $t$에서의 길이 $K$의 윈도우 $W_t$는 아래와 같이 나타냅니다.

$$W_t = \{\mathbf{x}_{t-K+1}, ..., \mathbf{x}_{t-1}, \mathbf{x}_t\}$$

원래의 시계열 $\tau$를 윈도우들의 연속 $\mathcal{W} = \{W_1, ..., W_T\}$로 나타낼 수 있으며 모델의 입력으로 사용됩니다. 이진 변수 $y \in \{0,1\}$이 주어졌을 때, 이상 탐지의 목적은 알려지지 않은 윈도우 $\hat{W_t}, t>T$에 대해서 라벨 $y_t$를 해당 윈도우의 이상치 점수에 기반해 맞추는 것입니다. 이후에는 $W$는 훈련에 사용된 윈도우, $\hat{W}$는 알려지지 않은 윈도우를 의미합니다.

### 2.2 Unsupervised Anomaly Detection

#### AutoEncoder (AE)

 오토인코더는 인코더 $E$와 디코어 $D$로 이루어진 비지도 인공 신경망입니다. 인코더 부분은 입력 $X$를 받아 잠재 변수 $Z$에 맵핑 시킵니다. 디코더는 잠재 변수 $Z$를 다시 입력 공간에서 $R$로 재구성합니다. 그 후 입력 벡터 $X$와 재구성 된 $R$의 차이를 재구성 오차라고 부릅니다. 이때 $\text{AE}(X) = D(Z), Z=E(X)$ 이며 $\lVert \cdot \rVert_2$는 L2 정규화를 의미합니다. 정리하면 다음과 같습니다.

 $$\mathcal{L}_\text{AE} = \lVert X-\text{AE}(X)\rVert_2$$

**오토 인코더** 기반의 이상 탐지는 재구성 오차를 이상치 점수로 사용합니다. 점수가 높은 시점은 이상치로 간주됩니다. 훈련 단계에서는 오직 정상 데이터만 사용됩니다. 추론 단계에서는 AE는 정상 데이터를 잘 재구성하지만, 이상 데이터에 대해서는 학습한 적이 없기 때문에 재구성을 잘 하지 못 합니다. 하지만 이상치가 너무 작아서 정상 데이터와 상대적으로 비슷하다면 이상은 발견되지 않을 것입니다. 이는 AE가 입력된 데이터를 최대한 잘 재구성하는 것에 초점을 맞춰 학습되기 때문에 발생하는 문제입니다.

#### Generative Adversarial Network (GAN)

**GAN**은 두 개의 네트워크가 서로 동시에 학습되면서 적대적으로 min-max 게임을 수행하는 비지도 기반 인공 신경망입니다. 하나의 네트워크에서는 생성자 $G$가 최대한 현실적인 데이터를 생성하게 되며, 다른 네트워크에서는 판별자 $D$가 데이터가 $G$가 생성한 데이터인지 혹은 실제 데이터인지를 구별합니다. $G$의 훈련 목표는 $D$가 잘 구분하지 못 할 정도로 실제 데이터와 유사한 데이터를 생성하는 것이고, $D$의 훈련 목표는 두 경우를 잘 구분하는 것입니다.

AE와 비슷하게 GAN 역시 정상 데이터를 훈련에 사용합니다. 훈련 후에는 $D$가 이상 탐지기로 사용됩니다. 입력된 데이터가 학습된 데이터의 분포와 다르다면 $D$는 이를 $G$가 생성한 데이터라고 판단하여 가짜로 분류합니다. 하지만 GAN의 학습은 mode-collapse 및 non-convergence 문제로 매우 어려우며, 이러한 문제는 보통 $G$와 $D$의 불균형으로 발생합니다. 

#### UnSupervised Anomaly Detection (USAD)

이 논문에서는 AE의 구조를 두 단계의 적대적 훈련 프레임워크에 활용한 USAD를 제안합니다. AE가 이상이 포함되지 않은 데이터가 입력 되었을때 이를 식별하도록 하여 AE의 한계를 극복할 수 있도록 돕습니다. 또 AE 구조가 적대적 훈련 단계에서 더 안정적으로 학습이 되도록 하며,  GAN에서 학습 단계에서 발생하는 문제점들을 극복하도록 돕습니다.

![image](https://user-images.githubusercontent.com/35906602/184304386-658a1c2a-fb82-477d-996d-703164fe4889.png){: width="700"}{: .align-center} 

Figure 1. Proposed architecture illustrating the information flow at training (left) and detection stage (right). 
{: style="text-align: center; font-size:0.7em;"}

USAD는 크게 인코더 네트워크 $E$와 2개의 디코더 네트워크 $D_1$, $D_2$로 구성됩니다.  이 3개의 요소들은 두 개의 오토인코더 $\text{AE}_1$과 $\text{AE}_2$로 구성되어 있는 구조 안에서 연결되어 있으며, 두 오토인코더는 같은 인코더를 공유합니다.

$$\text{AE}_1(W) = D_1(E(W)), \text{AE}_2(W) = D_2(E(W))$$

위 식의 구조는 두 단계를 통해 학습됩니다. 우선 두 오토인코더들은 윈도우 $W$의 정상 입력들을 재구성하도록 훈련됩니다. 그 후 두 오토인코더들은 서로 적대적으로 학습되어 $AE_1$은 $AE_2$를 속이려고 하고, $AE_2$는 데이터가 실제 데이터인지 혹은 재구성 된 데이터인지 판별합니다. 

**Phase 1 : Autoencoder training.** 우선 각각의 AE에서 입력을 잘 재구성하도록 훈련시킵니다. 입력 데이터 $W$는 인코더 $E$를 통해 잠재 공간 $Z$로 압축되고, 각각의 디코더에서 재구성됩니다. 식으로 나타내면 다음과 같습니다.

$$
\begin{aligned}
\mathcal{L}_{\text{AE}_1} = \lVert W-\text{AE}_1(W)\rVert_2 \\
\mathcal{L}_{\text{AE}_2} = \lVert W-\text{AE}_2(W)\rVert_2
\end{aligned}
$$ 

**Phase 2 : Adversarial training.** 그 후 $\text{AE}_2$는 실제 데이터와 $\text{AE}_1$로부터 만들어진 데이터를 구분하도록 학습되며, $\text{AE}_1$은 $\text{AE}_2$를 속이도록 학습됩니다. $\text{AE}_1$로부터 만들어진 데이터는 $E$를 통해 다시 $Z$로 압축되고 $\text{AE}_2$를 통해 재구성됩니다. $\text{AE}_1$의 목적은 $W$와 $\text{AE}_2$의 출력의 차이를 최소화 하는 것입니다. $\text{AE}_2$의 목적은 이 차이를 최대화 하는 것입니다. 목적 함수를 식으로 나타내면 다음과 같습니다.

$$\min_{\text{AE}_1}\max_{\text{AE}_2}\lVert W-\text{AE}_2(\text{AE}_1(W))\rVert_2$$ 

따라서 정리하면 다음과 같습니다.

$$
\begin{aligned}
\mathcal{L}_{\text{AE}_1} = +\lVert W-\text{AE}_2(\text{AE}_1(W))\rVert_2 \\
\mathcal{L}_{\text{AE}_2} = -\lVert W-\text{AE}_2(\text{AE}_1(W))\rVert_2
\end{aligned}
$$ 

**Two-phase training.** 이 구조에서 오토인코더는 두 개의 목적을 가집니다. $\text{AE}_1$은 첫번째 단계에서는 $W$의 재구성 오차를 최소화 해야 하고, 두번째 단계에서는 $W$와 $\text{AE}_2$의 출력 사이의 차이를 최소화 해야 합니다. $\text{AE}_2$는 첫번째 단계에서는 $W$의 재구성 오차를 최소화 시키면서 두번째 단계에서는 입력 데이터가 $\text{AE}_1$로부터 재구성 된 출력의 재구성 오차를 최대화 시켜야 합니다. 따라서 이를 식으로 나타내면 다음과 같습니다. 이때 $n$은 훈련 epoch를 의미합니다.

$$
\begin{aligned}
\mathcal{L}_{\text{AE}_1} = \frac{1}n+\lVert W-\text{AE}_1(W)\rVert_2 + \left(1 - \frac1n\right)\lVert W-\text{AE}_2(\text{AE}_1(W))\rVert_2\\
\mathcal{L}_{\text{AE}_2} = \frac{1}n+\lVert W-\text{AE}_2(W)\rVert_2 - \left(1 - \frac1n\right)\lVert W-\text{AE}_2(\text{AE}_1(W))\rVert_2
\end{aligned}
$$ 

알고리즘으로 나타내면 다음과 같습니다.

![image](https://user-images.githubusercontent.com/35906602/184308829-0d24db28-475e-4451-ae87-af4f1b16a862.png){: width="400"}{: .align-center} 

여기서 $\text{AE}_2$는 GAN의 판별자 $D$와는 다른 방식으로 작동합니다. 입력이 실제 값이냐 혹은 재구성된 값이냐에 따라서 다른 방식으로 손실이 계산됩니다.

**Inferences.** 추론 단계에서는 이상치 점수는 다음과 같이 정의됩니다.

$$\mathbf{A}(\hat{W}) = \alpha\lVert\hat{W}-\text{AE}_1(\hat{W})\rVert_2 + \beta\lVert\hat{W}-\text{AE}_2(\text{AE}_1(\hat{W}))\rVert_2$$

여기서 $\alpha + \beta = 1$ 이며 false-positive와 true-positive의 trade-off를 조정하는 파라미터로 사용됩니다. 만약 $\alpha$를 $\beta$보다 크게 설정한다면, true positive의 수를 줄이면서 동시에 false positive의 수를 줄이게 됩니다. 이렇게 조정 가능한 파라미터는 산업에서 중요한 요소입니다. 알고리즘으로 나타내면 다음과 같습니다.

![image](https://user-images.githubusercontent.com/35906602/184309746-1ee3849e-d8c2-4010-a3bb-8173633d267f.png){: width="400"}{: .align-center} 

## 3. EXPERIMENTS AND RESULTS

실험 결과는 다음과 같습니다. 이때 With는 *point-adjust*를 의미하는데, 비교 대상 모델과의 직집적인 비교를 위해 이상이 발생할 경우 연속적으로 발생할 가능성이 높다고 가정하고 한 시점에서만 이상이라고 판단되어도 해당 시점이 포함된 전체 부분을 이상이라고 간주한 결과를 의미합니다. 실험에 대한 더 자세한 설명은 논문을 참조해주세요.

![image](https://user-images.githubusercontent.com/35906602/184313516-4263b3f7-f97e-49f6-ab38-6698e365754e.png){: width="700"}{: .align-center} 

![image](https://user-images.githubusercontent.com/35906602/184313576-b09c875d-b02c-4704-ad6f-ca82d39379b4.png){: width="400"}{: .align-center} 

![image](https://user-images.githubusercontent.com/35906602/184313642-09cc9b33-f5b2-4ad7-9491-8ccbcbf3fca1.png){: width="400"}{: .align-center} 

![image](https://user-images.githubusercontent.com/35906602/184313700-38d17c7b-58d6-4ef3-8a6b-bda4ec8cbbdc.png){: width="400"}{: .align-center} 

![image](https://user-images.githubusercontent.com/35906602/184313768-27dbc47b-6b9b-42d7-bc59-edaf61cde80b.png){: width="700"}{: .align-center} 

![image](https://user-images.githubusercontent.com/35906602/184313816-3e233bc1-35a6-4ff6-9aea-344c2c6de7d7.png){: width="400"}{: .align-center} 

## 4. Conclusion

이 논문에서는 오토인코더에 적대적 학습 방법을 결합한 **USAD**가 제안 되었습니다. 오토인코더와 적대적 학습 방법 각각의 장점을 결합하였으며, 공개 된 벤치마크 데이터셋에서 뛰어난 이상탐지 성능을 보였습니다. 이에 더해서 빠른 학습 시간, 그리고 파라미터에 강건하다는 장점도 보였습니다. 