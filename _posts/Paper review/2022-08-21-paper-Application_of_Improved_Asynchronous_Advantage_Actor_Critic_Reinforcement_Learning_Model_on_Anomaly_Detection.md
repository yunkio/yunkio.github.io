---
date: 2022-08-21
title: "[Paper Review] Application of Improved Asynchronous Advantage Actor Critic Reinforcement Learning Model on Anomaly Detection"
categories: 
  - Paper Review
tags: 
  - Reinforcement Learning
  - Anomaly Detection
toc: true  
toc_sticky: true 
---
# Paper contents

Application of Improved Asynchronous Advantage Actor Critic Reinforcement Learning Model on Anomaly Detection

Zhou, K., Wang, W., Hu, T., & Deng, K.

Entropy (2021)

https://www.mdpi.com/1099-4300/23/3/274

## 0. Abstract

이상 탐지 연구는 전통적으로 수학적 및 통계적 방법론을 통해 수행되어 왔습니다. 최근 강화학습은 AlphaGo 등 매우 여러 분야를 걸쳐 뛰어난 성과를 보였습니다. 하지만 강화학습을 이상 탐지에 적용한 연구는 매우 드뭅니다. 이 논문은 *adaptable asynchronous advantage actor-critic* 이라는 강화학습 모델을 이상 탐지 분야에 적용하는 것을 목적으로 합니다. 기존의 전통적인 머신러닝 기법 및 적대적 생성 모델을 활용한 방법과 성능 비교가 이루어졌습니다. 제안된 모델은 시퀀스, 이미지 두 가지 유형의 이상치에 대한 *attention* 메커니즘 및 *CNN*을 각각 제안했습니다. 실험 결과 본 모델의 유용성을 검증하였습니다. 기존의 *state-of-the-art* 모델에 비해 뛰어나거나 적어도 경쟁 가능한 정도의 성능을 보였습니다.

## 1. Introduction

이상치는 데이터에서 예상되지 않는 패턴을 의미합니다. 이상치의 유형은 *point*, *contextual*, *collective*가 있으며, 각각은 단일 데이터, 컨텍스트, 혹은 집합을 의미합니다. 이상 탐지는 데이터의 다수를 차지하는 유형과 큰 차이를 보이는 관측치, 이벤트, 아이템 등의 식별을 의미합니다. 

### 1.1 Traditional Methods for Anomaly Detection
기존의 방법은 수학적 및 통계적 도구를 활용하여 기존 데이터와의 거리 등을 활용해 이상치 점수를 계산합니다. *SVM*, *IF*, *LOF* 등의 방법이 여기에 속합니다. 이러한 알고리즘들은 효과적이긴 하지만 데이터의 표현에 대해 과도하게 단순회된 가정을 사용하거나 알고리즘의 확장성이 매우 좋지 않다는 문제가 있습니다. 

이상 탐지는 데이터의 문제로 매우 어렵습니다. 라벨링 문제가 존재하며, 데이터는 갈수록 규모가 커지기 때문에 점점 더 어려워지며, 이상 혹은 정상에 대한 라벨은 존재하지 않는 경우가 대부분입니다. 게다가 다른 유형의 데이터는 서로 다른 유형의 이상 패턴을 가집니다. 

지도 학습, 비지도 학습, 반지도 학습은 각각 어느정도 성과를 거두었지만 한계를 보이고 있습니다. 라벨이 전혀 존재하지 않는 **비지도 학습**은 대부분의 데이터가 정상이며 작은 군집을 이상이라고 가정합니다. 하지만 고차원의 데이터는 일반적으로 희소한 분포를 가지고 있기 때문에 무수히 많은 작은 군집이 존재합니다. **반지도 학습**은 정상에 대한 라벨을 지니고 있는 상황을 가정하기 때문에 정상 데이터에 대한 모델링이 가능하지만, 학습되지 않았으나 정상인 데이터로 인해 오분류가 많이 발생합니다. **지도 학습**의 경우 라벨이 필요하다는 현실적인 한계가 있으며, 학습되지 않았거나 새로운 유형의 이상치에 대해서는 탐지가 불가능합니다.

### 1.2 Deep Reinforcement Learning for Anomaly Detection

이 논문에서는 **A3C** 방법을 기반으로 한 **adaptable deep neural network** 모델을 제안합니다. **A3C**는 정책 및 상태 가치 함수를 동시에 학습하고 가치 함수는 에이전트로 하여금 최대한 많은 보상을 얻도록 합니다. 시계열 탐지를 위해서는 *attention* 기반의 *RNN* 이 *actor*의 가치 함수 추정으로 제안되었으며, 비디오나 이미지에는 *attention* 기반 *RNN* 대신 *CNN*이 사용 되었습니다. 또한 새로운 관측치가 정상 패턴을 따르는지를 감지하기 위해 *DNN*이 보상 함수를 대신하게 됩니다. 

강화학습은 다음과 같은 측면에서 지도 학습 혹은 비지도 학습과 차이점을 보입니다. 

* 함수가 일반적으로 *i.i.d를 가정하지 않습니다*.
* 에이전트는 그들이 받게 되는 데이터에 영향을 끼치게 됩니다.
* 피드백은 드문 빈도로 이루어지며 항상 딜레이 되어 있습니다.

모델링 과정과 강화학습을 위한 훈련 셋 및 테스트 셋을 구성하는 과정은 지도 학습과 매우 다릅니다. 데이터는 정상성을 가정하지 않으며 에이전트가 행동을 어떻게 하느냐에 의존하게 됩니다. 에이전트는 기본적으로 훈련의 복잡성을 더 강화할 수 있는 방향으로 어떤 데이터를 볼 것이냐를 정합니다.

## 2. Preliminaries

### 2.1 Reinforcement Learning

강화학습을 이상 탐지에 적용할 때 가장 중요한 요소는 문제 정의입니다. 환경, 상태, 행동, 그리고 보상을 어떻게 정의할 것이냐에 따라 성과가 매우 달라지게 됩니다. 강화학습 에이전트는 각 시점에서 입력을 상태 정보로 맵핑하게 됩니다.  그 후 행동을 실행하게 되며 환경으로부터 양의 보상 혹은 음의 보상으로 피드백을 받게 됩니다. 에이전트는 우선 상태를 바탕으로 행동을 취하고, 보상을 받게 되며, 환경 안에서 상태의 변화를 관측하고 잠재 보상을 최대화 할 수 있도록 정책을 업데이트 합니다. 최적의 정책의 목표는 시간의 흐름에 따라 얻을 수 있는 보상을 최대화 하는 것입니다.

![image](https://user-images.githubusercontent.com/35906602/185802205-289e5380-cf37-4c6a-a631-a4bb18212f10.png){: width="600"}{: .align-center} 

Figure 1. Agent environment interaction (left), action and reward sequences (right).
{: style="text-align: center; font-size:0.7em;"}

에이전트는 가치 기반, 정책 기반, 혹은 가치 함수 및 정책 양 쪽을 가지는 actor-critic으로 분류 될 수 있습니다. 정책은 에이전트의 행동을 결정하며, 상태에서 액션으로 향하는 맵핑 함수로 볼 수 있습니다. 정책은 $p(a\vert s)$로 모델링 되며 주어진 상태 $s$에서 각 행동 $a$에 대한 확률입니다. 여기서 정책은 입력으로 $s$를 받고 출력으로 $a$를 내보내는 신경망으로 구성 할 수 있습니다. MDP의 상태-가치 함수는 다음과 같습니다. 상태 $s$로 부터 정책 $\pi$를 따라서 얻게 되는 기대되는 리턴이며, $G_t$는 상태 $s$에서부터 에피소드의 끝까지 총 얻게 되는 보상의 총량이며, $\gamma$는 할인률을 의미합니다.

$$
\begin{aligned}
v_\pi(s) & = E_\pi[G_t\vert S_t=s]=E_\pi[R_{t+1}+\gamma v_\pi(S_{t+1})\vert S_t = s] \\
&= \sum_a\pi(a\vert s)\sum_{s',r}p(s',r\vert s,a)[r + \gamma v_\pi(s')] 
\end{aligned}
$$

고정된 정책에서의 MDP는 곧 MRP가 되며 행렬 연산을 통해 풀 수 있습니다. 고정되었다는 의미는 곧 보상이 (상태, 액션) 쌍의 함수임을 의미하며 여기서 보상은 확률 분포가 아닙니다. 가치 함수는 다음과 같이 표현됩니다. MDP $M = (S,A,P,R,\gamma)$와 정책 $\pi$가 주어졌을 대, 연속된 상태 $s_1, s_2, ...$는 마르코프 과정 $(S,p^\pi)$이며 상태 보상의 시퀀스 $S_1, R_2, S_2, ...$는 MRP $(S, p^\pi, R^\pi, \gamma)$ 입니다.

$$\begin{aligned}
v_\pi(s) &= \sum_{a\in A}{\pi(a\vert s)\left\{r(s,a) + \gamma\sum_{s' \in S}{T(s'\vert s,a)v_\pi(s')}\right\}} \\
&= r^\pi_s+\gamma\sum_{s' \in S}T^\pi_{s' s}v_\pi(s')
\end{aligned}$$

여기서 $r^s_\pi$는 상태 $s$에서의 정책 $\pi$를 따를 때 보상을 의미하며, $T^\pi_{s's}$는 상태 $s$에서 $s'$로 갈 때의 정책 $\pi$에서의 전이 함수를 의미합니다. 가치 함수는 $\vec{v_\pi} = r^\pi + \gamma T^\pi\vec{v_\pi}$로 표현 가능합니다. 이때 $\vec{v_\pi}$는 벡터이며 $T^\pi$는 전이 함수를 의미합니다. 반복적 방법이 행렬을 푸는데 사용됩니다. 상태-행동 가치 함수인 $q_\pi(s,a)$는 상태 $s$, 취해진 행동 $a$, 정책 $\pi$에 따르는 예상되는 리턴을 의미합니다.

... WIP

## 3. Materials and Methods

### 3.1 Definitions

이상 탐지기 $d$는 $t$ 시점에서 하나의 상태 $\Upsilon_t$를 관측하며 이 탐지는 POMDP *부분 관측 마르코프 결정 과정*으로 모델링 가능합니다. 이상 탐지기 $d$는 강화학습 환경의 에이전트에 해당하며, 주어진 정책 $\pi$에서 $d$가 주어진 상태 $s$에서 취하는 행동 $a$는 확률 분포 $\pi:=p(A\vert S)$로 정의됩니다. 이때 $S$와 $A$는 각각 상태와 행동의 집합을 의미합니다. 정책은 파라미터 $\theta$를 포함한 신경망으로 모델링되며, 입력은 상태가 되고 출력은 행동이 됩니다. 이상 탐지이기 때문에 여기서 $1$은 이상, $0$은 정상을 의미하게 됩니다. 행동 공간은 $d$가 각각의 행동을 취할 확률로 표현되며,  이로부터 보상을 얻게 됩니다. 목표는 제한적인 시간에서 최대의 보상을 얻는 것입니다. 각 시점 $t$에서 선택된 행동 $\hat{a_t}$와 정답 행동 $a_t^*$를 비교하여 선택된 행동과 정답 행동이 같다면 보상으로 $1$, 틀렸다면 $0$이 주어집니다. 최종적으로 가치 함수 $V_\pi$의 성능은 다음과 같이 정의됩니다.

$$V_\pi = \sum_{s \in S}w^\pi(s)\sum_{a\in A}R(s,a)\pi(s,a)$$

여기서 $w^\pi(s)$는 상태 $s$에서의 시스템의 확률이며, $R(s,a)$는 상태 $s$와 행동 $a$에서 출발한 보상의 총 합입니다. 성능은 정책 $\pi$를 따랐을 때 예상되는 보상의 총 합입니다. $V_\pi$는 신경망으로 구현되며, 현재의 상태 및 행동이 입력으로 들어가며 네트워크는 전체 에피소드 관점에서의 현 상태의 가치를 출력으로 뱉습니다. 최적의 성능을 가지는 $\pi^*$는 $\pi^* = \argmax_\pi V_\pi$를 만족합니다. $\pi$는 경험을 통해 계속해서 학습되며 더 많은 보상을 얻도록 향상됩니다.

시계열 이상 탐지에서는 슬라이딩 윈도우 방법$(t_i, t_{i+1}, ..., t_{i+n})$이 사용되었습니다. 탐지기는 정책을 따라 행동하며 매 윈도우마다 보상을 얻습니다. 슬라이딩 윈도우가 시계열 전체를 훑고 지나간 후 보상이 더해집니다. 

### 3.2 Anomaly Detector Architecture

![image](https://user-images.githubusercontent.com/35906602/185804184-cc686be0-b95f-4837-b9f8-e67fb6dfbff2.png){: width="600"}{: .align-center} 

Figure 2. Anomaly detection architecture based on A3C.
{: style="text-align: center; font-size:0.7em;"}

A2C는 두 독립적인 신경망을 가지고 있으며 각각의 뉴런들은 서로 공유되지 않고 $\theta$에 의해 파라미터화 되어 있습니다. A2C는 가치 및 정책 기반의 알고리즘을 *critic*이 *advantage* 값을 계산하여 행동의 가치 뿐만 아니라 어떻게 향상될 수 있는지까지 제공하도록 하여 발전시켰습니다. **A3C**는 이에 더해서 비동기적인 특성을 더하고 각각의 *worker*들이 서로 상호작용하도록 하여 더 빠르고 유연하게 했습니다. A3C는 전역 네트워크 및 복수의 *worker*를 포함하고 있습니다. 각각의 네트워크는 *critic* 및 *actor*의 출력을 갖도록 모델링 되어 있습니다. 첫번째 출력은 주어진 상태 $V(s)$의 기대되는 보상을 스칼라 값으로 나타내며, 두번째 출력은 가능한 모든 행동 $(s,a)$의 확률 분포를 벡터로 가집니다.

현재 *worker*의 구조에는 주로 기본적인 신경망 구조가 사용됩니다. 하지만 이 논문에서는 목적으로 하는 문제에 따라 다른 구조를 사용할 것을 제안합니다. *(실제로는 A3C가 제안된 논문에서 이미 목적에 따라 RNN, CNN 등의 다양한 구조를 사용하고 있습니다.)* 가령 이미지에 대한 이상 탐지에서는 *CNN*을 기반으로 한 구조가 사용되며, 시계열에 대한 이상 탐지에는 *attention*을 기반으로 한 *RNN* 구조가 사용됩니다. 과정은 다음과 같습니다

1. 각각의 *worker*가 전역 네트워크로 초기화 됩니다.
2. 각각의 환경과 상호작용 합니다.
3. 가치 및 정책 손실을 추정합니다.
4. 손실을 통해 경사를 계산하고, 모든 *worker*들의 경사를 평균내어 전역 네트워크를 업데이트 합니다.
5. 다시 첫번째 단계로 돌아갑니다.

![image](https://user-images.githubusercontent.com/35906602/185804561-6e440398-5267-4462-8558-e13ba2a99783.png){: width="500"}{: .align-center} 

Figure 3. Proposed network for actor.
{: style="text-align: center; font-size:0.7em;"}

강화학습을 모델링 할 때는 문제 정의가 매우 중요합니다. 여기서는 목적으로 하는 문제에 따라 다양한 형태의 신경망 구조를 사용했습니다. A3C 모델을 바탕으로 한 이상 탐지 학습 알고리즘은 다음과 같습니다.

![image](https://user-images.githubusercontent.com/35906602/185805080-d839c89b-5d45-40be-b49d-ef31b5b34cea.png){: width="700"}{: .align-center} 

Figure 4. Training process for detector and evaluator.
{: style="text-align: center; font-size:0.7em;"}

![image](https://user-images.githubusercontent.com/35906602/185805104-80594524-1681-4bfd-bf6d-5616fa8a0127.png){: width="500"}{: .align-center} 

각 *worker*는 데이터로부터 배치 $s_{t_1}a_{t_1}s_{t_1+1},...,s_{t_i}a_{t_i}s_{t_i+1}$을 샘플링하며, 상태는 정책 네트워크 ($\theta'$는 *worker*, $\theta$는 전역 네트워크) 및 상태 가치 네트워크 ($\theta_{v'}$는 *worker*, $\theta_v$는 전역 네트워크)에 입력됩니다. 추정된 가치 함수 $R_t$와 실제 가치 함수 $V_t$의 차이를 의미하는 *advantage* 값 $A_t = R_t - V_t$는 정책 함수를 근사한 네트워크를 학습하는데 사용됩니다. 만약 $A_t$가 양의 값을 가진다면 행동 $a_t^n$의 확률을 늘리고, 음의 값을 가진다면 확률을 낮추는 방향으로 학습이 진행됩니다. 각각의 *worker*는 $\theta'$ 및 $\theta_{v'}$를 업데이트하며 이 파라미터들은 전역 네트워크로 전해집니다. 전역 파라미터는 특정 행동이 과대평가 되는 것을 막기 위해 각각의 *worker*들의 파라미터들을 평균내어 정해집니다. 정책의 엔트로피 $(H(\pi(a_j\vert s_j;\theta'))$는 탐험을 촉진하고 지역적 최적값에 수렴하는 것을 막기 위해 사용됩니다.

*neural sampler*는 훈련 샘플의 배치 $s_{t_1}a_{t_1}s_{t_1+1},...,s_{t_i}a_{t_i}s_{t_i+1}$를 제공하게 됩니다. 각 탐지기는 상태 $s_t$에서 행동을 실행하고 보상 $r_t$를 받으며 다음 상태 $s_{t+1}$로 가게 됩니다. 정책 손실 함수는 $A_t\log\pi(\hat{a_t})$의 형태를 띄며, 확률 분포에서의 선택된 행동 $\hat{a_t}$의 로그 확률에 $A_t$를 가중치로 곱합니다. *actor*와 *critic* 신경망은 동시에 학습되며 두 신경망의 손실 모두 최소화하는 방향으로 학습됩니다. 

![image](https://user-images.githubusercontent.com/35906602/185805134-91f8023c-5197-4720-96f0-26ad30b0964a.png){: width="600"}{: .align-center} 

Table 1. Comparaions among RL, GAN and the proposed models.
{: style="text-align: center; font-size:0.7em;"}

## 4. Result

벤치마크 데이터셋을 활용해 성능 평가를 진행했습니다. **AWID, NSL_KDD, Time Series Dataset** 등이 사용되었습니다. 결과는 다음과 같습니다.

### 4.1 AWID

![image](https://user-images.githubusercontent.com/35906602/185805407-f950396f-fdd9-4ac8-845e-99377ecf9055.png){: width="600"}{: .align-center} 

Table 2. Numbers for AWID reduced training and test dataset.
{: style="text-align: center; font-size:0.7em;"}

![image](https://user-images.githubusercontent.com/35906602/185805322-19d21563-ed0f-4aad-958d-93bdc86707ca.png){: width="600"}{: .align-center} 

Table 3. Test results comparisons among proposed method and others.
{: style="text-align: center; font-size:0.7em;"}

### 4.2 Time Series Anomaly Test

슬라이딩 윈도우 메커니즘이 사용되었습니다. 슬라이딩 윈도우의 사이즈는 25로 설정되었으며, reward는 TP일 경우 5, FP일 경우 -1, TN일 경우 1, FN일 경우 -5로 설정되었습니다. 

![image](https://user-images.githubusercontent.com/35906602/185805638-8d34b471-2394-4f2e-a829-a804efed0fe7.png){: width="400"}{: .align-center} 

Figure 5. Action-value function for actions of time series anomaly.
{: style="text-align: center; font-size:0.7em;"}

위 그림이 의미하는 바는 불분명합니다. 논문에는 별다른 설명이 없습니다.

![image](https://user-images.githubusercontent.com/35906602/185806145-a4e46b5e-22c9-4e19-be87-5d0b70a12a0d.png){: width="700"}{: .align-center} 

Figure 6. Time series anomaly test results without smooth. (b)smoothed error for the time series anomalies.
{: style="text-align: center; font-size:0.7em;"}

위 그림은 제안된 방법이 아닌 Q-learning에 RNN을 적용한 방법으로, 이상 탐지가 아니라 시계열 예측을 통해 실제 값과 예상된 값의 차이를 통해 이상탐지를 한 것 같습니다. F1-score가 약 0.69정도가 나왔다고 합니다.

![image](https://user-images.githubusercontent.com/35906602/185806329-1bf966ab-755d-4e30-a12a-a783935be99b.png){: width="400"}{: .align-center} 

Figure 7. Time series anomaly using the RNN and Q-learning of RL.
{: style="text-align: center; font-size:0.7em;"}

위 그림은 설명과는 다르게 실제로는 제안된 방법을 통한 결과를 시각화 한 것 같습니다. 제안된 방법은 F1 score가 약 0.84정도 나왔다고 합니다. 사실 이 부분에 있어서 본 논문에 그림과 설명이 안 맞는 부분이 많아서 해석이 다소 어렵습니다.

### 4.3 NSL-KDD Network Anomaly Test

![image](https://user-images.githubusercontent.com/35906602/185806521-845641ca-c54f-4415-9e06-6415cd1e0c74.png){: width="600"}{: .align-center} 

Table 4. Metrics comparisons among proposed model and others.
{: style="text-align: center; font-size:0.7em;"}

![image](https://user-images.githubusercontent.com/35906602/185806573-37e1609e-02e5-4db4-85eb-5f4bf28901e9.png){: width="700"}{: .align-center} 

Figure 8. (a) Test scores for different attacks. (b) Confusion matrix for five anomalies.
{: style="text-align: center; font-size:0.7em;"}

![image](https://user-images.githubusercontent.com/35906602/185806650-9adede9c-0a04-4ffc-aef0-0c6f08d377cf.png){: width="700"}{: .align-center} 

Figure 9. Rewards and losses comparisons among different models by epochs.
{: style="text-align: center; font-size:0.7em;"}

![image](https://user-images.githubusercontent.com/35906602/185806658-79aff87e-633b-497b-839c-929d82d5c21e.png){: width="700"}{: .align-center} 

Figure 10. Training time comparisons among different models by epochs.
{: style="text-align: center; font-size:0.7em;"}

## 5. Conclusion

좋은 논문은 아닌거 같습니다.
