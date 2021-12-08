---
date: 2021-11-11
title: "[Paper Review] Policy-based reinforcement learning for time series anomaly detection"
categories: 
  - Paper Review
tags: 
  - 머신러닝
  - 강화학습
  - Anomaly Detection
  - 논문 리뷰
toc: true  
toc_sticky: true 
---

# Paper contents

Policy-based reinforcement learning for time series anomaly detection

Mengran Yu, Shiliang Sun

Engineering Applications of Artificial Intelligence 95 (2020)

https://www.sciencedirect.com/science/article/pii/S0952197620302499

## 0. Abstract

시계열 이상 탐지는 IoT 기술의 발전 및 센서의 증가로 인한 스트리밍 데이터의 증가로 점점 더 중요해지고 있습니다. 기존의 방법은 도메인에 맞춰져 있거나, 혹은 현실의 데이터에는 적용하기 힘든 강한 가정이 필요했습니다. 강화학습은 두 한계를 극복할 수 있는 방법입니다. 이 논문에서는 **시계열 이상 탐지** 문제에 강화학습을 적용할 수 있는 **정책 기반 강화학습 프레임워크**를 제안합니다. **PTAD** *policy-based time series anomaly detector*는 제약 조건 없이 시계열 데이터와의 상호작용을 통해 학습됩니다. 실험 결과는 학습 데이터와 시험 데이터가 같은 원천에서 나온 경우에도, 아닌 경우에도 뛰어난 성능을 보이고 있음을 보여줍니다. 게다가 정밀도와 재현율 간의 균형도 잘 지킬 수 있습니다.


## 1. Introduction

점점 더 많은 센서들이 사용되면서 많은 양의 시계열 데이터가 수집되고 있습니다. 공간적, 시간적 맥락에 얽혀있을 수 있는 수 많은 복잡한 비정상 패턴이 존재하며 이러한 패턴들은 주기적 혹은 계절적일 수도 있으므로 비정상성을 발견하는 것은 매우 어렵습니다. 

시계열 이상 탐지는 많은 연구가 있어왔으며 대부분은 센서에 대한 것입니다. 센서는 고장에 취약하며, 고장난 센서는 여러가지 심각한 문제점을 야기합니다. 최근에는 지도학습 방법론뿐만 아니라 반지도학습이나 비지도학습 방법론도 연구되고 있으며, 일반적으로는 특정 데이터 혹은 특정한 분야의 응용을 위해 디자인 되었습니다. 일반적으로 적용할 수 있는 비정상 탐지를 설계하는 것은 더 어렵습니다. 일반적으로 적용할 수 있는 모델 중에 시계열 벤치마크 데이터에 좋은 성능을 보이는 모델은 찾기 힘듭니다. 대부분 데이터에 대한 분석 혹은 강한 가정이 필요하기 때문입니다.

강화학습에서 에이전트는 유용한 정보를 배웠을 때 긍정적인 피드백을 받게 되고, 쓸모없는 정보를 배웠을 때는 부정적인 피드백을 받습니다. 에이전트는 자동으로 아무런 가정이나 제약없이 환경과 상호작용하는 일반적인 프레임워크를 학습하게 됩니다. 이러한 특징은 비정상 감지 문제를 해결하는 데 중요한 특징입니다. 

하지만 이 분야에 대한 연구는 굉장히 부족합니다. 이미 **DQN** 알고리즘을 을 기반으로 한 값 기반의 강화학습을 적용한 연구가 있지만 간단한 지침일 뿐이였습니다. 실험 결과는 강화학습을 통한 비정상 행동의 탐지의 가능성을 보여주었습니다. 

이 논문에서는 정책 기반의 강화학습 방법이 가치 기반의 강화학습 방법과 비교하여 어떤지, 그리고 시계열 비정상 탐지 문제에 효과적인 해결책이 될 수 있는지를 연구합니다. 시계열 비정상 탐지 문제에 정책 기반의 강화학습 프레임 워크를 적용하고 일반적인 비정상 탐지기를 제안합니다. 실험 결과는 가치 기반의 강화학습에 비해 동질적 혹은 이질적인 데이터에 더 효과적인 탐지기를 제안합니다. 

이 논문의 기여는 다음과 같습니다. 첫 번째로, 비정상 탐지를 다루는데 가장 발전된 알고리즘인 **A3C** *asynchronous advantage actor-critic*을 기반으로 한 **PTAD** *policy-based DRL time series anomaly detector*를 제안했습니다. 두 번째로 가치 기반의 강화학습 탐지기 및 state-of-the-art 비정상 탐지 방법보다 더 나은 성능을 보이고 있습니다. 게다가 PTAD에서 비롯한 최적 확률적 탐지 정책은 정상과 비정상을 구분하는 임계치를 조정할 수 있습니다. 이를 통해 정밀도와 재현율의 균형을 제어할 수 있습니다.

## 2. Preliminaries

### 2.1 Background: Reinforcement Learning

논문에서는 강화학습에 대한 간단한 배경지식과 논문에서 제안하는 모델의 기반이 된 policy-based RL의 방법 중 하나인 actor-critic에 대해 소개하고 있습니다. 일반적인 내용이니 논문을 참고해주세요.

### 2.2 Formalization of RL based time series anomaly detection problem

시계열 비정상 탐지는 MDP로 고려할 수 있습니다. 현재 타임 스텝의 비정상성을 판단하여 이상 감지를 유발하는지 여부에 따라 환경이 변화하기 때문입니다. 그리고 다음 결정은 환경의 변화에 영향을 받습니다. 따라서 강화학습 프레임워크를 시계열 이상 감지에 적용하는 것은 적절한 방법일 수 있습니다.

* State
에이전트의 다음 action은 이전의 결정들과 현재 시계열로 이루어진 변화하는 환경에 영향을 받습니다. 따라서 state는 두 파트로 이루어집니다. 첫번째는 이전에 선택된 action들의 시퀀스인 $s_\text{action}=(a_{t-m},a_{t-m+1},\dots,a_{t-1})$ 이며 두번째는 현재의 시계열인 $s_\text{time} = (x_{t-m+1},x_{t-m+2},\dots,x_t)$ 입니다. 그리고 state space $S$의 크기는 무한합니다.

* Action
에이전트의 action $A = \{0,1\}$입니다. 이때 0은 정상을 의미하며, 1은 비정상이 감지됐음을 의미합니다.

* Reward
효과적인 정책을 수립하기 위해서는 적절한 보상 함수를 만드는 것이 중요합니다. 여기서는 라벨이 되어있는 학습 데이터를 사용했으며 라벨을 이용해 보상 함수를 설계했습니다. 에이전트의 행동의 결과를 보고 TP, FP, FN, TN 여부에 따라 각각 일정한 양의 보상을 주는 식으로 설계할 수 있으며, 보상의 양을 조절하여 어떤 것을 더 중요하게 고려할 지 설정할 수 있습니다.

## 3. Policy-based time series anomaly detector (PTAD)

![image](https://user-images.githubusercontent.com/35906602/142819224-e0b97ff4-b33d-44c4-ab6a-0d4e80347367.png){: width="600"}{: .align-center} 

Figure 1. The asynchronous interactions between the PTAD (the agent) and the labeled time series repository (the environment)
{: style="text-align: center; font-size:0.7em;"}

![image](https://user-images.githubusercontent.com/35906602/142819323-e52fa174-dc2b-47ab-bfe4-daf4a323e32e.png){: width="600"}{: .align-center} 

Figure 2. The internal structure of the PTAD.
{: style="text-align: center; font-size:0.7em;"}

강화학습 기반의 이상 탐지 상황에서 환경은 라벨이 된 시계열 데이터의 저장소와 같습니다. 이 데이터를 이용해 환경은 에이전트를 훈련시키기 위한 특정 state를 생성하고 에이전트의 Action을 평가합니다. 어떻게 시계열 비정상 탐지기를 작동시키고 최적화 할 지를 구현하는 에이전트의 설정도 중요합니다. 이 논문에서는 현재 $n$ 타임 스탬프와 과거의 $n$개의 결정을 입력으로 받아 출력으로 다음 타임 스탬프의 action을 결정합니다. 여기엔 어떠한 가정이나 제약이 없기 때문에 비슷한 시계열 비정상 탐지의 경우에도 적용될 수 있습니다.

### 3.1 The proposed approach

PTAD는 A3C 알고리즘으로 설계되었으며, A3C는 연속적인 사례들간의 상관관계를 줄이기 위해 비동기적 메커니즘을 적용했습니다. 위의 **Figure 1**은 PTAD의 에이전트와 환경 사이의 비동기적 상호작용을 나타내고 있습니다. **밑의 박스**는 환경을 나타냅니다. 각각 시계열 데이터를 가지만 $n$개의 독립적인 환경이 존재하며 이때 시퀀스의 순위는 일관적이지 않게 매겨집니다. 각 환경은 에이전트에게 시계열의 타임 스탬프를 주고, 받는 action에 따라 변화합니다. **위의 박스**는 PTAD를 의미하며 하나의 전역망과 $n$개의 지역망으로 이루어져 있습니다. 모든 망은 actor-critic 프레임워크를 가집니다. 각 지역망들은 각자의 환경을 탐험*explore*하며 받는 보상*reward*을 통해 기울기*gradient*를 계산합니다.  이 논문의 실험에서는 특정한 패턴의 비정상에 과적합되는 현상을 방지하기 위해 모든 에이전트가 서로 다른 시작 환경을 가집니다. 전역망은 기울기*gradient*들을 받아 정책*policy*를 최적화합니다.

Figure 2는 PTAD의 내부 구조이며, 세 가지 주요 요소들로 이루어져 있습니다. **LSTM**은 state로부터 연속적인 정보를 추출하기 위해 사용되며 인코딩된 속성을 출력합니다. State의 입력들은 정책을 추정하는 actor망과 state-value 함수를 근사하는 critic 망에 의해 유사하게 처리됩니다. FC 레이어는 LSTM의 출력을 입력으로 받으며, **actor망**에서는 각 action의 확률을 도출해내고, softmax 레이어를 통해 어떤 action을 취할 것인지를 결정합니다. **Critic 망**에서는 현재 state의 가치를 계산합니다. 

위의 과정에서는 목표 정책을 업데이트하지는 않으며, 대신 기울기를 계산하기 위한 샘플을 모으게 됩니다. Actor망의 손실과 기울기는 정책 $\pi$와 관련된 정책 경사 정리*policy gradient theorem*와 critic 망에서 주어지는 advantage function $A(s,a)$를 통해 계산됩니다. 이때 손실은 $r + v(s')$과 $v(s)$ 사이의 차이를 의미하며, $r$은 critic 망에서 주어진 state $s$일 때 actor 망에서 주어진 action을 통해 얻어진 reward를 의미합니다. 전역망이 매개변수들을 갱신하면 일관성을 유지하기 위해 지역망으로 보내게 됩니다.

### 3.2 Trait comparison

가치 기반의 강화학습 이상 탐지기에 비해 제안된 PTAD는 몇 가지 이점을 지니고 있습니다. 가치 기반의 탐지기는 결정론적**deterministic**인 정책을 가지게 되므로 특정 state에서 항상 같은 action을 선택합니다. 하지만 PTAD는 확률적**stochastic**한 정책을 산출하므로 판단의 임계치를  조정할 수 있습니다. 이를 이용하면 정밀도와 재현율 사이의 균형을 조절할 수 있으며 이는 실제로 알고리즘을 적용할 때 큰 이점을 가져옵니다. 

## 4. Experiments

### 4.1 Datasets

실험을 위해 이상 탐지를 위한 고전적인 데이터셋인 **Yahoo benchmark dataset**과 **Numenta Anomaly Detection (NAB)**가 사용되었습니다. 

#### Yahoo benchmark dataset

![image](https://user-images.githubusercontent.com/35906602/142984345-9c95b3d9-33ba-4a3f-981f-b3de740fbddf.png){: width="600"}{: .align-center} 

Figure 3. Yahoo benchmark samples.
{: style="text-align: center; font-size:0.7em;"}

데이터셋은 비정상을 포함하고 있는 시계열 데이터로 이루어져 있으며, {A1, A2, A3, A4}의 부분 데이터셋으로 이루어져 있습니다. 이 논문에서는 원천 데이터와 목표 데이터가 같을 경우를 시험하기 위해 A1과 A2가 사용되었습니다. 각각 67개, 100개의 시계열이 포함되어 있습니다. A1은 실제 야후의 로그인 데이터며 복잡한 시간적 패턴을 포함합니다. 각 시계열은 다양한 비정상 유형을 포함하고 있으며 시계열의 길이도 각각 다릅니다. 반면 A2의 경우 길이가 같으며 일시적인 비정상이 대부분이기 때문에 상대적으로 이상을 탐지해내기 쉽습니다.  반면 원천 데이터와 목표 데이터가 다른 경우를 시험하기 위해서는 4가지 유형이 모두 사용되었으며 총 367개의 시계열이 포함됩니다.

#### NAB dataset


![image](https://user-images.githubusercontent.com/35906602/142985120-3ec8a62c-b675-40ea-8fda-d2f5920bf111.png){: width="600"}{: .align-center} 

Figure 4. NAB dataset.
{: style="text-align: center; font-size:0.7em;"}

NAB dataset은 보통 알고리즘의 실시간 적용을 평가하기 위해 사용됩니다. 58개의 라벨된 실제 혹은 합성된 시계열이 있으며 각 시계열은 1000~22,000개의 타임 스탬프로 구성됩니다. 몇몇 시계열은 주기적이며 급격한 상승은 비정상이거나 혹은 맥락과 관계가 없을 수도 있습니다.

### 4.2 Evaluation metric

$$F_1 = \frac{2 \times \text{precision} \times \text{recall}}{\text{precision} + \text{recall}}$$

널리 사용되는 $F_1$ 스코어를 사용했습니다. 

### 4.3 Comparing methods and experimental setups

* **RNN-TAD** 인공 신경망 기반 비정상 탐지기로, LSTM을 활용해 학습해서 예측값과 실제값 사이의 차이를 활용해 비정상을 탐지합니다.
* **VTAD** 가치 기반 강화학습 시계열 비정상 탐지기이며 DQN을 사용합니다.

### 4.4 Comparison results

#### 4.4.1 Performance on the same source and target datasets

![image](https://user-images.githubusercontent.com/35906602/142987701-267d4259-67a4-4a74-9f97-3a17c931d225.png){: width="600"}{: .align-center} 

Figure 5. Training rewards of single time series with PTAD model in the Yahoo A1 and A2
{: style="text-align: center; font-size:0.7em;"}


Yahoo dataset의 A1과 A2를 비교하기 위해 RNN-TAD, VTAD, PTAD, 그리고 Twitter를 사용했습니다. Twitter는 통계적 특징만 학습합니다. 다른 3개의 알고리즘에 대해서는 학습 데이터와 시험 데이터를 8:2로 나누었습니다. 강화학습의 환경이 다양한 길이와 비정상 패턴을 포함한 시계열을 가지고 있기 때문에 누적 기대 보상의 변동성이 높습니다. 따라서 Figure 5와 같이 하나의 시계열에 대한 보상을 나타냈습니다. 비정상의 숫자가 상대적으로 적기 때문에 처음에 무작위 정책이 높은 보상을 받게 됩니다. 보상은 학습을 진행하며 진동하면서 상승합니다.

![image](https://user-images.githubusercontent.com/35906602/142988050-504140a7-4aca-4a06-995d-d1816bb411ae.png){: width="600"}{: .align-center} 

Table 1. Performance comparisons on the test part of the Yahoo A1
{: style="text-align: center; font-size:0.7em;"}

PTAD가 모든 시계열에 대해 가장 좋은 성능을 보이고 있지는 않지만 평균은 다른 모델에 비해 높으며, 표준편차도 제일 낮습니다. 또한 다른 모델들은 $F_1$ 스코어가 0.5를 넘는 경우가 적은 것으로 보아 A1 데이터에 대해 일반화를 제대로 못 하고 있다고 해석할 수 있습니다. 또 강화학습을 적용한 방식이 Twitter나 RNN-TAD에 비해 높은 것으로 보아 강화학습이 시계열 비정상 탐지를 위한 좋은 도구가 될 수 있다는 점을 알 수 있습니다. A2 데이터셋에 대해서는 RNN-TAD, VTAD, PTAD 모두 완벽한 성능을 보입니다.

#### 4.4.2 Performance on the different source and target datasets

Twitter, Skyline, Numenta, Numenta TM, contextOSE, RNN-TAD, VTAD, PTAD에 대해 Yahoo benchmark dataset을 원천으로 해서 NAB dataset을 목표로 하는 실험을 진행했습니다. 앞의 5개의 모델은 현재 시계열만 보고 비정상을 판단하기 때문에 Yahoo 데이터가 필요없으며 따라서 뒤의 3 모델에 대해 Yahoo 데이터로 학습을 진행했습니다.

![image](https://user-images.githubusercontent.com/35906602/142989519-d5ff4ee8-1d42-411d-a737-216dfecd95e0.png)
{: width="700"}{: .align-center} 

Table 2. Performance comparisons trained on Yahoo Benchmark dataset and are tested on Numenta dataset
{: style="text-align: center; font-size:0.7em;"}

대부분의 subset에서 PTAD가 제일 좋은 성능을 보이고 있습니다. 다른 state-of-the-art를 달성한 오픈 소스 탐지 기법과 비교하여 딥러닝을 사용한 RNN-TAD, VTAD, PTAD가 더 좋은 성능을 보이고 있습니다. 특히 강화학습 기반 기법은 다른 기법에 비해 한가지 경우를 제외하고는 훨씬 더 좋은 성능을 보이고 있습니다. 다만 Yahoo 데이터에는 주기적 데이터가 존재하지 않고, 비정상이 존재하지 않는 경우에 대해서는 학습하지 못 했기 때문에 *artificialNoAnomaly*에 대해서는 좋지 않는 성능을 보이고 있습니다. 

![image](https://user-images.githubusercontent.com/35906602/142990330-3430679d-e13a-431a-8279-1d65cb5d69a9.png){: width="600"}{: .align-center} 

Figure 6. Satisfactory detection results of PTAD in NAB datasets. The reward is 5 if the detector correctly distinguishes an anomaly, otherwise −1. The reward is 1 if the detector correctly judges a normal, otherwise −5.
{: style="text-align: center; font-size:0.7em;"}

위와 같은 만족스러운 결과를 볼 수 있었습니다. 왼쪽은 200-400 타임 스탬프 사이의 급증은 비정상이라고 보았으나 400-600 타임 스탬프에서 다시 나타난 급증에 대해서는 비정상이라고 판단하지 않았습니다. 오른쪽의 경우 복잡하고 잘 보이지 않는 패턴에 대해서도 어느정도 잘 대응하고 있음을 볼 수 있습니다. 

### 4.5 Adjustability of the PTAD

![image](https://user-images.githubusercontent.com/35906602/142991016-2a5ce015-3d1d-4955-855a-97f70823b8dc.png){: width="600"}{: .align-center} 

Figure 7. Variations of the precision, the recall and the F1 score under different thresholds in the ‘‘realKnownCause/null-nyc-taxi’’ and ‘‘realTraffic/null-occupancy-6005’’ time series.
{: style="text-align: center; font-size:0.7em;"}


PTAD의 정책은 의사결정에 대한 확률값을 제공하기 때문에 임계치를 조절함으로써 정밀도와 재현율 사이의 균형을 맞출  수 있습니다. 가령 임계치가 높게 설정된다면 탐지기는 비정상을 더 자주 탐지해낼 것이며 이는 높은 정밀도가 요구되는 경우에 적합합니다. 위와 같이 임계치를 조절함에 따라 $F1$ 스코어가 차이를 보임을 볼 수 있습니다. 가치 기반의 강화학습 방법론에선 조절이 불가능합니다.

![image](https://user-images.githubusercontent.com/35906602/142991486-437f3522-a54c-4368-a194-2b49b5c31575.png){: width="600"}{: .align-center} 

Figure 8. Comparing the unsatisfactory detection results of the ‘‘null-art-daily-perfect-square-wave’’ time series in the Numenta dataset. 
{: style="text-align: center; font-size:0.7em;"}

위와 같이 비정상을 탐지해낼 확률을 더 높게 설정한 경우 정밀도가 크게 향상되는 경우가 있을 수 있으며, 정밀도가 더 중요한 경우 이러한 방법을 활용할 수 있습니다.

### 5. Conclusion

이 논문에서는 일반적으로 적용 가능한 시계열 비정상 탐지를 위한 정책 기반의 강화학습 프레임워크를 제안했습니다. 이를 위해서는 A3C 알고리즘을 활용했습니다. 가치 기반의 강화학습 시계열 비정상 탐지기와 비교하면 이 논문에서 제안하는 탐지기는 더 좋은 탐지 성능을 보이면서 계산 복잡도도 낮췄습니다. 또 원천 데이터와 목표 데이터가 다를 경우, 같을 경우 양쪽에서 기존의 기법들을 앞섰습니다. 게다가 정책이 확률적이기 때문에 정밀도와 재현율 사이의 균형을 맞출 수도 있습니다.

후속 연구로, 주기성과 같은 정보들을 어떻게 적용할 것인지를 고려할 것입니다. 이에 더해 강화학습 예측기가 다음 타임 스탬프의 값을 예측하여 실제 값과 비교하는 방식으로 훈련하여 label의 영향을 덜 받고 반지도 학습 혹은 비지도 학습으로 확장하는 방법을 모색할 것입니다. 