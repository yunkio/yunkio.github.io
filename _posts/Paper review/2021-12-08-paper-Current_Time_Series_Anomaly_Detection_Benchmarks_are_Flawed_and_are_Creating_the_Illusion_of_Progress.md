---
date: 2021-12-08
title: "[Paper Review] Fault Diagnosis of an Autonomous Vehicle With an Improved SVM Algorithm Subject to Unbalanced Datasets"
categories: 
  - Paper Review
tags: 
  - 딥러닝
  - Fault Detection
  - 논문 리뷰
toc: true  
toc_sticky: true 
---
# Paper contents

Fault Diagnosis of an Autonomous Vehicle With an Improved SVM Algorithm Subject to Unbalanced Datasets

Qian Shi, Hui Zhang

IEEE Transactions on Industrial Electronics, 2021.

https://ieeexplore.ieee.org/document/9097402

## 0. Abstract

이 논문에서는 고장을 발견하는 관측자를 설계하는 것의 복잡성을 피하기 위해 SVM 분류기를 바탕으로 한 steering actuator 고장 진단을 제안합니다. 특히 데이터의 불균형이 성능에 악영향을 끼칠 수 있기 때문에 이를 해결하기 위해 lda를 통한 undersampling의 과정 및 grey wolf optimizer 알고리즘을 활용한 임계치 조정이 제안 되었습니다. 이러한 과정을 통해 기존 분류 모델보다 더 뛰어난 성능을 보였습니다.

## 1. Introduction

자율주행차의 경우 시스템에 대한 신뢰도가 매우 중요합니다. 이러한 시스템의 고장 진단에 대한 많은 연구가 진행되어 왔으며 크게 model-based 접근법과 process-history-based 접근법으로 나눌 수 있습니다. 

**Model-based** 접근법에서는 고장 진단은 UIO *unknown input observer* 설계 문제로 볼 수 있습니다. 알려지지 않은 고장이 동적 시스템에 추가 입력으로 모델링되며 이를 추정하는 것이 고장 진단의 목표입니다. 

이와 달리 **process-history-based** 방법은 고장 진단을 분류 문제로 정의하여 관찰자를 정의하는 문제의 복잡성을 피합니다. CAN *controller area network*은 자율주행차에 널리 사용되었으며 차량 시스템의 데이터 확보가 용이하여 시스템의 이력 데이터를 기반으로 개발된 process-history-based 방법은 빠른 응답, 더 쉬운 적용, 그리고 사전 지식에 대한 요구 사항이 적다는 점 등의 장점이 있습니다.

이러한 방법의 분류기는 주로 통계적 방법이나 머신러닝 알고리즘을 통해 만들어집니다. 특히 브레이크 시스템의 고장 감지를 위해서 정상 데이터와 고장 데이터의 불균형을 다룰 수 있도록 수정된 SVM이 제안되었습니다. 여태 연구를 통해 제안된 대부분의 고장 진단 접근방법은 센서를 통해 측정된 데이터를 사용하였으며 데이터의 통계적 특징을 얻기 위해 전처리 되었습니다.

이러한 연구들과는 다르게 자율 주행차의 고장 진단은 환경과 이용자의 명령에 따라 작동 환경이 다르기 때문에 분류기에 사용되는 특징들이 다양한 환경에는 둔감하면서도 동시에 고장에는 민감해야 합니다. 이를 위해 2-DOF *two-degree-of-freedom* 차량 모델이 자율주행차의 움직임을 추정하기 위해 사용되었으며, 실제 값과 측정된 값의 차이가 훈련 데이터로 사용 되었습니다. 

이 논문에서는 SVM 알고리즘이 분류 특징에 강건하고 SVM 분류기는 작은 샘플 셋(support vector)에 의존하며 이론적으로 분류기의 성능이 보장된다는 사실에 고려하여 SVM을 채택했습니다. 하지만 고장 데이터는 정상 데이터에 비해 매우 수집되기 힘들다는 문제가 있습니다. 이와 같은 데이터 불균형에 의해 전통적인 SVM 방법은 편향된 결과를 산출하게 됩니다. 이러한 문제를 해결하기 위해 이 논문에서는 데이터의 불균형을 고려할 수 있는 향상된 SVM을 제안합니다. 이 논문의 기여는 다음과 같습니다.

첫 번째로, 제안된 steering 고장 진단 방법은 모델 기반의 residual generator와 SVM이 혼합된 방법입니다. 두 번째로, 데이터의 언더샘플링을 위해 LDA를 적용함으로써 균형 잡힌 훈련 데이터셋을 만들어 냈습니다. 세 번째로 GWO *Grey Wolf Optimizaer* 라는 swarm intelligence optimization 방법을 적용하여 SVM 알고리즘의 임계치를 결정해 불균형 데이터의 분류 성능을 높였향상시켰습니다.

## 2. Structure and Metrics of Improved Fault Diagnosis Method

### 2.1. Framework of Fault Diagnosis

![image](https://user-images.githubusercontent.com/35906602/145175092-9083032a-87f3-43db-81db-70f5aec22170.png){: width="600"}{: .align-center} 

Figure 1. Framework of fault diagnosis for autonomous vehicle’s steering actuator.
{: style="text-align: center; font-size:0.7em;"}

이 논문에서는 steering actuator의 고장을 분석합니다. Steering actuator의 고장은 크게 세 가지 분류로 나눌 수 있습니다. 현재 조향 각도의 값을 유지하는 **stuck**은 베어링의 결함이나 shaft의 변형이 있을 때 발생합니다. 기계적 요소가 마모되면 명령의 일부만 작동하는 **loss of effectiveness**가 발생합니다. 또 조향 각도가 명령을 벗어나는 **bias**는 센서의 오프셋으로 인해 발생할 수 있습니다. 

또 조향 명령이 원하는 조향 입력이라고 가정하고 차량의 모든 액츄에이터와 센서 중 오직 조향 액츄에이터만 고장날 가능성이 있다고 가정합니다. vehicle model을 사용하여 요 각속도 *yaw rate*와 횡속도 *lateral speed*를 추정함과 동시에 실제 값을 측정합니다. 그 후 잔차 벡터를 생성하기 위해 추정 값과 실제 값을 비교합니다. 이렇게 생성된 2D 잔차 벡터를 활용하여 SVM을 통해 고장을 진단합니다. 이를 통해 모델이 지나치게 복잡해지고 환경 및 조건에 따라 결과가 달라지는 것을 막을 수 있습니다. 이와 같은 방법은 steering actuator 뿐만 아니라 다른 방법에도 확장이 가능합니다.

### 2.2. SVM Algorithm for Fault Diagnosis

고장 진단은 분류 문제로 생각할 수 있습니다. 정상 데이터를 1, 고장 데이터를 -1로 라벨링 했습니다. SVM의 목적은 이 데이터를 활용해 최적의 분류기를 만드는 것입니다. 고장 진단을 위해 SVM을 적용할 때 고장 데이터에 비해 정상 데이터가 훨씬 많아 불균형 문제가 발생합니다. 보통 불균형 데이터에선 SVM이 제 성능을 내지 못 합니다. 특히 고장 데이터를 오분류 하는 것은 매우 위험하기 때문에 더더욱 문제가 될 수 있습니다. 따라서 이를 해결하기 위한 평가지표 및 불균형 데이터를 위한 SVM을 설명하겠습니다.

### 2.3. Evaluation Metrics of the Proposed Method

![image](https://user-images.githubusercontent.com/35906602/145228861-fcc5f46d-e60c-46aa-9f2e-7a1c6cba8f8c.png){: width="400"}

일반적으로 사용하는 평가 지표인 accuracy는 고장 데이터를 오분류 하더라도 높게  나오므로 이와 같은 불균형 데이터에서는 적합하지 않습니다. 따라서 이 논문에서는 평가 지표로 G-mean이 사용됩니다. G-mean은 민감도 *sensitivity*와 특이도 *specificity*의 기하 평균입니다. 이 평가 지표를 사용하면 적은 클래스의 오분류에 대해 더 민감하게 반응합니다.

## 3. Improved SVM for Unbalanced Data

이 논문에서는 정보가 많이 담긴 인스턴스를 선택하기 위해 LDA가 전처리 과정에 사용되며, 이렇게 선택된 인스턴스의 수는 고장 데이터의 수와 같습니다. 이렇게 균형잡힌 데이터가 선택된 후에는 분류기의 임계치가 GWO 알고리즘을 통해 최적화 됩니다.

### 3.1. Undersampling of Majority Class by LDA

데이터 불균형을 다루기 위해서는 소수 클래스를 ROS 혹은 SMOTE등의 알고리즘을 통해 새 데이터를 생성하는 오버 샘플링을 할 수도 있습니다. 하지만 이렇게 생성된 데이터들은 기존 데이터와 무관하거나 혹은 의미있는 정보를 제공하지 않을 수 있습니다. 반면 언더 샘플링 방법은 크기가 큰 클래스에서 데이터를 제거하는 방식으로 이루어지며, 오버샘플링보다 더 성능이 좋다는 선행 연구들이 있습니다. 따라서 여기서는 언더샘플링을 적용했습니다.

언더샘플링 방법에 대한 RUS *random undersampling*을 포함한 여러 연구가 있습니다. 그 중 많은 연구가 특징 공간 위에서 인스턴스간의 거리를 바탕으로 이루어집니다. 이 논문에서는 한발 더 나아가 LDA를 통해 투사된 거리를 적용하여 더 의미있는 데이터를 선택하도록 했습니다. 보통은 분류 혹은 차원을 줄이기 위한 전처리 과정으로 사용되지만 여기서는 majority class에서 의미있는 데이터를 선택하기 위해 사용됩니다. LDA가 의미있는 인스턴스들, 즉 정보를 많이 담고 있다고 여겨지는 샘플을 선택합니다. 자세한 과정은 다음과 같습니다.

1. 다수의 정상 데이터 $n_1$개와 소수의 고장 데이터 $n_2$개를 수집합니다.
2. LDA를 통해 두 클래스 사이의 분리점*seperation point* $\theta$를 구합니다.
3. 다수 클래스의 데이터들과 $\theta$ 사이의 유클리디안 거리를 구한 뒤, 다수 클래스에서 $\theta$와 가장 가까운 $n_2$개 데이터를 찾습니다. 이 데이터들이 의미있는 데이터를 담고 있다고 가정합니다.
4. 이렇게 선택된 $n_2$개의 majority 데이터와 원래 존재하던 $n_2$개의 minority 데이터를 사용하여 균형있는 데이터셋을 만듭니다.
5. 이렇게 만들어낸 데이터셋으로 SVM 분류기를 훈련시킵니다.
6. 위에서 선택되지 않은 majority 클래스의 $n_1 - n_2$개 데이터를 사용하여 분류 오류를 시험합니다.
7. 만약 분류 오류가 감소했다면 minority 클래스의 데이터, 오분류된 데이터, 그리고 majority 클래스의 서포트 벡터를 조합하여 새로운 훈련 데이터를 만들고 위 과정을 반복합니다.

### 3.2. Threshold Adjustment for an Optimal G-Mean

위 과정을 통해 LDA를 바탕으로 한 언더샘플링이 수행됐습니다. 위와 같이 균형잡힌 데이터를 만들어낸다고 하더라도 여전히 SVM을 통해 경계선을 만드는 것은 어려울 수 있습니다. 따라서 위에서 훈련된 SVM을 조정해야 합니다. 경계선 $b$를 조정하는 일은 최적화 방법으로 볼 수 있습니다. 최적화 식은 다음과 같습니다.

$$\max_b\text{G-mean}(b)$$

이를 위해 이 논문에서는 군집 지능법*swarm intellgance method*인 GWO 알고리즘을 적용했습니다. 과정은 다음과 같습니다. 

1. 정상 데이터와 고장 데이터를 포함하고 있는 residual 벡터들을 훈련 데이터 $D_{\text{tr}}^\text{original}$과 시험 데이터 $D_{te}$로 1:1 비율로 나눕니다.

2. 불균형 훈련 데이터셋 $D_\text{tr}^\text{original}$을 다시 훈련셋 $D_\text{tr}$과 검증셋 $D_\text{v}$ 1:1 비율로 나눕니다.

3. 훈련 데이터셋 $D_\text{tr}$의 $n_1+n_2$ 샘플을 LDA 언더샘플링 과정에 넣습니다.

4. 이 과정에서 검증셋 $D_v$는 적합성 함수 $\text{F}(b)$의 평가를 위해 쓰이며, 여기서 $\text{F}(b)$는 G-mean을 의미합니다. 이때 SVM 분류기는 서포트 벡터 $x_\text{su}$, 서포트 벡터 라벨 $y_\text{su}$, 라그랑주 승수 $\Lambda^*$를 포함합니다.

여기서 grey wolf group의 크기 $m$과 최대 반복 횟수는 유저에 의해 설정됩니다.

1. 입력으로 $x_\text{su}$, $y_\text{su}$, $\Lambda^*$, $b$를 넣고 $x_\text{test}$를 넣어 G-mean $z$를 계산합니다.
2. grey wolf population을 초기화합니다.
3. 최적의 search agent들인  $X_\alpha$, $X_\beta$, $X_\delta$를 찾습니다.
4. 최대 반복 횟수까지 GWO를 활용해 각 search agent의 위치를 조정합니다.
5. 최대 반복 횟수에 도달하면 $X_\alpha$를 최적의 경계선 $b*$로 사용합니다.

GWO의 출력인 $X_\alpha$는 최적의 임계치 $b^*$ 입니다. Search agent는 가능한 해결책을 의미하며, 위에서 $X_\alpha$는 가장 높은 $F(b)$를 가지는 search agent, $X_\beta$는 두 번째로 높은 $F(b)$를 가지는 search agent, $X_\delta$는 세 번째로 높은 $F(b)$를 가지는 search agent를 의미합니다.

GWO은 exploration과 exploitation 사이의 균형을 잘 맞추며 따라서 local optimum을 피할 수 있습니다. 더해서 다른 최적화 방법들에 비해 좋은 성능을 보이고 있습니다.

## 4. Validation and Experiment Results

### 4.1. Model Validation

![image](https://user-images.githubusercontent.com/35906602/145255529-e616c56b-1e4d-4b71-a7c7-a9020778303a.png){: width="600"}{: .align-center} 


KEEL로부터 나온 널리 쓰이는 공개된 데이터 셋 17가지를 사용해 검증을 진행했습니다. 위 표에서 IR은 불균형의 비율을 의미합니다. 이때 비교 대상은 널리 대중화된 불균형 데이터를 다루는 접근 방식 6가지를 선택했습니다. 또한 제안된 방법의 효과성을 보이기 위해 표준적인 SVM 역시 실험에 사용 되었습니다. 결과는 다음과 같습니다.

![image](https://user-images.githubusercontent.com/35906602/145255906-e2713b4c-9689-45e2-8f28-a0935e75c5e9.png){: width="700"}{: .align-center} 

값은 G-mean을 의미하며, 표준 편차를 함께 표시했습니다. 표준적인 SVM은 데이터의 불균형 비율에 매우 큰 영향을 받음을 알 수 있습니다. 이 논문에서 제안된 방법론이 17개의 데이터 중 8개의 데이터에서 가장 높은 성능을 보이고 있습니다. 또한 다른 언더샘플링 방법 (RUS, CL-SVM, NM2-SVM)과 비교하면 이 논문에서 제안된 방법이 모든 데이터에서 더 좋은 성능을 보이고 있습니다.

### 4.2. Steering Actuator Fault Diagnosis for an Autonomous Vehicle

![image](https://user-images.githubusercontent.com/35906602/145260033-3eb34c2e-b98e-499f-ba4e-9f007a03edc5.png){: width="600"}{: .align-center} 

Figure 2. 2-DOF bicycle model of vehicle lateral dynamics for estimated signals.
{: style="text-align: center; font-size:0.7em;"}



이제 측정값과 예측값의 차이를 통한 잔차의 분류를 통해 steering actuator의 고장을 진단해보도록 하겠습니다. 데이터의 출처는 East Liberty, OH, USA의 Transportation Research Center에서 수행된 시험 주행입니다. Longitudinal, lateral, yaw, 그리고 차량의 pitch motion이 측정되었으며, 예측값은 2-DOF bicycle model을 통해 얻어졌습니다. 앞서 말했던 세가지 오류에 대해 제안된 프레임워크의 분류 결과는 다음과 같습니다.

![image](https://user-images.githubusercontent.com/35906602/145269255-4c27becd-d93e-4da1-a47c-f575f210a988.png){: width="500"}{: .align-center} 

Figure 3-1. Test residual samples of fault one and normal class.
{: style="text-align: center; font-size:0.7em;"}


![image](https://user-images.githubusercontent.com/35906602/145269260-fc2c1eee-d758-41e6-9f85-3181112639da.png){: width="500"}{: .align-center} 

Figure 3-2. Test residual samples of fault two and normal class.
{: style="text-align: center; font-size:0.7em;"}


![image](https://user-images.githubusercontent.com/35906602/145269273-547ec606-4470-4267-9730-00be4870d3ba.png){: width="500"}{: .align-center} 

Figure 3-3. Test residual samples of fault three and normal class.
{: style="text-align: center; font-size:0.7em;"}

오류가 발생한 경우에 대해서는 거의 확실하게 분류를 해내고 있다는 점을 알 수 있습니다. 

![image](https://user-images.githubusercontent.com/35906602/145269544-d95f9fca-5be0-4fa7-9419-1b4093fb1bb6.png){: width="500"}{: .align-center} 

다른 모델과 G-mean 값을 비교한 결과는 위와 같으며, 다른 모델에 비해 뛰어난 성능을 보이고 있음을 알 수 있습니다.


## 5. Conclusion

고장 진단의 데이터 불균형 문제를 다루기 위해서 향상된 SVM 프레임워크가 제안 되었습니다. 제안된 프레임워크는 균형잡힌 데이터셋을 만들기 위한 LDA를 바탕으로 한 언더샘플링 방법 및 GWO 알고리즘을 바탕으로 한 결정 경계 최적화 방법을 포함하고 있습니다. G-mean 평가 지표가 적용되었으며 공개된 불균형 데이터인 KEEL를 바탕으로 6개의 방법과 비교를 진행했습니다. 위 방법들과의 비교를 통해 제안된 방법이 더 나은 G-mean 값을 가진다는 점을 보였습니다. 이에 더해 차량 steering actuator의 현장 데이터의 고장 진단에도 적용되어 모든 종류의 고장에 대해 더 나은 성능을 보였습니다. 