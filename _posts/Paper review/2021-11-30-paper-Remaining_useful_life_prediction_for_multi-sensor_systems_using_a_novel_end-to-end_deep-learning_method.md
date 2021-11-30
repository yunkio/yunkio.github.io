---
date: 2021-11-30
title: "[Paper Review] Remaining Useful Life Prediction For Multi-sensor Systems Using a Novel End-to-end Deep-learning Method"
categories: 
  - Paper Review
tags: 
  - 딥러닝
  - RUL
  - 논문 리뷰
toc: true  
toc_sticky: true 
---
# Paper contents

Remaining useful life prediction for multi-sensor systems using a novel end-to-end deep-learning method

Yuyu Zhao, Yuxiao Wang

Measurement, 2021

https://www.sciencedirect.com/science/article/pii/S0263224121006515

## 0. Abstract

복잡한 시스템에서 기존 RUL 예측의 간접적인 방식은 보편성과 정확성을 제한합니다. 정확한 RUL 측정을 위해서는 RUL과 실제 데이터 사이의 잠재적 관계를 잘 파악하는 것이 중요합니다. 이를 위해 이 논문에서는 end-to-end RUL 예측 방법론을 제안합니다. 다변량 시계열 데이터를 다루기 위한 LSTM 인코더-디코더 구조가 사용됩니다. 또 입력된 특성과 시간적 상관관계의 적응적 추출과 평가를 위한 두단계의 attention 메커니즘이 사용됩니다. 그 후 다층 퍼셉트론을 통해 RUL 예측이 수행됩니다. 이 모델을 통해 아무런 사전지식 없이 선택적으로 핵심적인 정보에 집중할 수 있으며, 이는 RUL 예측의 정확도를 높이는 데 큰 의미가 있습니다. 제안된 모델의 효과성과 우수성이 turbofan engine 데이터를 통해 입증 되었습니다.

## 1. Introduction

공학적 시스템이 점점 더 통합되고 복잡해지면서 작은 고장 및 오작동이 전체 시스템에 많은 피해를 줄 수 있습니다. 이를 다루기 위해 PHM 방법론이 개발되어 공학적 시스템의 신뢰성과 효과성을 위해 널리 사용되고 있습니다. PHM은 기존의 스케쥴된 유지가 아닌, 현재 상태를 기반으로 한 유지를 하도록 하며 이때 제일 중요한 것은 RUL 예측입니다. RUL 예측은 고장이 발생하기 전 유지 계획을 위한 의미있는 정보를 제공합니다.

RUL 예측을 위한 많은 연구가 있어왔으며, 일반적으로는 모델 기반 혹은 데이터 기반으로 나뉩니다. 모델 기반의 방법은 시스템 고장에 대한 정확한 사전 지식에 의존하며 그렇기 때문에 현대의 복잡한 시스템은 적용하기 어려움이 있습니다. 더해서 각각의 시스템마다 다른 방식을 적용해야 하기 때문에 일반적으로 사용할 수 없습니다. 따라서 복잡한 시스템에서 모델 기반의 방법을 적용하는건 어렵고 실용적이지 않습니다. 

반면 데이터 기반 방법은 모니터링 데이터의 가용성 향상으로 복잡한 시스템의 RUL 예측에 더 유용해졌습니다. 멀티 센서 데이터와 시스템의 RUL 사이의 잠재적인 관계를 파악하는데 초점을 둡니다. 통계적 모델이나 확률 과정을 이용한 통계적 방법이 대표적인 방법입니다. 하지만 이런 방법은 데이터의 통계적 성격에 대한 가정에 의존합니다. 게다가 센서 데이터와 RUL 값이 HI *Health Indicator*를 기반으로 간접적으로 얻어지기 때문에 이 과정에서 정보의 손실이 발생할 수 있습니다. 

대신 머신러닝 알고리즘을 바탕으로 한 RUL 예측이 각광받고 있습니다. 여러가지 전통적인 머신러닝 모델이 사용됐지만 이러한 방법들은 보통 전문가의 feature engineering이 필요합니다. 하지만 딥러닝의 경우 더 큰 데이터에서 자동적으로 잠재적인 요소들을 얻어낼 수 있습니다. CNN, DBN, LSTM 등의 다양한 종류의 네트워크들이 적용됐습니다. 또 데이터에서 중요한 속성들을 추출하기 위해 AE가 사용됐으며, 시퀀스에서 정보를 선택하고 RUL을 예측하기 위해 GRU가 사용되기도 했습니다. 하지만 이러한 방법들은 멀티 센서 데이터들을 통합하여 RUL 예측을 하는 데에는 실패했습니다. 관련있는 속성들이 추출 될 수는 있지만 중요도와 상관없이 모두 똑같이 다뤄졌으며, 이는 예측의 정확도에 영향을 끼칠 수 있습니다. 더해서 현재의 RUL과 각각 다른 시간 간격을 지닌 과거 데이터의 시간적 상관관계는 RUL 예측 정확도에 매우 중요한 요소지만 여태까지는 반영되지 못 했습니다. 

이 논문에서는 멀티 센서에 대한 진단 문제를 다루게 됩니다. 앞서 언급되었던 접근 방식들의 한계를 극복하기 위해, 멀티 센서 데이터를 최대한 활용하는 실용적이고 더 정확한 end-to-end 방법론을 제안합니다. 주요 구조는 시계열 데이터와 장기 의존성을 학습할 수 있는 LSTM을 바탕으로 구성되었으며, 더 중요한 정보에 집중할 수 있도록 해주는 2단계의 attention 구조를 활용했습니다. 이를 통해 관련 있는 속성 및 시간적 상관관계를 동적으로, 적응적으로 추출 및 평가할 수 있습니다. 이를 통해 제안된 모델은 멀티 센서 데이터에서부터 RUL 예측을 직접적으로 해냅니다. 

## 2. Problem statement and method overview

![image](https://user-images.githubusercontent.com/35906602/144195176-88e2cf4b-f3f4-4d67-9572-9c77a6c1a0a2.png){: width="700"}{: .align-center} 

Figure 1. Brief flowchart of the proposed RUL prediction method
{: style="text-align: center; font-size:0.7em;"}

이 논문에서는 온도, 진동, 압력과 같은 요인들이 센서를 통해 저장되고 각 요소들은 시스템이 끝날 때까지 저장되는 run-to-failure 상황이라고 가정합니다. 각 멀티 센서 데이터는 다중 시계열의 형태를 띄고 있습니다. 모델 자체는 오프라인 상황에서 만들어지며 모델이 만들어진 후에는 과거 및 현재 데이터로 실시간 예측이 가능합니다. 

더 구체적으로, 오프라인 단계에서는 센서 데이터가 전처리되어 모델의 입력으로 들어가게 되고, 출력값은 실제값과 비교하여 모델이 학습됩니다. 온라인 단계에서는 현재 및 과거의 멀티 센서 데이터가 전처리되어 잘 훈련된 모델을 통해 실시간으로 예측을 수행하게 됩니다. 

## 3. Data preprocessing

### 3.1. Definition of RUL label

RUL은 현재로부터 시스템이 작동할 수 없을 때까지 남은 시간을 의미합니다. 실제 시스템에서는 구체적인 RUL 값은 존재하지 않습니다. 대신 run-to-failure 데이터에서는 해당 시스템의 최대 동작 시간과 현재 시간의 차로 구할 수 있습니다. 더해서 대부분의 경우 어느정도 작동 시간까지는 시스템의 상태 저하가 거의 일어나지 않으므로 이런 경우 RUL을 상수로 부여할 수 있습니다. 이를 반영하면 RUL은 비선형적으로 감소하게 됩니다.

### 3.2. Data normalization

각 센서는 서로 다른 범위의 값을 지니고 있으므로 일반화가 필요합니다. 일반적인 표준화 방법을 사용하고 있으므로 더 구체적인 설명은 생략합니다.

## 4. Proposed RUL prediction model

![image](https://user-images.githubusercontent.com/35906602/144201388-078bf4ee-ac65-42fd-876f-11f48bc4c694.png){: width="500"}{: .align-center} 

Figure 2. Graphical illustration of the proposed model.
{: style="text-align: center; font-size:0.7em;"}

RUL 예측은 곧 다변량 시계열 예측으로 볼 수 있습니다. LSTM은 시계열 데이터를 다루는데 적합하고 장기 의존성을 잘 학습합니다. LSTM에 대한 더 자세한 설명은 생략하겠습니다. 

멀티 센서 데이터에서는 각 데이터의 각 속성이 RUL에 영향을 미치는 정도가 다릅니다. 더해서 RUL 예측 결과는 시간에 따라 다르게 영향받으며, 시스템 상태의 악화의 변화 정도에 따라 시간적 상관관계가 달라집니다. 따라서 입력되는 가장 중요한 속성 및 시간적 상관관계를 추출하기 위해 attention이 사용 됐습니다.

이 논문에서는 LSTM의 인코더-디코더 구조가 모델의 핵심적인 구조로 사용됐으며 feature attention 및 temporal attention 구조가 적용 되었습니다. 전체적인 구조는 위 그림과 같습니다.

### 4.1. Multi-sensor data fusion via feature attention mechanism

![image](https://user-images.githubusercontent.com/35906602/144205578-8021b09a-87f3-4b7b-aad6-e771fee64c17.png){: width="600"}{: .align-center} 

Figure 3. Graphical illustration of the feature attention-based encoder.
{: style="text-align: center; font-size:0.7em;"}

복잡한 공학적 시스템에서 각 속성은 시스템의 상태 악화에 미치는 영향력이 서로 다릅니다. 이 속성들을 전부 비슷하게 고려하면 결과에 좋지 않은 영향을 미칠 수 있기 때문에 각 타임 스텝에서 각 속성의 영향력을 평가하기 위해 attention 구조를 적용했습니다. 이를 활용하면 아무런 사전 지식없이 멀티 센서 데이터를 다룰 수 있게 됩니다. 일반적으로 사용되는 attention 구조와 다르지 않으니 설명은 생략하겠습니다.

### 4.2. Temporal correlation extraction via temporal attention mechanism

![image](https://user-images.githubusercontent.com/35906602/144206276-cb4ab866-3834-4aa6-9dd9-19d2377d83c5.png){: width="600"}{: .align-center} 

Figure 4. Graphical illustration of the temporal attention-based decoder.
{: style="text-align: center; font-size:0.7em;"}

앞선 단계에서 feature attention-based 인코더는 멀티 센서 데이터에서 관련있는 속성을 평가하고 추출하기 위해 사용됐습니다. 인코더의 hidden state는 예측 결과에 서로 다르게 기여하는 각 시간에서의 여러 센서의 융합 결과입니다. 이에 더해서 동적으로 서로 다른 시간의 시간적 상관관계를 잡아내기 위한 temporal attention 구조가 디코더로 설계 되었습니다. 

이를 통해 관련있는 인코더의 hidden state들은 모든 시간에 걸쳐 적응적으로 선택됩니다. 이 역시 일반적인 two-stage attention 구조와 다르지 않으니 구체적인 설명은 생략하도록 하겠습니다. 

### 4.3. Final output generation via multilayer perceptron

앞서 설명한 구조들로 멀티 센서 데이터의 관련 있는 속성의 정보들과 시간적 상관관계를 추출할 수 있습니다. 마지막 디코더 단의 hidden state는 전체 과거 데이터를 담고 있으며 결과적으로 RUL 예측 결과를 얻을 수 있습니다. 따라서 이 정보들을 MLP의 입력으로 넣어 최종적인 출력을 산출하게 됩니다.

## 5. Experimental verification

실험은 공개된 C-MAPSS Turbofan Engine Dataset을 통해 진행됐습니다. 

### 5.1. C-MAPSS turbofan engine dataset

#### 5.1.1. Dataset overview

4개의 데이터셋으로 구성되어 있으며, 각 데이터셋은 서로 다른 조건과 고장 유형을 가집니다. 첫번째 데이터셋 FD001을 예로 들면 이 데이터셋은 다시 학습셋과 시험셋으로 나뉩니다. 학습셋에는 100개의 터보팬 엔진으로부터 나온 21개 센서의 데이터로 구성되어 있으며 고장 발생까지의 데이터로 이루어집니다. 시험셋은 마찬가지로 100개의 터보팬 엔진으로 이루어져 있지만 고장나기 전에 데이터가 끝납니다. 실제 RUL값은 RUL_FD001에 주어집니다. 나머지 3개의 데이터셋도 비슷하게 구성되어 있습니다. 실험의 목적은 학습셋에서 모델을 학습하고 시험셋에서 RUL 값을 예측하는 것입니다. FD001에는 학습셋에는 총 20,631개의 사이클이 있으며 시험셋에는 총 13,096개의 사이클이 있습니다.

#### 5.1.2. Data preprocessing

작동 내내 같은 값을 가지는 센서들이 있으므로 이 센서들은 학습시키지 않았습니다. 나머지 센서의 값들은 전부 학습에 사용됐으며, 앞서 말한 것과 같이 일반화 과정을 거쳐서 모델에 입력으로 들어갔습니다.

### 5.2. Performance Metrics

![image](https://user-images.githubusercontent.com/35906602/144239193-6edba41e-346f-4193-9c0a-3873b4b13029.png){: .align-center} 


평가 지표로는 두 지표가 쓰였습니다 첫 번째는 많이 쓰이는 RMSE입니다. RMSE는 초반부의 예측과 후반부의 예측을 동등하게 평가합니다. 두 번째 평가 지표는 다른 논문에서 제안된 평가 지표로, 후반부의 예측에 더 가중치를 두어 평가하는 방법으로, 실용적으로는 더 의미 있습니다. 

### 5.3. RUL prediction results and discussions

#### 5.3.1. RUL prediction results
![image](https://user-images.githubusercontent.com/35906602/144236750-f2f135ea-3d4f-432c-8686-a957e6e08438.png){: width="500"}{: .align-center} 

Figure 5. Feature attention weights of two samples.
{: style="text-align: center; font-size:0.7em;"}

위 그림은 FD001 데이터셋의 두 샘플의 feature attention을 나타낸 것입니다. 입력 속성은 14개고 타임 스텝은 30을 보므로 총 14 x 30개의 attention 가중치를 얻게 됩니다. 각 속성은 다른 영향령을 가지고 있다는 것을 알 수 있으며, 또 시간의 흐름에 따라 조금씩 다른 값을 보임도 알 수 있습니다. 

![image](https://user-images.githubusercontent.com/35906602/144237305-9c03caf4-f769-4f93-bc36-c906e76a0d18.png){: width="500"}{: .align-center} 

Figure 6. Temporal attention weights of two samples.
{: style="text-align: center; font-size:0.7em;"}

위 그림은 위에서 봤던 같은 샘플의 Temporal attention 가중치 입니다. 현재와 가까울수록 점점 더 높은 값을 가짐을 알 수 있습니다. 이는 우리의 직관과도 일치하는 결과입니다. 

![image](https://user-images.githubusercontent.com/35906602/144237657-3a8cf5b2-d9da-44af-aa48-7f2941480175.png){: width="600"}{: .align-center} 

Figure 7. Examples of RUL prediction results.
{: style="text-align: center; font-size:0.7em;"}

위 정보들을 활용해서 최종적으로 RUL값을 예측하게 됩니다. 직관적으로 쉽게 알 수 있는 그래프이므로 부연 설명은 생략합니다.


#### 5.3.2. Comparisons

![image](https://user-images.githubusercontent.com/35906602/144239256-53c9ade7-dea4-4206-b93d-167dbab8d08d.png){: width="600"}{: .align-center} 

결과는 위와 같습니다. 논문의 저자는 RMSE에 대해서는 1-FCLCNN-LSTM같은 모델은 50 스텝을 보고 있어 30 스텝을 보고 있는 본 논문의 모델보다 예측에 더 유리하지만, 이 논문에서 제안하는 모델은 계산상의 효율도 같이 고려했으므로 30스텝만 봤다고 얘기하고 있습니다. 또한 후반부의 예측 결과에 더 높은 가중치를 두는 Score가 실용적으로는 더 의미있다고 주장하고 있습니다. 


## 6. Conclusions

이 논문에서는 RUL 예측을 위한 end-to-end 모델을 제안합니다. 기여는 다음과 같습니다.

* 더 핵심적인 정보에 집중하기 위한 attention 구조가 쓰였으며 실제로 확인도 가능하기 때문에 더 실용적입니다. 더해서 이 방법을 활용하면 사전 지식이나 feature engineering이 필요 없습니다.

* 멀티 센서 데이터로부터 나오는 많은 양의 데이터를 한꺼번에 활용해 직접적으로 RUL을 예측합니다. 과정이 늘어난다면 정보의 손실이 발생할 수 있습니다.

* 여러 시스템에 널리 사용될 수 있습니다.

