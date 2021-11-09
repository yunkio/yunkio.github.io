---
date: 2021-11-09
title: "[Paper Review] Multi-Sensor Fault Detection, Identification, Isolation and Health Forecasting for Autonomous Vehicles"
categories: 
  - Paper Review
tags: 
  - 머신러닝
  - Fault Detection
  - Forecasting
  - Health Index
  - 논문 리뷰
toc: true  
toc_sticky: true 
---
# Paper contents

Multi-Sensor Fault Detection, Identification, Isolation and Health Forecasting for Autonomous Vehicles

Saeid Safavi, Mohammad Amin Safavi, Hossein Hamid, Saber Fallah

Sensors, 2021.

https://www.mdpi.com/1424-8220/21/7/2547

## 0. Abstract

자율주행 연구에는 주행의 신뢰성과 정확성이 매우 중요합니다. 센서의 결함으로 인해 고장이 발생하는 경우가 많으며 치명적인 결과로 이어질 수 있습니다. 그러므로 가능한 조기에 문제를 예측하는 것이 중요합니다. 이를 위해 이 논문에서는 다중 센서 시스템에 다중 fault를 예측, 식별, 감지하기 위한 아키텍쳐를 제안합니다. 두 개의 구분되는 딥 뉴럴 네트워크를 사용하며 좋은 성능을 얻었습니다. 또 모델의 결과를 활용해 **건강 지수** *health index*를 도입하고, 이를 예측하는 네트워크를 만들었습니다.

## 1. Introduction

센서 모니터링 시스템의 목적은 Fault가 있는 센서를 감지, 격리 및 식별하며 센서의 성능과 신뢰성을 예측하는 것입니다. Fault는 크게 sensor fault, actuator fault, part or process fault로 분류됩니다. **Sensor fault**는 입력 모듈의 오류, **actuator fault**는 출력 모듈의 오류를 의미합니다. 이러한 위험을 통제하기 위한 방법으로는 시스템에 결함이 있는지 감지해내는 **fault detection**, 어떤 센서가 문제인지 찾아내는 **fault isolation**, 센서가 고장난 원인을 규명하는 **fault identification**, 그리고 센서의 현재 상태와 미래 상태를 보여주는 **sensor health forecasting strategy**가 있습니다. 

이러한 문제들을 해결하기 위해 기존에는 규칙 기반 혹은 모델 기반의 방법들이 제안되었지만, 이론적 모델과 규칙을 도출하기에는 너무 복잡하다는 문제가 있었습니다. 최근에는 컴퓨팅 파워의 발전, 센서 기술의 성장 등을 이유로 여러 분야에서 다양한 데이터 기반 접근 방식이 제안됐으며 더 효과적임이 입증되고 있습니다. 또한 딥러닝 기법은 최첨단 프레임워크로 주목받고 있습니다. 접근 방식은 크게 **(i) 지식 기반** : 모델, 규칙 및 온톨로지 기반 방법 , **(ii) 머신러닝** : SVM, K-NN, ANN, **(iii) 딥러닝** : DNN, RNN, CNN, AE, RBM과 같은  방법들로 나눌 수 있습니다. 

기존의 머신 러닝 방법 중 **SVM**과 **랜덤 포레스트**는 시스템 health monitoring을 위한 일반적인 기술입니다. 좋은 일반화 성능으로 높은 분류 및 회귀 정학도를 제공합니다. 하지만 대부분의 경우 수작업으로 만들어 낸 피쳐들을 필요로 합니다. 또 차원의 저주, 혹은 컴퓨팅 파워 등의 문제도 존재합니다. 딥러닝으로는 이 문제들을 극복 가능합니다.

![image](https://user-images.githubusercontent.com/35906602/140870773-1de47680-1bef-4df8-bd1b-613a3303738c.png){: width="500"}{: .align-center} 

Figure 1. Different forecasting categories
{: style="text-align: center; font-size:0.7em;"}


### Health prognostics

**Health prognostics**는 forecasting의 응용입니다. **Forecasting**은 크게 두 가지 주요 분류로 나누어지며, **PHM** *prognostics and health management* 연구의 대부분은 수명 예측에 집중됐습니다. 주로 터빈, 베터리, 기어, 베이렁 등의 부품에 적용됐으며 이러한 부품들은 고장의 특정 임계값을 가지며 시스템의 상태는 단조롭다*monotonic*고 여겨집니다. Health prognostic은 크게 **(i) RUL** *remaining useful life* : 회귀 모델을 활용한 잔존 수명 예측, **(ii) Early failure detection** : 분류 모델을 활용한 고장 감지, **(iii) anomaly detection** : 비정상 탐지, 분석, 예측으로 나눌 수 있습니다.

**RUL** 예측은 이전의 고장 데이터가 있어야 하며, 일부 부품의 경우 가속 시험을 통해 이 이러한 데이터를 얻을 수 있습니다. 하지만 이런 방법은 제한된 부품에만 적용이 가능하며 특히 전기 센서의 경우 데이터가 거의 존재하지 않습니다. 

**Early failure detection**의 목적은 과거 데이터의 연속된 패턴을 학습해서 어떤 패턴이 고장으로 이어지는 지를 식별해내는 것입니다. 이진 분류 문제로 표현되기도 합니다. 단점으로는 주로 긴 기간의 상태는 보기 힘들고 바로 앞의 상태만 예측할 수 있기 때문에 *early* 라고 불립니다.

### 이 논문에서는..

이 논문에서는 이상치 감지, 분석, 예측에서의 health monitoring strategy에 초점을 두고 있습니다. **Outlier detection**이라고도 불리는 이 방법은 전통적으로 fraud detection, target tracking, 네트워크 보안 등에 쓰이고 최근에는 무선 센서 네트워크에도 적용되고 있습니다. 이러한 맥락에서 **Sensor outlier**들은 주어진 시나리오에서 나타나지 않던 패턴이 관측될 경우의 불규칙한 값들을 의미합니다. 

Seonsor prognostic에서 중요한 과제 중 하나는 데이터 불균형입니다. 비정상적인 관측치는 정상 데이터에 비해 매우 드물게 발생합니다. 또 하나는 센서는 degradation path에 대한 정보가 부족하다는 것입니다. 따라서 일반적으로 기계 부품에 사용하는 RUL 방법은 센서에는 적용하기 어렵습니다.

이 논문의 기여는 다음과 같습니다. 첫째로 불균형한 데이터셋에서의 신뢰성 있는 health index measure를 제안합니다. 둘째로, 센서의 health index를 예측하고 치명적인 고장이 발생할 때 까지 남은 시간을 가늠할 수 있는 이상치 예측 알고리즘을 제안합니다. 이 알고리즘은 해석 가능한 attention 메커니즘과 분위 회귀를 사용하기 때문에 실용적입니다. 셋째로, 신뢰성 있는 고장 탐지를 위해 다중 센서를 위한 통일된 프레임워크를 제안합니다. 실시간 다중 센서의 고장을 탐지하고 HI (Health Index)를 만들기 위해 CNN 분류기가 사용되었습니다. 또 HI를 활용해 센서의 상태를 예측하기 위해서 단변량 시계열 예측 문제에 사용되는 TFT (Temporal Fusion Transformers) 가 사용됩니다. 더해서 고장의 유형을 식별하기 위한 특징 추출과 DNN 분류 모델을 제안했습니다. 이러한 방법들을 시험하기 위해 아우디의 실제 데이터셋을 사용했습니다. 

## 2. Dataset Description

![image](https://user-images.githubusercontent.com/35906602/140879141-7bbb4247-9def-4268-909c-778a87bdb0ad.png){: width="600"}{: .align-center} 

Figure 2. Three vehicle bus data-points are depicted over time
{: style="text-align: center; font-size:0.7em;"}

아우디에서 최근에 공개한 A2D2 (버스 운행 데이터셋) 을 사용했습니다. 이 데이터는 버스가 고속도로, 도시, 교외 등에서 운행한 데이터이며 주행 기록, 가속, 속도, GPS, 브레이크 압력 등의 데이터가 포함되어 있습니다. 이 논문에서는 그 중에서도 교외와 고속도로 운행 데이터를 사용했으며 센서 데이터는 가속 페달, 핸들 각도, 그리고 브레이크 압력의 3개 센서에서 나온 데이터를 사용했습니다. 

이 데이터에는 세 개의 다른 도시의 데이터가 있는데, 그 중에서 데이터 수가 가장 많은 Ingolstadt의 데이터를 활용하여 네트워크를 훈련했으며, 가장 데이터 수가 적은 Gaimersheim의 데이터로 검증 및 매개변수 최적화를 했습니다. 

### Fault Injection for Fault Detection Task

센서 고장의 일부 원인은 일시적입니다. 가령 전자기 간섭등의 원인이 있을 수 있습니다. 이로 인해 센서의 health는 수명동안 유동적이게 되고, 따라서 기계 부품에 쓰이는 단조로운 health index는 효과적이지 않습니다. 따라서 이 논문에서는 센서 연결망의 health forecasting에 사용될 수 있는 단조롭지 않은 health index를 도입했습니다. 다양한 패턴의 고장이 발생할 수 있으며 이 논문에서는 간헐적 혹은 연속적 결함을 다음과 같이 분류하여 데이터에 삽입했습니다.

* Drift fault : 신호가 실제 값에서 선형적으로 멀어짐
* Hard over fault : 센서가 측정 가능한 범위의 값을 반환하고 포화점으로 빠르게 증가함
* Erratic fault : 노이즈가 크게 증가하여 실제 값 주변에서 진동이 강하게 발생함
* Spike fault : 신호 값이 간헐적으로 증가하며 스파이크 결함의 밀도는 점점 더 증가할 수 있음

![image](https://user-images.githubusercontent.com/35906602/140888554-8c1ef614-422d-4940-8e25-17b99f3839c5.png){: width="600"}{: .align-center} 

Table 1. Number of signals in all of the faulty/healthy combinations
{: style="text-align: center; font-size:0.7em;"}


이때 세 가지 센서에서 나오는 가능한 모든 고장의 조합을 삽입했습니다. 위 표에서 F와 H는 각각 faulty 신호와 healthy 신호를 의미합니다. 또한 결함의 강도를 시뮬레이션 하기 위해 정규 분포가 사용되었으며, Full-Scale Output (FSO)의 20%의 에러에 대해서는 erratic 혹은 drift 결함으로 정의되었습니다.

![image](https://user-images.githubusercontent.com/35906602/140889313-fe5ccfe8-4102-43da-b307-af31a8946932.png){: width="600"}{: .align-center} 

Figure 3. Example of sensor faults.
{: style="text-align: center; font-size:0.7em;"}


## 3. System Description

이 논문에서 다루는 주제는 **Integrated Vehicle Health Monitoring**이라고 불립니다. 현재의 상태를 살핌과 동시에 미래의 상태를 예측합니다. 전체 구조는 다음과 같습니다.

![image](https://user-images.githubusercontent.com/35906602/140891231-f308525e-3e2e-464f-8121-ea31ba72c6d9.png){: width="300"}{: .align-center} 

Figure 4. Proposed system architecture
{: style="text-align: center; font-size:0.7em;"}

### 3.1. Sensor Fault Detection

기존의 방법은 대부분 특징 추출과 분류의 두 가지 분리된 부분으로 이루어져 있습니다. 이때  수작업으로 특징 추출을 하는 것은 높은 계산 비용을 요구하고 실시간 적용에 어려움이 있습니다. 이 논문에서는 내재적인 적응형 구조를 활용하여 빠르고 정확한 센서 모니터링 및 조기 고장 감지 방법을 가능하도록 하며, 특징 추출 및 분류 단계를 하나의 학습 단계로 합쳤습니다. 이러한 접근법은 원 데이터에 효과적이며 특징 추출의 필요성을 완화하여 속도와 정확도 양쪽에서 좋은 성능을 보입니다. 자세한 구조는 다음 그림을 참고해주세요.

![image](https://user-images.githubusercontent.com/35906602/140892562-1bd17dc8-e46f-4ac5-901f-780b5dd77c5f.png){: width="500"}{: .align-center} 

Figure 5. The convolution structures of the suggested 1D CNN adaptive configuration
{: style="text-align: center; font-size:0.7em;"}

### 3.2. Sensor Fault Identification and Isolation

제안된 구조에서 Fault identification and isolation 위한 모델은 특징 추출과 DNN의 두 부분으로 이루어져 있습니다. 이 작업은 실시간으로 이루어 질 필요가 없기 때문에 별도의 특징 추출 구조를 포함하도록 했습니다. DNN은 고장 유형과 어떤 센서에서 고장이 발생했는지를 식별합니다. 

#### 3.2.1. Feature Extraction

![image](https://user-images.githubusercontent.com/35906602/140898298-2ebc1b93-50d1-4f74-b145-e330d256e03f.png){: width="600"}{: .align-center} 

Table 2. Time-domain feature definitions used for N data points in a sample X
{: style="text-align: center; font-size:0.7em;"}

신호에서 고장이 발견되면 신호들은 특징 추출 함수로 들어갑니다. 특징 추출을 위해서는 보통 시간 기반의 분석, 빈도 기반의 분석, 그리고 두 분석의 혼합이 사용됩니다. 이 논문에서는 위 표에 설명된 것과 같은 10가지의 시간 기반의 특징이 사용됐습니다.

#### 3.2.2. Fault Isolation and Identification Based on Multi-Class DNN

![image](https://user-images.githubusercontent.com/35906602/140899283-8762253b-e52f-43ef-9e64-92f9fec963b7.png){: width="400"}{: .align-center} 

Figure 6. The proposed multi-class DNN architecture
{: style="text-align: center; font-size:0.7em;"}

앞선 특징 추출 단계에서 추출해낸 특징을 바탕으로 DNN에 입력으로 넣게 됩니다. 자세한 구조는 위 그림을 참고해주세요.

### 3.3. Sensor Health Forecasting

이 논문에서는 TFT network를 바탕으로 한 health index 방법과 health forecasting strategy를 제안하고 있습니다.

#### 3.3.1. Health Index Definition

![image](https://user-images.githubusercontent.com/35906602/140900108-39e2d0b3-0d54-4d96-8116-fd9dea6d853d.png){: width="700"}{: .align-center} 

Figure 7. Three different degradation scenarios
{: style="text-align: center; font-size:0.7em;"}

A2D2 데이터가 degraded 신호를 포함하고 있지 않기 때문에, degradation process가 각 운행 시나리오에 인위적으로 도입 되었습니다. 이러한 degradation path는 앞선 연구를 참조하여 적용됐습니다. 위 그림은 각 degradation의 패턴과 이로부터 Health Index를 만드는 과정을 설명하고 있습니다. (a)는 선형적으로 증가하는 erratic fault, (b)는 지수적으로 증가하는 drift fault, (c)는 주기적인 degrade 패턴의 spike fault, (d)는 fault가 없는 경우를 각각 나타냅니다.

각 경우의 첫 번째 그래프는 실제 데이터를 의미하며 두 번째 그래프는 CNN에 통과한 출력값을 의미합니다. 세 번째 그래프는 두 번째 그래프의 필터링된 모습으로 고장 감지 네트워크의 전체적인 경향을 나타냅니다. 앞서 설명한 것과 같이 센서는 기계 부품과 달리 단조롭지 않다고 가정하므로 평탄화된 경향을 살펴보도록 했습니다. 마지막 그래프는 세 번째 그래프의 Health Score에서 Fault Score를 뺀 값으로 이 값이 최종적인 Health Index가 됩니다.

#### 3.3.2. Sensor Health Forecasting Strategy

![image](https://user-images.githubusercontent.com/35906602/140902531-7f701ec3-15fa-4562-bc71-f4fc00d0cc84.png){: width="600"}{: .align-center} 

Figure 8. Temporal Fusion Transformers (TFT) architectural design
{: style="text-align: center; font-size:0.7em;"}

센서 상태 예측을 위해 위에서 구한 Health Index로 TFT network를 학습 시킵니다. 다중 전망 예측은 변수들 간의 동적인 관계도 고려해야 합니다. TFT는 이러한 다중 전망 예측에 뛰어난 성능을 보이는 attention-based 모델로, 2020년에 제안됐습니다. 다양한 규모에서 시간적 연관성을 학습하게 되며 다양한 시나리오에서 고성능을 보입니다. TFT의 주요 구조는 다음과 같습니다.

* 아키텍처의 중복 요소를 건너뛰고 적응형 깊이와 네트워크 복잡성을 제공하는 Gate 구조
* 각 시점에서 적절한 변수를 선택하기 위한 네트워크
* Static covariate encoder를 활용해 컨텍스트 벡터를 인코딩하여 네트워크에 정적인 특성들을 통합 
*static covariate encoders to incorporate static features into network by encoding contextual vectors to condition temporal dynamics.*
* 장기, 단기 시간 관계를 둘 다 학습하기 위해 관찰된 신호와 시간에 따라 변하는 신호 양쪽에서 계산 수행, local 층에선 sequence-to-sequence 레이어가 사용되고 동시에 innovative interpretable multi-head attention block을 사용해 장기 의존성을 고려
* 목표 값의 스펙트럼을 평가하기 위한 Quantile forecast

## 4. Experimental Results and Discussion

A2D2 데이터는 health 진단의 맥락에서 사용된 적이 없습니다. 따라서 여기서는 2021년에 발표된 SVM을 활용한 고장 진단을 위한 프레임워크와 비교했습니다.

### 4.1. Sensor Fault Detection

![image](https://user-images.githubusercontent.com/35906602/140905378-41f07a62-8c32-4aba-9ae9-7fb4dda56ff9.png){: width="500"}{: .align-center} 

Table 3. Sensor fault detection performance
{: style="text-align: center; font-size:0.7em;"}

### 4.2. Sensor Fault Identification and Isolation

![image](https://user-images.githubusercontent.com/35906602/140905584-e2c974fe-5ab4-41c5-ada3-a1b283420240.png){: width="500"}{: .align-center} 

Table 4. Sensor isolation accuracy for SVM and DNN
{: style="text-align: center; font-size:0.7em;"}

![image](https://user-images.githubusercontent.com/35906602/140905701-8361dc84-7eb4-46e0-bcec-132d3163765e.png){: width="500"}{: .align-center} 

Table 5. Fault identification accuracy by multi-class DNNs
{: style="text-align: center; font-size:0.7em;"}

### 4.3. Sensor Health Forecasting

TFT 네트워크는 [시작점으로부터 흐른 시간, 도시 ID, Health Index sample]의 데이터를 입력으로 받습니다. 과거의 100 시점을 입력으로 받아서 미래의 50 시점을 예측하도록 합니다. 

#### 4.3.1. Performance Measure

분위수 회귀를 사용해서 각 시점에서 10%, 50%, 90%의 분위수에 해당하는 값을 예측하도록 했습니다. 

![image](https://user-images.githubusercontent.com/35906602/140908695-e9d49de9-9ae5-4c8e-b6f0-8de6206382d7.png){: width="200"}{: .align-center} 

$t$시점에서 예측된 $\tau$ 번째 앞 스텝의 $q$th 분위수를 의미합니다.

#### 4.3.2. Prediction Results

![image](https://user-images.githubusercontent.com/35906602/140906799-c5a308a4-6af0-4586-a8d3-e3b6438439b8.png){: width="500"}{: .align-center} 

Table 6. Quantile loss for test data
{: style="text-align: center; font-size:0.7em;"}

![image](https://user-images.githubusercontent.com/35906602/140907290-d3986b65-1c7d-41f4-90c3-b2962c0a92b3.png){: width="600"}{: .align-center} 

Figure 9. Test data vs TFT forecasts for three different quantiles
{: style="text-align: center; font-size:0.7em;"}


TFT 모델은 Ingolstadt 데이터로 학습되었습니다. 위 표는 테스트 데이터에서의 P10, P50, P90 분위수 손실을 나타내며, 그림은 각 모델을 사용했을 때의 예측값과 실제값의 비교입니다. P10은 가장 보수적으로 예측을 하기 때문에 최악의 시나리오로 사용할 수 있고, P50은 실제값과 가장 비슷하며, P90은 가장 너그럽게 예측합니다. 

## 5. Conclusions

이 논문에서는 센서 고장의 물리적 토대를 조사하고 이를 4가지로 분류했습니다. 궁극적 목표는 고장 감지, 식별, 예측을 포함한 health monitoring 프레임워크를 설계하는 것입니다. CNN 기반의 고장 감지기는 다중 센서에서 나타날 수 있는 고장 조합의 다양한 조합을 효과적으로 다룰 수 있음을 밝혔습니다. 고장 감지에서는 99.84%의 정확도를 보였습니다. 다음 단계로 만약 신호에서 고장이 감지됐다면 신호는 고장 식별 시스템을 통과하게 되고, 고장이 감지되지 않았다면 센서 health 예상 시스템을 통과하게 됩니다.

고장 식별 시스템으로는 다중 센서 제어 시스템에서 DNN 모델들이 활용됐습니다. 이를 통해 가장 인식하기 어려운 드리프트 오류 등을 포함하여 서로 다른 센서들에 대해 높은 식별 정확도를 보였습니다.

Health 예측 시스템에선 센서의 상태를 예측하기 위해 세 센서의 각기 다른 degradation path들을 활용하여 HI를 정의했습니다. HI는 TFT 네트워크에 입력되어 센서의 미래 상태와 발생할 수 있는 문제점을 예측하게 됩니다. 이 논문에선 성능 측정을 위해 분위수 손실을 활용하였습니다.