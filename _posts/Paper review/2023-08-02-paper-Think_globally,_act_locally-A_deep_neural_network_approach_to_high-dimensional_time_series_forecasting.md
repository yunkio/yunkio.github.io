---
date: 2023-08-02
title: "[Paper Review] Think globally, act locally: A deep neural network approach to high-dimensional time series forecasting"
categories: 
  - Paper Review
tags: 
  - Time Series Prediction
toc: true  
toc_sticky: true 
---

## Reference

Think globally, act locally: A deep neural network approach to high-dimensional time series forecasting

Sen, R., Yu, H. F., & Dhillon, I. S.

Advances in neural information processing systems (2019)

https://proceedings.neurips.cc/paper/2019/hash/3a0844cee4fcf57de0c71e9ad3035478-Abstract.html

## Motivation

최근 많은 시계열 데이터들은 서로 상관성이 있는 수 많은 시계열로 이루어진 매우 고차원의 다변량 시계열 데이터입니다. 이러한 고차원의 시계열 데이터를 분석하는 것은 어려운 일입니다. 이러한 시계열을 효과적으로 예측하기 위해서는 전역적인 (global한) 패턴을 분석함과 동시에 지역적인 특성을 함께 활용해야 합니다. 본 논문 발표 당시(2019년)의 대부분의 딥러닝을 활용한 접근법은 단변량 시계열을 대상으로 과거의 값을 바탕으로 미래의 값을 예측합니다. 

본 논문에서는 **DeepGLO** 를 제안합니다. **DeepGLO**는 전역적인 행렬 분해 시간적 합성곱 네트워크로 제약되는 전역적인 행렬 분해 모델(Global matrix factorization model regularized by a temporal convolution network)을 활용하며, 동시에 각 시계열의 지역적인 특성을 포착하는 또 다른 시간적 네트워크 (another temporal network)를 사용합니다. 

제안하는 모델은 일반화나 스케일링 같은 과정 없이 다양한 특성을 지닌 고차원의 시계열 데이터를 통해 훈련 가능하며 실험적인 결과를 통해 매우 뛰어난 성능을 보였습니다.

## Proposed Method

### LeveledInit: Handling Diverse Scales with TCN
![image](https://github.com/ML4ITS/mtad-gat-pytorch/assets/35906602/b8bddb36-4c04-42da-a58a-7f3a9a974dcb){: width="400"}{: .align-center} 

Figure 1. An illustration of a TCN.
{: style="text-align: center; font-size:0.7em;"}

딥러닝에서는 개별 시계열들이 다양한 스케일을 가질 때 학습이 어려울 수 있습니다. 이를 해결하기 위해서는 보통 정규화 혹은 스케일링 방법들이 활용됩니다. 

**TCN**은 1D 합성곱 층으로 이루어진 다층 신경망입니다. 각 레이어의 *dilation* 하이퍼 파라미터를 설정하고, 네트워크는 필터 크기와 레이어의 수를 기반으로 *look-back*을 지닙니다. 이를 통해 TCN은 이전 값을 입력으로 받아 한 단계 미래의 예측 값을 출력하게 됩니다.

![image](https://github.com/ML4ITS/mtad-gat-pytorch/assets/35906602/3d05cb37-5071-4e2f-9e9e-f8723c51802b){: width="400"}{: .align-center} 


본 논문은 TCN의 파라미터 초기화를 위해 **LeveledInit**을 제시합니다. 기존의 *Xavier* 방식이나 *He* 방식은 매우 깊은 네트워크에는 적합하지 않다고 주장합니다. 네트워크를 서로 다른 세부 초기화 전략을 가진 여러 단계로 나누어 줍니다. 주어진 윈도우의 과거 값들을 예측하는 형태로 네트워크 파라미터를 초기화시키며, 이를 통해 학습 도중에 네트워크가 예측한 평균 주변의 분산들을 학습하도록 하며, 이 분산들은 상대적으로 스케일에서 자유롭습니다. 처음 단계에서는 *vanishing* 문제가 발생하지 않도록 매우 공격적으로 초기화를 진행하고, 깊은 곳에 있는 네트워크일수록 *exploding* 문제가 발생하지 않도록 더 보수적인 값을 취합니다. 이러한 방법을 통해 더 안정적인 활성화가 가능해지며, 수렴이 빨라지고, 일반화 성능이 높아진다고 주장합니다. 

### DeepGLO

**DeepGLO**는 전역적으로 생각하며 지역적으로 행동하도록 설계된 예측 모델입니다. 전역적으로는 행렬 분해를 통해 제약된 TCN (TCN-MF)를 사용하며, 지역적으로는 LeveledInit이 적용된 TCN을 사용합니다. 

![image](https://github.com/ML4ITS/mtad-gat-pytorch/assets/35906602/e9911dfe-112f-4ada-893c-cfc82e8321a0){: width="400"}{: .align-center} 

![image](https://github.com/ML4ITS/mtad-gat-pytorch/assets/35906602/5f957361-9bfd-48cd-9b6a-975e21352c7e){: width="600"}{: .align-center} 

Figure 2. We show some of the basis time-series extracted from the traffic dataset, which can be combined linearly to yield individual original time-series. 
{: style="text-align: center; font-size:1em;"}

![image](https://github.com/ML4ITS/mtad-gat-pytorch/assets/35906602/5390494d-389a-4220-a70d-011e1b7c0ef5){: width="400"}{: .align-center} 

![image](https://github.com/ML4ITS/mtad-gat-pytorch/assets/35906602/80f302f5-250d-40c2-86b8-247d96ac1baa){: width="600"}{: .align-center} 

Figure 3. An illustration of DeepGLO. The TCN shown is the network, which takes in as input the original time-points, the original covariates and the output of the global model as covariates. Thus this network can combine the local properties with the output of the global model during prediction.
{: style="text-align: center; font-size:1em;"}


## Experiments 

![image](https://github.com/ML4ITS/mtad-gat-pytorch/assets/35906602/74a4f8ee-d632-4031-ac37-937914ed00c9){: width="800"}{: .align-center} 

실험 결과입니다. 성능이 좋았다고 합니다.


W.I.P