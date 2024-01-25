---
date: 2024-01-25
title: "[Paper Review] Drift doesn’t Matter: Dynamic Decomposition with Diffusion Reconstruction for Unstable Multivariate Time Series Anomaly Detection"
categories: 
  - Paper Review
tags: 
  - Anomaly Detection
toc: true  
toc_sticky: true 
---

## Reference

Drift doesn’t Matter: Dynamic Decomposition with Diffusion Reconstruction for Unstable Multivariate Time Series Anomaly Detection

C Wang, Z Zhuang, Q Qi, J Wang, X Wang, H Sun, J Liao

Neural Information Processing Systems (2023)

https://openreview.net/forum?id=aW5bSuduF1

## Introduction

본 연구는 다변량 시계열 이상 탐지에 대해서 다루고 있습니다. 기존 방법들은 불안정한 데이터 패턴과 이로 인한 모델의 조정을 위해 높은 훈련 비용이 소모됩니다. 이 논문은 이러한 문제들을 해결하기 위해 **Dynamic Decomposition with Diffusion Reconstruction (D3R)**이라는 새로운 접근 방식을 제안합니다. D3R은 불안정한 데이터를 *stable* 부분과 *trend* 부분으로 분해하여, 지역적인 슬라이딩 윈도우 방법의 한계를 극복하고, *noise diffusion*을 통해 정보의 병목을 외부적으로 제어합니다.  이를 통해 본 논문은 장기간 다변량 시계열을 위한 동적 분해 방법, 정보의 병을 제어하기 위한 새로운 접근 방식, 그리고 이를 통해 불안정한 데이터셋을 효과적으로 처리하는 D3R 이상 탐지 네트워크를 제안합니다.

## Proposed Method

![image](https://github.com/yunkio/SSL_Tutorial/assets/35906602/94cebc99-a54d-45bd-a45d-c88e3c4815c3){: width="600"}{: .align-center} 

Figure 1. The architecture of D3R mainly consists of two modules: dynamic decomposition and diffusion reconstruction.
{: style="text-align: center; font-size:0.7em;"}
![image](https://github.com/ML4ITS/mtad-gat-pytorch/assets/35906602/db62d218-ac12-4655-9225-19c6e57f3cee

D3R의 구조는 크게 **dynamic decomposition module** 과 **diffusion reconstruction module** 로 구성됩니다. **dynamic decomposition module**은 데이터 인코더와 시간 인코더를 사용하여 데이터와 시간 특징을 모델링합니다. 그런 다음 안정적인 구성 요소를 추출하기 위해 쌓인 분해 블록을 사용하고 offset subtraction을 통해 트렌드 구성 요소를 도출합니다. 반면에 **diffusion reconstruction module**은 외부에서 정보 병목을 구축하기 위해 noise diffusion을 사용합니다. 백본 네트워크를 사용하여 오염된 데이터를 재구성하며, 재구성 오는 이상 점수로 사용됩니다. 데이터 인코더와 재구성 백본 네트워크 모두 시간적 및 차원 의존성을 모델링하기 위해 쌓인 공간-시간 트랜스프모 블록을 포함합니다. 또한 모델의 강건성을 높이기 위해 훈련 중에 disturbance 전략을 사용합니다.

### Data Preprocessing

**Time stamp hardembedding** 1차원으로 구성된 timestamp를 5차원으로 재구성하며, 각 차원은 각각 minute of the hour, hour of the day, day of the week, day of the month, month of the year을 의미합니다.

**Labeled stable component construction** 트렌드를 추출하기 위해 moving average를 사용하며, 이를 통해 라벨이 달린 stable component $S=X-T$를 얻을 수 있습니다. 이렇게 라벨이 된 stable component를 통해 훈련 데이터가 불안정할 때 발생할 수 있는 훈련 방해를 어느정도 방지할 수 있습니다.

**Disturbance Strategy** 모델의 강건성을 위해 훈련 데이터의 각 변수에 $[-p, p]$의 uniform distribution을 vertical drift로 적용합니다.

### Dynamic Decomposition

Dynamic decomposition module은 크게 4개의 구성요소로 이루어집니다. 

**Data encoder** 시간적 및 차원적 의존성을 포착하기 위해 spatio-temporal transformer 블록을 기반으로 구현되며, 이는 모델 내에서 $d_\text{model}$이라는 은닉 상태 차원을 가지는 $\mathbb{R}^{n×d_\text{model}}$ 형태의 $H_\text{data}$ 출력을 생성합니다

**Time encoder** temporal trasnformer로만 구성되어 있으며, 이를 통해 모델은 시점의 $H_\text{data} \in \mathbb{R}^{n×d_\text{model}}$ 형태의 시간적 correlation을 얻을 수 있습니다.

**Stacked decomposition blocks** Stacked decomposition blocks로 이루어진 Data-time mix-attention을 통해 stable component $\hat{S} \in \mathbb{R}^{n×k}$를 얻습니다.

**offset subtraction** Horizontal drift를 해결하기 위해 offset subtraction을 통해 $\hat{T}_d\in \mathbb{R}^{n×k}$를 출력합니다.

### Diffusion reconstruction

Diffusion reconstruction module은 크게 2개의 구성요소로 이루어집니다.

**Noise Diffusion** 외부에서 정보 병목을 생성하는 역할을 합니다. 이는 입력 데이터에 의도적으로 소음을 도입하여 '오염시키는' 방식으로 이루어집니다.

**Backbone network** 이 소음에 의해 오염된 데이터를 직접 재구성하는 임무를 맡고 있습니다. 이를 통해, 모듈은 데이터 내의 이상을 정확하게 식별하려고 하며, 이 과정에서 발생하는 재구성 오차는 이상치 점수로 사용됩니다.

### Joint optimization

D3R 구조에서 Dynamic Decomposition과 Diffusion Reconstruction은 긴밀하게 연결되어 있습니다. Dynamic Decomposition은 장기간의 불안정한 다변량 시계열의 고유한 특성, 즉 안정적인 구성 요소를 학습하는 데 중점을 둡니다. 실제 안정적인 구성 요소($S$)와 추정된 안정적인 구성 요소($\hat{S}$) 사이의 평균 제곱 오차(MSE)를 이 모듈의 손실 함수로 사용합니다. 한편, 확산 재구성 모듈은 noise diffusion에 의해 오염된 데이터를 재구성하는 업무를 맡고 있습니다. 특히 이 모듈의 출력은 $X_d$의 재구성된 버전입니다. $X$의 재구성은 $X^d$에서 drift $\mathbf{d}$를 빼서 얻어집니다. 동적 분해 모듈과 마찬가지로, 실제 데이터 X와 그 재구성된 버전 Xˆ 사이의 MSE를 확산 재구성 모듈의 손실 함수로 직접 사용합니다. D3R은 훈련 과정에서 end-to-end로 학습되기 위해 두 목적 함수를 더해서 사용합니다.

## Experiments

실험 결과는 다음과 같습니다. 
![image](https://github.com/yunkio/SSL_Tutorial/assets/35906602/6b47e567-913c-4564-a3d3-c636351eeb84){: width="600"}{: .align-center} 

![image](https://github.com/yunkio/SSL_Tutorial/assets/35906602/e28f717d-babe-428a-b377-5ff56b91188b){: width="600"}{: .align-center} 

![image](https://github.com/yunkio/SSL_Tutorial/assets/35906602/55704915-4807-453b-bb48-f73a731c47c4){: width="600"}{: .align-center} 





