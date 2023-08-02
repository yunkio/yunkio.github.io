---
date: 2023-08-02
title: "[Paper Review] Temporal regularized matrix factorization for high-dimensional time series prediction"
categories: 
  - Paper Review
tags: 
  - Time Series Prediction
toc: true  
toc_sticky: true 
---
# Paper contents

Temporal regularized matrix factorization for high-dimensional time series prediction.

Yu, H. F., Rao, N., & Dhillon, I. S.

Advances in neural information processing systems (2016)

https://proceedings.neurips.cc/paper_files/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html

## Motivation

기후학 및 수요 예측 등 많은 분야에서 시계열 예측이 중요해지고 있습니다.

![image](https://github.com/yunkio/SVM_tutorial/assets/35906602/e231d836-c62f-4fba-aa6d-4d151fc7ac9e){: width="400"}{: .align-center} 

Figure 1. Matrix Factorization model for multiple time series. F captures features for each time series in the matrix Y, and X captures the latent and time-varying variables.
{: style="text-align: center; font-size:0.7em;"}

고차원 시계열 데이터에는 계산 비효율성으로 인해 자기 회귀 모델 (AR) 혹은 동적 선형 모델 (DLM) 등의 고전적인 시계열 모델이 적합하지 않습니다. 행렬 인수 분해 (MF)는 고차원 시계열을 분석하기 위해 활용되었으나, 시간 임베딩 간의 순서를 고려하지 않아 시계열 분석에 적합하지 않습니다. MF에 그래프 기반 방법을 적용하여 시간 종속성을 처리하려는 시도가 있었으나, 시점 간의 음의 상관 관계가 있을 때 적용이 어렵다는 한계가 있습니다.

저자들은 이러한 기존 시계열 예측 방법론의 한계를 극복하기 위해 데이터 기반의 시계열 학습 및 예측을 위한 **TRMF** *Temporal regularized matrix factorization* 프레임워크를 제안합니다. 

## Proposed Method

**TRMF**는 고차원 시계열 분석을 위한 프레임워크입니다. 잠재적인 시간 임베딩 간의 시간 종속성의 구조를 설명하기 위해, 전통적인 행렬 인수 분해와 달리 시간 구조를 포함시켜 시계열 분석에 더 적합한 방법입니다.

#### 행렬 표현

시계열 데이터는 행렬의 형태로 나타낼 수 있으며, 여기서 행은 개별 시계열을 의미하고 열은 시점을 의미합니다. 기존 행렬 인수 분해 접근 방식은 이 행렬의 각 항목을 두 잠재 벡터의 내적으로 추정하며, 하나는 시계열(행)을 나타내고 다른 하나는 시점(열)을 나타냅니다.

#### 시간 의존성

기존의 행렬 분해는 시간 임베딩의 순서를 고려하지 않기 때문에 시계열 데이터에 적합하지 않습니다. 반면 TRMF는 시간 구조를 행렬 분해 공식에 포함시키기 위해 시간 정규화를 활용합니다. 이를 통해 데이터 기반의 시간 의존성 학습이 가능해집니다.

#### Autoregressive Temporal Regularizer

![image](https://github.com/yunkio/SVM_tutorial/assets/35906602/633a95a6-0009-474f-b038-fa2f90668e90){: width="500"}{: .align-center} 

Figure 2. Graph-based regularization for temporal dependencies.
{: style="text-align: center; font-size:0.7em;"}

Autoregressive Temporal Regularizer는 각 시간 임베딩이 이전 임베딩의 선형 조합으로 표현되기 때문에 시간 임베딩 간의 AR 구조를 띄고 있습니다. 기존의 많은 행렬 분해 방식들은 시간 의존성을 처리하기 위해 그래프 기반의 정규화를 사용하지만, 이러한 방법은 시점 간의 음의 상관 관계가 있을 때 한계를 보입니다. 하지만 Autoregressive Temporal Regularizer를 활용하면 음의 상관 관계도 잘 처리가 가능합니다.

#### 특징
TRMF를 활용하면 시계열 데이터의 결측치 및 노이즈를 효과적으로 처리할 수 있으며, 데이터 기반의 시간 의존성을 학습하여 미래 값의 예측 또한 가능합니다. 이러한 TRMF의 특징은 TRMF 프레임워크를 더 유연하고 적응 가능하게끔 만들어 주기 때문에 다양한 특징을 지닌 시계열 데이터를 일반적으로 처리 가능하며, 예측 성능 또한 뛰어납니다.


## Experiments

![image](https://github.com/yunkio/SVM_tutorial/assets/35906602/39f62494-df2f-458a-909e-5a3e7244ca5d){: width="600"}{: .align-center} 

![image](https://github.com/yunkio/SVM_tutorial/assets/35906602/9da7556e-2156-422d-90c3-342e012845c1){: width="400"}{: .align-center} 

![image](https://github.com/yunkio/SVM_tutorial/assets/35906602/f1dade32-f3dc-457f-812d-88026327db01){: width="500"}{: .align-center} 

시계열 예측 및 Imputation, 그리고 Scalability에서 2016년 당시의 기존 방법론보다 뛰어난 성능을 달성 했다는 내용입니다. 특히 커다란 데이터 셋에서 좋은 성능을 보였습니다.