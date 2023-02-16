---
date: 2023-02-16
title: "[Paper Review] Anomaly detection for tabular data with internal contrastive learning"
categories: 
  - Paper Review
tags: 
  - Anomaly Detection
  - Contrastive Learning
  - Representation Learning
toc: true  
toc_sticky: true 
---

# Paper contents

Deep Learning-based Multi-Horizon Forecasting for Automated Material Handling System Throughput in Semiconductor Fab

Choi, J., Kang, H., Kim, J., Choi, H., Lee, Y., & Kang, P

IEEE Transactions on Semiconductor Manufacturing (2022)

https://ieeexplore.ieee.org/abstract/document/9946427

## Motivation

![image](https://user-images.githubusercontent.com/35906602/216506926-b2fb8318-3257-44fb-ac0c-4b744f86839e.png){: width="500"}{: .align-center} 

Figure 1. System throughput trends are affected by various internal and external variables in the AMHS environment.
{: style="text-align: center; font-size:0.7em;"}

Automated material handling systems (AMHS)는 반도체 공정에서 웨이퍼를 적재적소에 공급하는 것을 목적으로 합니다. 이를 위해서 생산 시스템의 상태를 모니터링하고, 더 나아가서 미래의 상태를 미리 예측 할 수 있다면 최적의 material handling 환경을 유지할 수 있습니다. 이 논문에서는 이를 위한 Multihorizon 예측을 위한 딥러닝 기반의 프레임워크를 제안합니다. 이를 위해서 우선 이상치 탐지 모델을 통해 이상치를 제거하여 훈련 데이터로 사용해 더 강건한 예측을 가능하도록 했습니다. 

## Method

![image](https://user-images.githubusercontent.com/35906602/216507068-654caeee-c081-4143-94f8-2ecbdd21a213.png){: width="500"}{: .align-center} 

Figure 2. AMHS thorughput forecasting framework architecture.
{: style="text-align: center; font-size:0.7em;"}

이 논문에서 제안하는 시스템의 구조는 크게 3가지 단계로 구성됩니다. 

### Data collection and system feature extraction

데이터를 수집하는 단계입니다. 크게 특별한 점은 없으므로 설명은 생략하겠습니다.

### Anomaly interpolation

![image](https://user-images.githubusercontent.com/35906602/216507536-d4bc49e6-e9e3-4b68-82a4-4398bdfc8bb8.png){: width="500"}{: .align-center} 

Figure 3. Anomaly detection and replace anomalies using linear interpolation.
{: style="text-align: center; font-size:0.7em;"}


수집 된 데이터에서 이상치를 제거하는 단계입니다. 여기서는 Isolation Forest가 활용됩니다. iForest를 통해 이상치를 제거하고, 선형 보간법을 사용해서 해당 이상치를 대체합니다. 

### DL-based multi-horizon forecasting and interpretation

![image](https://user-images.githubusercontent.com/35906602/216508046-335ff553-4c82-4e60-be7c-d18013afc632.png){: width="500"}{: .align-center} 

Figure 4. Deep learning-based multi-horizon forecasting models.
{: style="text-align: center; font-size:0.7em;"}


제안되는 프레임 워크에서는 iterative 방법과 directed 방법을 함께 사용합니다. 이를 위해서 본 논문에서는 4가지 모델 (LSTM, DeepAR, N-BEATS, TFT)의 앙상블을 활용하였으며 다른 딥러닝 방법도 얼마든지 추가 가능합니다. 또한 TFT에 포함되어 있는 Variable Section module을 통해 해석력도 갖출 수 있습니다. 

## Experiments

![image](https://user-images.githubusercontent.com/35906602/216508908-f1af0026-e7d0-4cc3-b280-de1ad2eb8f0b.png){: width="400"}{: .align-center} 

Figure 5. Time series cross-validation for experiment
{: style="text-align: center; font-size:0.7em;"}

실험은 다음과 같습니다. 크게 특별한 내용은 없으므로 자세한 설명은 논문을 참고해주세요.


![image](https://user-images.githubusercontent.com/35906602/216509175-99dc7df1-59d2-4d69-afd2-494ca74f4be6.png){: width="600"}{: .align-center} 
