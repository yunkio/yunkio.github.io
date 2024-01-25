---
date: 2024-01-25
title: "[Paper Review] MEMTO: Memory-guided Transformer for Multivariate Time Series Anomaly Detection
"
categories: 
  - Paper Review
tags: 
  - Anomaly Detection
toc: true  
toc_sticky: true 
---

## Reference

MEMTO: Memory-guided Transformer for Multivariate Time Series Anomaly Detection

J Song, K Kim, J Oh, S Cho

preprint

https://arxiv.org/abs/2312.02530

## Introduction

본 논문은 데이터 불균형 및 라벨이 없는 이상 탐지 등의 과제를 해결하기 위한 **MEMTO**(Memory-guided Transformer for multivariate time series)를 제안합니다. MEMTO는 정상 패턴을 대표하는 **Gated Memory Module**을 사용하며, 다양한 정상 패턴에 적응하기 위해 점진적으로 훈련됩니다. 기존 방법들(OC-SVM, SVDD, isolation forest, LOF)과 딥러닝 방법(DAGMM, Deep SVDD, THOC, LSTM-VAE)과 달리, MEMTO는 메모리 항목에 저장된 정상 패턴의 특성을 사용하여 이상을 재구성하기 어렵게 함으로써 재구성 기반 방법의 과도한 일반화 문제를 극복합니다. 안정적인 훈련을 보장하기 위한 2단계의 훈련 패러다임이 도입되었으며, 이상 탐지를 위해 잠재 공간과 입력 공간을 모두 고려하는 bi-dimensional deviation-based detection criterion이 사용됩니다. 


## Proposed Method

!![image](https://github.com/yunkio/SSL_Tutorial/assets/35906602/9b689061-e1ab-4466-8411-8e7b3eaaa2bb){: width="600"}{: .align-center} 

Figure 1: Illustration of proposed MEMTO.
{: style="text-align: center; font-size:0.7em;"}

위 그림은 MEMTO의 구조를 나타냅니다. 사실, Transformer를 활용했다는 것을 제외하면 기존의 Memory Autoencoder와 별다른 차이가 없습니다. 메모리를 초기화하기 위해 K-means를 사용한다는 차이점이 있지만 유사한 연구가 이미 선행 연구에서 많이 진행 되었습니다.

## Experiments 
![image](https://github.com/yunkio/SSL_Tutorial/assets/35906602/64b0a949-d6b0-487c-ac21-b8b73ccaaf89){: width="600"}{: .align-center} 

![image](https://github.com/yunkio/SSL_Tutorial/assets/35906602/693b6916-650e-43fc-9b7d-c5549c0cdb76){: width="400"}{: .align-center} 

![image](https://github.com/yunkio/SSL_Tutorial/assets/35906602/9ef56c5a-5c5e-4e8d-a8c7-d07ed83de1f6){: width="400"}{: .align-center} 

![image](https://github.com/yunkio/SSL_Tutorial/assets/35906602/68ee2f7e-0dd7-44c3-a479-7ad56dc0fe10){: width="500"}{: .align-center} 

Figure 2: Distribution of LSD values for normal and abnormal samples.
{: style="text-align: center; font-size:0.7em;"}

![image](https://github.com/yunkio/SSL_Tutorial/assets/35906602/0448d759-38ec-47b9-be00-56407055f570){: width="400"}{: .align-center} 

![image](https://github.com/yunkio/SSL_Tutorial/assets/35906602/fa43f4d8-640d-42a5-a4e1-5c2decbdd629){: width="500"}{: .align-center} 

Figure 3: Visualization of anomaly scores on SMD.
{: style="text-align: center; font-size:0.7em;"}

![image](https://github.com/yunkio/SSL_Tutorial/assets/35906602/dcf2cb1d-d1f1-42f1-a6d6-1a1eb6247fb9){: width="300"}{: .align-center} 

Figure 4: Performance across different numbers of memory items in the Gated memory module 
{: style="text-align: center; font-size:0.7em;"}

실험 결과 요약입니다. 별다른 특이사항은 없어 보입니다.