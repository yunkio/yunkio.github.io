---
date: 2023-08-16
title: "[Paper Review] Residual attention convolutional autoencoder for feature learning and fault detection in nonlinear industrial processes"
categories: 
  - Paper Review
tags: 
  - Anomaly Detection
toc: true  
toc_sticky: true 
---
## Reference

Sparse one-dimensional convolutional neural network-based feature learning for fault detection and diagnosis in multivariable manufacturing processes

Yu, J., Zhang, C., & Wang, S.

Neural Computing and Applications (2022)

https://link.springer.com/article/10.1007/s00521-021-06575-6

## Motivation

산업 현장에서, 특히 비선형적일 경우 Feature Engineering과 Fault Detection은 항상 중요한 도전과제입니다. 결함을 빠르고 정확하게 감지할 수 있는 능력은 잠재적인 피해를 방지할 수 있습니다. 전통적인 방법들은 내재된 복잡성을 처리하는 데 뛰어나지 않으므로, 더 발전된 모델이 필요합니다.

## Proposed Method

![image](https://github.com/yunkio/SVM_tutorial/assets/35906602/6ca2823c-8973-45db-a5c6-23a2ec5f584c){: width="700"}{: .align-center} 

Figure 1. Network Structure of RACAE.
{: style="text-align: center; font-size:0.7em;"}

논문에서는 **Residual Attention Convolutional AutoEncoder (RACAE)** 를 제안합니다. 이 방법은 Residual Attention 블록과 인코더와 디코더 구성 요소 사이에 전략적으로 배치된 Attention 층을 활용합니다.

* Feature Extraction : 이 구조의 주요 목표는 특징 추출의 성능을 높이는 것입니다. Attention 구조를 활용하여 모델이 데이터의 더 관련성 있는 부분에 중점을 두도록 하여 가장 중요한 특징을 잡아내도록 합니다.


* Training Efficiency : 이 구조의 또 다른 중요한 장점은 훈련 효율성입니다. Residual Connection을 활용하여 네트워크는 더 효율적으로 학습되며, Vanishing Gradient 문제를 방지합니다.

* Evaluation Metric : 제안된 방법의 성능을 측정하기 위해 T2 및 SPE가 사용됩니다. 이러한 지표를 통해 오토인코더에 의해 재구성된 데이터와 원래 입력된 데이터 간의 차이를 계산합니다. 

![image](https://github.com/yunkio/SVM_tutorial/assets/35906602/3d3fb0e1-fc36-4309-bae4-9b31edd4bc22){: width="700"}{: .align-center}

Figure 2. Process Monitoring Based on RACAE
{: style="text-align: center; font-size:0.7em;"}

![image](https://github.com/yunkio/SVM_tutorial/assets/35906602/bbf99f03-9227-460d-8f00-1210d03439a5){: width="600"}{: .align-center}

학습 및 추론 과정은 위와 같습니다.

## Experiments 

본 논문에서는 총 3가지 경우에 대해 실험을 실시합니다.

### Numerical Case
제안 된 방법론을 테스트하기 위해 시뮬레이션 된 데이터입니다.

![image](https://github.com/yunkio/SVM_tutorial/assets/35906602/68eff068-ed12-4743-b61a-c4acfc644e88){: width="500"}{: .align-center}

### Tennessee Eastman Process (TEP) 

공정 시스템에서 널리 사용되는 벤치마크입니다. 현실의 적용 가능성과 성능에 대한 정보를 제공할 수 있습니다.

![image](https://github.com/yunkio/SVM_tutorial/assets/35906602/ea889e07-f414-49e0-a4e6-b3a2479205e3){: width="500"}{: .align-center}

![image](https://github.com/yunkio/SVM_tutorial/assets/35906602/f3f494e8-f920-47f2-b49a-f5140622dc81){: width="500"}{: .align-center}

### Continuous Stirred-Tank Reactor (CSTR)

다른 현실의 시나리오로, 지속적으로 변화하는 동적 환경에서 제안된 방법론이 어떻게 수행되는지에 대한 통찰력을 제공할 수 있습니다.

![image](https://github.com/yunkio/SVM_tutorial/assets/35906602/f4ac49c6-42db-4cfb-94a7-9211e1a4e6fb){: width="500"}{: .align-center}

