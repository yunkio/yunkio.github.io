---
date: 2023-02-02
title: "[Paper Review] A Docker Container Anomaly Monitoring System Based on Optimized Isolation Forest"
categories: 
  - Paper Review
tags: 
  - Time Series
toc: true  
toc_sticky: true 
---

# Paper contents

A Docker Container Anomaly Monitoring System Based on Optimized Isolation Forest

Z. Zou, Y. Xie, K. Huang, G. Xu, D. Feng and D. Long

IEEE Transactions on Cloud Computing (2022)

https://ieeexplore.ieee.org/abstract/document/8807263

## Motivation

클라우드의 사용량이 늘어남에 따라 보안 문제가 큰 이슈가 되고 있습니다. 이 논문은 이러한 문제를 모니터링을 통한 이상 탐지 시스템으로 해결하고자 합니다. 특히 **Isiolation Forest**를 활용하였으며, 기존 Isolation Forest가 무작위로 변수를 선택하던 방법을 취하는 반면 이 논문에서는 변수 선택에 가중치를 주는 방법을 제안합니다. 

## Contribution

주요 Contribution은 다음과 같습니다.

* 자동적으로 모니터링 기간 및 이상의 원인을 분석하는 다차원 이상 모니터링 시스템을 제안합니다.
* 각기 다른 자원 사용량에 따라 가중치를 다르게 설정하도록 최적화 된 Isolation Forest 알고리즘을 제안합니다.
* 제안 된 방법을 AWS 환경의 실제 & 시뮬레이트 된 데이터를 통해 검증합니다.

## Proposed Method

![image](https://user-images.githubusercontent.com/35906602/216261913-60083616-75f6-4b16-8659-a3e626fe7f49.png){: width="600"}{: .align-center} 

Figure 1. System architecture.
{: style="text-align: center; font-size:0.7em;"}

전체적인 시스템 구조는 크게 4가지 구성 요소로 구성됩니다 : *Monitoring agent*, *Monitoring data storage*, *Anomaly detection*, *Anomaly analysis*. 각 host machine마다 하나의 monitoring agent가 존재하며, monitoring data storage는 각 host에서 모니터링 데이터를 받게 됩니다. 최근 기간의 모니터링 데이터를 저장하며 데이터는 Anomaly detection module로 보내집니다. Isolation Forest를 기반으로 한 이상 평가 방법론을 통해 데이터를 평가하게 되고, Anomaly analysis 모듈은 이상으로 의심되는 데이터를 받아서 분석을 수행합니다.

### Monitoring Agent & Monitoring Data Storage

![image](https://user-images.githubusercontent.com/35906602/216263781-e4da3fd5-80e2-474f-bbd7-d1056ddd2a05.png){: width="600"}{: .align-center} 

Figure 2. Monitoring agent internal design.
{: style="text-align: center; font-size:0.7em;"}


Monitoring Agent 및 Monitoring Data Storage는 데이터를 어떻게 받아서 처리할 것 인지에 대한 내용으로, 주로 살펴보고자 하는 내용은 이상 탐지 관련 내용이므로 생략 하겠습니다. 결국 데이터를 잘 처리해서 Anomaly Detection 모듈로 보내게 됩니다.


### Anomaly Detection

Isolation Forest는 훌륭한 알고리즘이지만 컨테이너 환경에는 그대로 적용하기에는 무리가 있습니다. 컨테이너 모니터링에서는 크게 4가지 지표를 고려하게 됩니다 (CPU 사용량, 메모리 사용량, disk read and write rate, network speed). iForest 알고리즘에서는 위 4가지 지표들이 변수로 사용되어 데이터를 나누게 됩니다. 하지만 전통적인 iForest 방법은 각 변수가 선택 될 확률이 무작위입니다. 하지만 컨테이너 환경에서는 컨테이너의 어플리케이션에 따라 특정 지표에 더 의존적일 뿐만 아니라 더 민감합니다. 따라서 이러한 특징을 고려 할 필요가 있습니다.

이 문제를 해결하기 위해 본 논문에서는 최적화 방법을 제안하고 있습니다. 최적화 방법의 목표는 변수 선택에서 완전 무작위가 아니라 더 중요하게 고려 해야 할 변수를 선택하도록 가중치를 부여하는 것입니다. 따라서 이를 위한 self-learning 방법론이 제안합니다.  

![image](https://user-images.githubusercontent.com/35906602/216265864-a906cf49-bb90-4828-8f36-9a40eb8450cb.png){: width="300"}{: .align-center} 

$W_0$은 자원 사용량의 초기 가중치 값으로, 1로 사용됩니다. $\epsilon$은 자원 임계치이며, $N_i$는 $i$ 시점에서의 자원 사용 비율입니다. $p$는 자원 사용을 측정하기 위한 시간입니다. 만약 $x > 0$ 이라면, $f(x)=1$이 되고, 아니라면 $f(x) = 0$ 입니다. 만약 자원 사용량 값이 항상 0이라면 컨테이너는 해당 자원을 사용하지 않습니다. 따라서 이런 경우에는 가중치를 0으로 부여하게 됩니다. $M$ 값이 클수록 해당 컨테이너는 해당 자원에 더 치중 되어 있는 것을 의미합니다. 

편향 파라미터인 $M$은 각 자원 지표의 가중치로 사용됩니다. 기본적으로는 모든 자원 지표는 1의 가중치를 가지게 되며, 어느정도 기간동안 사용량을 살펴 볼 것인지를 정합니다. 본 논문에서는 10분을 사용합니다. 그 후에는 해당 기간동안 편향 파라미터 $M$을 구하고, $M$에 따라 가중치를 부여합니다. 자세한 과정은 다음과 같습니다.

![image](https://user-images.githubusercontent.com/35906602/216265236-7b4a2b1d-3b7e-45ab-9d3f-71f58ae39ce3.png){: width="400"}{: .align-center} 

Algorithm 1. Weighted Random Algorithm
{: style="text-align: center; font-size:0.7em;"}

![image](https://user-images.githubusercontent.com/35906602/216274974-fd3c2f28-ce90-4f37-b8c3-3fe36659f46d.png){: width="600"}{: .align-center} 

Figure 3. Isolation forest construction process
{: style="text-align: center; font-size:0.7em;"}

iForest 알고리즘을 통해서는 어떤 변수로 인해 이상이 발생 했는지는 알 수 없습니다. 따라서 이러한 문제를 해결하기 위해 다음과 같은 방법을 제안합니다.

1. Isolation tree를 구성 할 때, 만약 트리 분할 될 때 leaf node가 생성되었다면, 분할에서 사용된 변수들은 isolation feature라고 불립니다.
2. 각 데이터에 대해 isolation feature group $S(S_1, S_2, ...)$를 만듭니다. $S_i$는 $i$로 숫자가 매겨진 변수가 isolation feature로 사용 된 횟수를 의미합니다.

앞서서 정의된 방법에 따라 더 중요하게 봐야 할 변수는 더 높은 가중치를 가져 변수로 선택 될 확률이 높아지며, 이상의 원인일 확률도 높습니다. 

### Anomaly analysis

이상 분석의 경우 클라우드에 저장된 로그 등을 참고해서 원인을 분석하겠다는 내용입니다. 여기서는 생략하도록 하겠습니다.

## Experiment

![image](https://user-images.githubusercontent.com/35906602/216275758-dffa3960-a554-4842-9067-332a0f130f73.png)
{: width="600"}{: .align-center} 

Table 1. The Result COmparison of Anomaly Detection on Memcached and Web Search
{: style="text-align: center; font-size:0.7em;"}

실험 결과는 위와 같습니다. 자세한 내용은 논문을 참고해주세요.

## Conclusion

클라우드 사용량을 활용한 다변량 이상 탐지를 포함한 전체적인 시스템의 구조를 제안한 논문입니다. 이상 탐지 관점에서 참고할 부분은 iForest 부분입니다. 변수 선택법 부분에서 차이를 주었는데 제시한 실험 결과에서는 original과 성능 차이가 매우 크게 나고 있습니다. 더 살펴 볼 필요가 있을 것 같습니다.
