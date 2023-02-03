---
date: 2023-02-02
title: "[Paper Review] Ensemble learning based predictive framework for virtual machine resource request prediction"
categories: 
  - Paper Review
tags: 
  - Time Series
toc: true  
toc_sticky: true 
---

# Paper contents

Ensemble learning based predictive framework for virtual machine resource request prediction 

Kumar, Jitendra, Ashutosh Kumar Singh, and Rajkumar Buyya

Neurocomputing (2020)

https://www.sciencedirect.com/science/article/pii/S0925231220301892

## Motivation

최근 클라우드 서비스가 활성화 되면서 많은 양의 컴퓨팅 자원들이 제공되고 있으며, 탄소 소모량을 최소화 하는 것이 이슈가 되고 있습니다. 따라서 서비스 공급자는 서비스의 품질을 낮추지 않고 자원 소모량을 최적화 하는 것이 매우 중요한 문제입니다. 이러한 맥락에서 앙상블 러닝을 통한 자원 소모량 예측을 제안합니다. 이때 보팅을 위한 가중치 설정에는 블랙홀 이론에서 영감을 받은 메타휴리스틱 알고리즘이 사용됩니다.

## Method

![image](https://user-images.githubusercontent.com/35906602/216503830-b3c492c7-72b5-48b8-a423-8ef8ae2c210a.png){: width="500"}{: .align-center} 

Figure 1. Block diagram of proposed predictive framework.
{: style="text-align: center; font-size:0.7em;"}

![image](https://user-images.githubusercontent.com/35906602/216504103-ffcbf625-bdf5-4fa8-8b49-5ead9ab5d186.png){: width="500"}{: .align-center} 

Figure 2. Detailed workflow of proposed predictive framework.
{: style="text-align: center; font-size:0.7em;"}

크게 'Data Analysis', 'Expert Learning', 'Voting Engine'의 세 가지 핵심 모듈이 존재합니다. 

### Ensemble expert learning using ELM

여러 예측 모델을 사용해서 미래의 값을 예측하게 되며, 앙상블의 최종 출력은 voting engine을 통해 결정됩니다. 여기서는 $k$개의 다층 신경망을 base expert로 사용합니다. 크게 특별한 내용은 없습니다.

### Expert architecture selection

![image](https://user-images.githubusercontent.com/35906602/216504846-a1150220-645a-48c3-bcf9-41344af6d603.png){: width="400"}{: .align-center} 

신경망의 하이퍼 파라미터를 어떻게 설정 할 것인지에 대한 내용입니다. 단순히 직관으로 정하는 것이 아니라 여러 알고리즘을 통해 정하게 됩니다.

### Weight allocation scheme

![image](https://user-images.githubusercontent.com/35906602/216505677-f9c17f3a-6b86-44c6-bd5b-3a2fbd33b792.png){: width="400"}{: .align-center} 


본 모델은 앙상블을 통해 결과를 계산하는데, 여기서는 각 모델의 가중치를 어떻게 부여 할 것이냐에 대한 내용을 다루게 됩니다. 여기서 블랙홀 현상에서 영감을 받은 최적화 알고리즘이 사용됩니다. BhOA (The blackhole optimization algorithm)은 population based method로, 탐생 공간의 최적 해에 도달 할 때까지 반복하게 됩니다. 

## Experiment

실험 결과는 다음과 같습니다.
![image](https://user-images.githubusercontent.com/35906602/216505885-b2c69427-13d6-4843-9e46-8c2bee084abd.png){: width="500"}{: .align-center} 



![image](https://user-images.githubusercontent.com/35906602/216505750-5a33ab96-9525-4cd6-8b78-b5bcef8e9a39.png){: width="500"}{: .align-center} 

여기서 **RelMAE** 는 단순히 이전 값을 그대로 사용 한 것에 비해 상대적으로 오류가 얼마나 작은 지를 나타내는 지표입니다. 실험에 특별한 점은 없으므로 설명은 생략하겠습니다.

## Conclusion

본 논문에서는 클라우드 시스템에서 계산량을 최소화하며 자원 소모량을 예측하는 모델을 제안했습니다. 특히 복잡한 구조의 딥러닝을 사용하기 보다는 다층 신경망 구조의 네트워크 여러개를 앙상블하는 방법을 제안했으며, 그 과정에서 신경망의 하이퍼 파라미터를 어떻게 정의 할 것인지, 그리고 각 신경망의 가중치를 어떻게 부여할 것인지 등의 내용을 제시하고 있습니다. 다만 저자는 최근의 sota 모델과 비교하여 좋은 성능을 보여주고 있다고 주장하지만, 비교 대상 모델은 sota라고 부를 수 있는 모델이 없다는 점이 아쉬운 점입니다.

