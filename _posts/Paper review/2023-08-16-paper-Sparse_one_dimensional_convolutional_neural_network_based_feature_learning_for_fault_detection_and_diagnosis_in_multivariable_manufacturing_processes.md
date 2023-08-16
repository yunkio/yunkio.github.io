---
date: 2023-08-16
title: "[Paper Review] Sparse one-dimensional convolutional neural network-based feature learning for fault detection and diagnosis in multivariable manufacturing processes"
categories: 
  - Paper Review
tags: 
  - Fault Diagnosis
toc: true  
toc_sticky: true 
---

## Reference

Sparse one-dimensional convolutional neural network-based feature learning for fault detection and diagnosis in multivariable manufacturing processes

Yu, J., Zhang, C., & Wang, S.

Neural Computing and Applications (2022)

https://link.springer.com/article/10.1007/s00521-021-06575-6

## Introduction

산업 프로세스에서 데이터의 크기와 복잡성은 매우 빠르게 증가하고 있고, 이를 위해 딥러닝 모델이 주목받고 있습니다. SAE *stacked autoencoder*, DBN *deep belief network*, CNN *convolutional neural network*는 고장 진단 분야에서 널리 사용되고 있습니다. 하지만 PCA, ICA, PLS와 같은 고전적인 모델은 고차원의 신호 데이터에서 효과적인 특징을 추출하는데 한계가 있습니다. 단변량의 프로세스 데이터를 다루기 위한 효과적인 모델이 필요합니다.

## Proposed Method

![image](https://github.com/ML4ITS/mtad-gat-pytorch/assets/35906602/b8965ce4-b505-496f-acd3-344b84c2b592){: width="600"}{: .align-center} 

Figure 1. Network Structure of S1-DCNN.
{: style="text-align: center; font-size:0.7em;"}
![image](https://github.com/ML4ITS/mtad-gat-pytorch/assets/35906602/db62d218-ac12-4655-9225-19c6e57f3cee){: width="500"}{: .align-center} 

Figure 2. Construction of a sparse layer.
{: style="text-align: center; font-size:0.7em;"}

S1-DCNN 모델은 단변량 프로세스 신호 데이터를 처리하기 위한 새로운 신경망 모델입니다. 기존의 CNN 모델이 2차원의 이미지 데이터를 처리하기 위해 설계된 것과 달리, 이 모델은 단변량의 입력층과 단변량의 커널 필터를 가지고 단변량의 신호를 처리하기 위해 설계 되었습니다.

이 모델은 라벨이 있는 데이터를 통해 지도학습으로 학습됩니다. 데이터셋을 모델에 입력으로 집어넣기 위해, 먼저 일반화를 진행합니다. 

![image](https://github.com/ML4ITS/mtad-gat-pytorch/assets/35906602/deaefbcf-0754-45cc-ba0f-fe678dcf1858){: width="500"}{: .align-center} 

Figure 3. The procedure of the S1-DCNN-based method for process fault detection and diagnosis
{: style="text-align: center; font-size:0.7em;"}

또한 이 모델의 핵심적인 기여는 특징을 선택하는 능력에 있습니다. 잘 설계된 sparse layer를 통해서 중복되는 표현을 제거하고, 표현이 데이터를 더 잘 반영하도록 합니다. 특히 분류에 필요가 없는 특징의 중복을 잘 제거합니다. 더해서 이러한 특징 추출을 활용하여 특징의 시각화가 가능해지며, 데이터를 각 층에서 해석하는 능력을 모델에 부여합니다.

## Experiments 
![image](https://github.com/ML4ITS/mtad-gat-pytorch/assets/35906602/4dc062ef-e253-4705-91c5-c0140d4720c6){: width="600"}{: .align-center} 

![image](https://github.com/ML4ITS/mtad-gat-pytorch/assets/35906602/9a1990a4-86d9-4016-a1fc-a10567ad5b33){: width="600"}{: .align-center} 

![image](https://github.com/ML4ITS/mtad-gat-pytorch/assets/35906602/541ce500-7778-4fdb-ab3d-5eff9e3fbf93){: width="500"}{: .align-center} 

S1-DCNN은 벤치마크 데이터인 TEP *Tennessee Eastman Process*를 통해 검증 되었습니다. TEP는 전형적인 비선형 화학 프로세스로, 복잡한 변수와 강한 상호작용을 가지고 있습니다. 이 데이터는 주로 FDD *Fault Detection and Diagnosis*에 활용됩니다. 

![image](https://github.com/ML4ITS/mtad-gat-pytorch/assets/35906602/a6a827ca-63e1-4c62-b527-45e10a58286a){: width="700"}{: .align-center} 

Figure 4. Feature visualization of each layer in S1-DCNN
{: style="text-align: center; font-size:0.7em;"}

![image](https://github.com/ML4ITS/mtad-gat-pytorch/assets/35906602/eef4db18-a032-4e37-8622-c11dbec7cad1)Figure 5. Network visualization of each layer in S1-DCNN on case 1: a normal data, b fault #4, c fault #13
{: style="text-align: center; font-size:0.7em;"}


성능이 뛰어날뿐만 아니라 시각화를 통해 인사이트도 제공할 수 있습니다.


