---
date: 2022-08-06
title: "[Paper Review] SimCLR: A Simple Framework for Contrastive Learning of Visual Representations"
categories: 
  - Paper Review
tags: 
  - Time Series
  - Representation Learning
  - Contrastive Learning
toc: true  
toc_sticky: true 
---
# Paper contents

A Simple Framework for Contrastive Learning of Visual Representations

Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. 

International conference on machine learning (2020)

http://proceedings.mlr.press/v119/chen20j.html

## 0. Abstract

이 논문은 **SimCLR**: a simple framework for contrastive learning of visual representation을 소개합니다. 최근에 제안된 대조적 자기지도 학습 알고리즘을 단순화 했습니다. 특히 대조적 학습의 어떤 요소가 표현*representation*을 잘 학습하도록 돕는지 체계적으로 연구했습니다.

* 데이터 증강은 매우 중요한 역할을 합니다.
* 표현과 대조 손실 사이에 비선형 변환을 도입하면 학습된 표현의 품질을 크게 향상됩니다.
* 지도 학습 방법론에 비해 배치 사이즈와 훈련 스텝에 더 많은 영향을 받습니다.

위 3가지 발견을 통해 기존의 반지도 혹은 자기지도 학습 방법들을 크게 뛰어넘는 성능을 보이는 방법론을 제안합니다. SimCLR를 통해 학습된 표현을 사용한 선형 분류 모델은 기존의 SOTA 자기지도 방법론에 비해 매우 커다란 성능의 향상을 가져왔습니다. 또한 매우 적은 비율(1%)의 라벨 데이터만을 사용해서 매우 좋은 성능을 보였습니다. 

## 1. Introduction

스스로 시각적 표현을 학습하는 것은 오래 된 문제였습니다. 이를 위한 기존의 대부분의 접근법은 *generative*, 혹은 *discriminative* 로 구분 가능합니다. 생성 접근 방식은 입력 공간에서의 픽셀들을 생성하거나 모델링 하는 것을 학습합니다. 하지만 픽셀 단계의 생성은 계산적으로 비효율적이며 표현 학습에 반드시 필요하지 않습니다. 분류 접근 방식은 지도학습과 비슷한 방법으로 목표 함수를 활용하여 표현을 학습합니다. 다만 이 방법은 라벨이 없는 데이터로 동작하기 위해서 pretext task가 요구되며, 이는 휴리스틱에 의존하게 되기 때문에 학습된 표현의 일반화 능력을 제한하게 됩니다.

이 논문에서는 성능이 좋을 뿐만 아니라 더 간단하고 특별한 구조가 필요하지 않은, 시각적 표현을 학습하는 간단한 대조 학습 프레임워크인 SimCLR를 제안합니다. 대조 표현 학습이 잘 작동하도록 하는 요소는 다음과 같습니다.

* 여러 데이터 증강 방법의 조합은 대조를 통한 효과적인 표현을 만드는데 매우 중요합니다. 더해서 비지도 기반의 대조 학습에서는 지도 학습에 비해 더 중요합니다.
* 표현과 대조 손실 사이에 학습 가능한 비선형 변형을 도입하는 것은 학습된 표현의 질을 향상 시킵니다.
* 정규화된 임베딩과 적절하게 조정된 temperature 파라미터는 대조 크로스 엔트로피 손실을 사용한 표현 학습에 도움이 됩니다.
* 대조 학습은 더 큰 배치 크기와 더 긴 훈련으로부터 지도학습에 비해 더 많은 효과를 얻습니다. 또한 더 깊고 넓은 네트워크 역시 도움이 됩니다.

## 2. Method

### 2.1 The Contrastive Learning Framework

![image](https://user-images.githubusercontent.com/35906602/183149065-a381fdc8-7b5f-4525-8bef-ed3c92c896a6.png){: width="400"}{: .align-center} 

Figure 1. A Simple Framework for Contrastive Learning of Visual Representations.
{: style="text-align: center; font-size:0.7em;"}

최근의 다른 대조 학습 방법론과 마찬가지로, SimCLR 역시 같은 데이터로부터 비롯된 서로 다른 증강 예제들 사이의 대조 손실을 활용해 표현 학습을 수행합니다. 다음 4가지의 주요한 요소가 있습니다.

* 확률적 데이터 증강 모듈이 사용됩니다. 주어진 데이터를 변환하여 상관관계가 있는 두 개의 뷰$\tilde{x}_i$,$\tilde{x}_j$를 생성하며, 이를 positive 쌍으로 간주합니다. 이 연구에서는 *random cropping*, *random color distortions*, *random Gaussian blur* 의 3가지 증강 기법을 활용합니다.
* 신경망 기반의 인코더 $f(\cdot)$ 을 통해 증강된 데이터로부터 표현 벡터를 추출해냅니다. 다양한 선택이 가능하며, 이 연구에서는 간단한 *ResNet*을 활용하였습니다. $h_i \in \mathbb{R}^d$가 *average pooling layer*의 결과라고 할 때, $h_i = f(\tilde{x}_i) = \text{ResNet}(\tilde{x}_i)$라고 나타낼 수 있습니다.
* 작은 신경망 *projection head* $g(\cdot)$이 표현을 대조 손실이 적용된 공간에 맵핑하기 위해 사용 되었습니다. $z_i = g(h_i) = W^{(2)}\sigma(W^{(1)}h_i)$를 얻기 위해 하나의 은닉층을 가진 MLP를 사용했으며, 여기서 $\sigma$는 ReLU를 의미합니다. 실험을 통해 $h_i$를 사용하는 것보다 $z_i$를 사용하는 것이 효과적임을 보였습니다.
* 대조 손실이 정의 되었습니다. positive 쌍 $\tilde{x}_i, \tilde{x}_j$를 포함하는 집합 $\tilde{x}_k$가 주어졌을 때, 대조 예측은 $\tilde{x}_k,{k \ne i}$에서부터 $\tilde{x}_i$가 주어지면 $\tilde{x}_j$를 찾는 것을 목적으로 합니다.

$N$개의 샘플을 포함한 미니배치를 무작위로 샘플링하여 미니배치로부터 비롯된 증강된 예시들을 활용해 대조 학습을 정의하였으며, 결과적으로 $2N$개의 데이터가 형성됩니다. Negative 예제는 따로 정의하지 않고 주어진 positive 쌍에서 나머지 $2(N-1)$개의 예제들을 Negative로 정의했습니다. $\ell_2$로 정규화 된 $u$와 $v$의 곱 (*코사인 유사도*)을 $\text{sim}(u,v) = \frac{u^\text{T}v}{\lVert u\rVert \lVert u\rVert}$로 정의할 때, 예제 $(i, j)$의 positive 쌍의 손실 함수는 다음과 같의 정의됩니다.

$$ \ell_{i,j} = -\log\frac{\exp(\text{sim}(z_i,z_j)/\tau)}{\sum^{2N}_{k=1}\mathbf{1}_{k\ne i}\exp(\text{sim}(z_i,z_k)/\tau)}$$

여기서 $\mathbf{1}_{[k \ne i]} \in \{0,1\}$은 $k \ne 1$일 때 1을 나타내는 지시 함수며, $\tau$는 *temperature* 파라미터 입니다. 최종 손실은 미니배치의 모든 positive 쌍들에 대해 계산되며, $(i, j)$ 혹은 $(j, i)$ 둘 다 계산됩니다. 이를 *NT-Xent*라고 부릅니다.

![image](https://user-images.githubusercontent.com/35906602/183153183-31527a51-49ca-44f2-bae8-d8efd0e0a8f4.png){: width="400"}{: .align-center} 

### 2.2 Training with Large Batch Size

메모리 뱅크와 같은 기존의 방법을 사용하지 않고 훈련 배치 크기 $N$을 256부터 8192까지 다양한 값으로 세팅하여 연구를 진행했습니다. 배치 사이즈가 8192일 경우 positive 쌍마다 16382개의 negative 샘플들을 얻게 됩니다. 배치가 큰 상황에서 안정적인 학습을 위해 SGD/Momentum 이 아닌 LARS optimizer를 사용했습니다.

### 2.3 Evaluation Protocol

대부분의 비지도 학습 연구와 마찬가지로 ImageNet ILSVRC-2012 데이터로 이루어졌습니다. 이에 더해 사전학습 된 결과를 전이 학습을 통해 여러 데이터셋에서 검증했습니다. 또 학습된 표현을 평가하기 위해 널리 쓰이는 방법과 마찬가지로 표현 학습 단계를 얼리고 선형 분류기를 통해 학습된 표현에 대한 성능을 평가했습니다. 

## 3. Data Augmentation for Contrastive Representation Learning

![image](https://user-images.githubusercontent.com/35906602/183237510-91e6402d-39bc-4915-9268-d99a1210216d.png){: width="500"}{: .align-center} 

Figure 2. Solid rectangles are images, dashed rectangles are random crops. By randomly cropping images, we sample contrastive prediction tasks that include global to local view ($B \rightarrow A$) or adjacent view ($D \rightarrow C$) prediction.
{: style="text-align: center; font-size:0.7em;"}

**Data augmentation defines predictive tasks.** 데이터 증강 기법은 대조 학습에 널리 사용되긴 했지만 필수적인 요소로 여겨지지는 않았습니다. 대부분의 경우 모델 아키텍쳐에 변형을 가해서 대조 학습을 수행했습니다. 본 연구에서는 단순한 무작위 잘라내기만으로도 복잡성을 피할 수 있음을 보였으며, 여러 증강 기법의 조합으로 더 광범위한 대조 예측 수행이 가능합니다.

### 3.1 Composition of data augmentation operations is crucial for learning good representation
![image](https://user-images.githubusercontent.com/35906602/183237244-0cab1d72-b4ac-47f0-806a-d642b2ee54e0.png){: width="700"}{: .align-center} 

Figure 3. Illustrations of the studied data augmentation operators. Each augmentation can transform data stochastically with some internal parameters (e.g. rotation degree, noise level). Note that we only test these operators in ablation, the augmentation policy used to train our models only includes random crop (with flip and resize), color distortion, and Gaussian blur.
{: style="text-align: center; font-size:0.7em;"}

데이터 증강 기법에 대한 체계적인 연구를 통해 핵심적인 증강 기법 몇 가지를 식별 했습니다. 잘라내기, 리사이즈, 돌리기 등의 **공간적 변환**, 색상 왜곡이나 가우시안 블러, 소벨 필터링과 같은 **모습 변환** 등이 있습니다. 이러한 각각의 증강 기법을 positive 쌍에 적용하여 실험을 진행하였습니다. 

![image](https://user-images.githubusercontent.com/35906602/183238272-61792577-6aa3-4175-b99f-c41be03799f0.png){: width="700"}{: .align-center} 

Figure 4. Linear evaluation (ImageNet top-1 accuracy) under individual or composition of data augmentations, applied only to one branch. For all columns but the last, diagonal entries correspond to single transformation, and off-diagonals correspond to composition of two transformations (applied sequentially). The last column reflects the average over the row.
{: style="text-align: center; font-size:0.7em;"}

위 표는 각각의 증강 기법을 활용해 학습된 표현을 이용해 선형 분류기에 넣은 결과입니다. 실험 결과 하나의 변환을 양 쪽에 적용하는 것은 좋은 결과를 보이지 못 함을 확인했습니다. 각기 다른 증강 기법을 적용 했을 때 positive 쌍을 찾는 것은 더 어렵지만 그렇게 학습된 표현의 질은 크게 향상됩니다. 특히 색 왜곡과 잘라내기를 같이 사용했을 때의 성능이 뛰어났습니다.

### 3.2 Contrastive learning needs stronger data augmentation than supervised learning

![image](https://user-images.githubusercontent.com/35906602/183238565-914259c5-8650-480f-b722-bd3cbbdd3f6d.png){: width="500"}{: .align-center} 

Table 1. Top-1 accuracy of unsupervised ResNet-50 using linear evaluation and supervised ResNet-50, under varied color distortion strength (see Appendix A) and other data transformations. Strength 1 (+Blur) is our default data augmentation policy.
{: style="text-align: center; font-size:0.7em;"}

색 왜곡을 강하게 가해서 증강 시킬수록 학습된 표현을 통한 선형 분류의 성능이 향상 되었습니다. 위의 표와 같이 기존 지도 학습 방법론에서 사용되던 확률적 증강 기법의 조합인 *AutoAug*는 비지도 표현 학습에서는 의미있는 성능을 보여주지 못 했습니다. 반대로 지도 학습에서는 색 왜곡을 강하게 가한다고 하더라도 성능에 도움이 되지 않았습니다. 이처럼 증강 기법이 성능에 도움이 되는 것은 분명하지만 지도 학습에서의 증강 기법과는 다른 특징을 보이고 있습니다.

## 4. Architectures for Encoder and Head

### 4.1 Unsupervised contrastive learning benefits (more) from bigger models

![image](https://user-images.githubusercontent.com/35906602/183238812-1abe306f-2c0a-484f-aefe-009d557ed7f6.png){: width="400"}{: .align-center} 

Figure 5. Linear evaluation of models with varied depth and width. Models in blue dots are ours trained for 100 epochs, models in red stars are ours trained for 1000 epochs, and models in green crosses are supervised ResNets trained for 90 epochs
{: style="text-align: center; font-size:0.7em;"}

인코더와 헤드에서 사용되는 모델의 깊이와 넓이를 크게 하는 것은 성능에 도움이 됩니다. 특히 크면 클수록 지도 학습과의 성능 차이가 점점 줄어드는 것에서 알 수 있듯이, 모델의 크기에 지도 학습보다 더 큰 영향을 받습니다. 

### 4.2 A nonlinear projection head improves the representation quality of the layer before it

![image](https://user-images.githubusercontent.com/35906602/183238565-914259c5-8650-480f-b722-bd3cbbdd3f6d.png){: width="500"}{: .align-center} 

Table 2. Accuracy of training additional MLPs on different representations to predict the transformation applied. Other than crop and color augmentation, we additionally and independently add rotation (one of {0$^◦$ , 90$^◦$ , 180$^◦$ , 270$^◦$}), Gaussian noise, and Sobel filtering transformation during the pretraining for the last three rows. Both $h$ and $g(h)$ are of the same dimensionality, i.e. 2048.
{: style="text-align: center; font-size:0.7em;"}

위에서 $g(h)$로 정의한 *projection head*에 대한 중요성을 다양한 실험을 통해 입증했습니다. *identity mapping*, *linear projection* 등의 여러 방법을 시도했으나 하나의 은닉층으로 구성된 *nonlinear projection*이 가장 좋은 성능을 보였습니다. 하지만 모든 경우에 *projection head* 이전의 층인 $h$가 가장 좋은 성능을 보였습니다.
이런 현상은 *nonlinear projection*에서 정보의 손실이 발생하기 때문이라고 추측됩니다. 특히 $z = g(h)$에서는 데이터 변환에 상관없이 학습되기 때문에 해당 정보가 손실됩니다. 따라서 $g$에서는 물체의 색 혹은 구성과 같은 정보를 잃게 됩니다. 따라서 비선형 변환 $g(\cdot)$을 활용함으로써 더 많은 정보가 $h$에 유지될 수 있습니다. 

## 5. Loss Functions and Batch Size

### 5.1 Normalized cross entropy loss with adjustable temperature works better than alternatives

![image](https://user-images.githubusercontent.com/35906602/183240290-40c83c38-892b-4b8b-9a51-f75cb73dcda1.png){: width="700"}{: .align-center} 

Table 3. Negative loss functions and their gradients. All input vectors, i.e. $u$, $v^+$, $v^−$, are $\ell^2$ normalized. NT-Xent is an abbreviation for “Normalized Temperature-scaled Cross Entropy”. Different loss functions impose different weightings of positive and negative examples.
{: style="text-align: center; font-size:0.7em;"}

대조학습에서 많이 쓰이는 손실 함수인 *logisstic loss*, *margin loss*가 아닌 **NT-Xent loss** 를 사용했습니다. *Temperature*와 함께 사용되는 $\ell^2$ 정규화가 효과적으로 다른 예들에 가중치를 매기며, temperature는 모델이 negative로부터 잘 배울 수 있도록 돕습니다. 또한 크로스 엔트로피와는 다르게, 다른 목적 함수들은 negative들의 상대적인 난이도에 가중치를 주지 못 합니다. 

![image](https://user-images.githubusercontent.com/35906602/183240515-b27cbe26-9fd9-435f-b2bf-4eaddaa3cca1.png){: width="500"}{: .align-center} 
Table 4. Linear evaluation (top-1) for models trained with different loss functions. “sh” means using semi-hard negative mining.
{: style="text-align: center; font-size:0.7em;"}

![image](https://user-images.githubusercontent.com/35906602/183240571-31ca7c77-1b89-42c0-8c7a-6c0ed481eb5c.png){: width="500"}{: .align-center} 
Table 5. Linear evaluation for models trained with different choices of $\ell^2$ norm and temperature $\tau$ for NT-Xent loss. The contrastive distribution is over 4096 examples.
{: style="text-align: center; font-size:0.7em;"}

이 논문에서 제안한 NT-Xent loss의 효과가 제일 좋았으며, 또한 $\ell^2$과 $\tau$의 효과도 검증했습니다.

### 5.2. Contrastive learning benefits (more) from larger batch sizes and longer training

![image](https://user-images.githubusercontent.com/35906602/183241044-70b35d47-48e7-4bd9-a2c3-e9d42df7224f.png){: width="600"}{: .align-center} 

Figure 6. Linear evaluation models (ResNet-50) trained with different batch size and epochs. Each bar is a single run from scratch.
{: style="text-align: center; font-size:0.7em;"}

배치의 크기가 크면 클수록 성능이 향상되며, 특히 훈련 에포크의 수가 적을수록 더 그렇습니다. 하지만 훈련 에포크가 늘어날수력 이 차이는 줄어들거나 사라집니다. 지도 학습과는 다르게 대조 학습은 배치 사이즈가 클수록 negative 예제가 늘어나므로 수렴에 도움이 됩니다. 

## 6. Comparison with State-for-the-art
![image](https://user-images.githubusercontent.com/35906602/183241344-fa51d346-e05d-45e4-8c3d-bc9c663d96c7.png){: width="500"}{: .align-center} 
Table 6. ImageNet accuracies of linear classifiers trained on representations learned with different self-supervised methods.
{: style="text-align: center; font-size:0.7em;"}

![image](https://user-images.githubusercontent.com/35906602/183241523-af65dd2c-7e8e-4771-81dc-c013a2f0af77.png){: width="500"}{: .align-center} 
Table 7. ImageNet accuracy of models trained with few labels.
{: style="text-align: center; font-size:0.7em;"}

![image](https://user-images.githubusercontent.com/35906602/183241147-0f87439b-14cc-4b01-9f63-6ee45c2baae4.png){: width="750"}{: .align-center} 
Table 8. Comparison of transfer learning performance of our self-supervised approach with supervised baselines across 12 natural image classification datasets, for ResNet-50 (4$\times$) models pretrained on ImageNet. Results not significantly worse than the best ($p > 0.05$, permutation test) are shown in bold. 
{: style="text-align: center; font-size:0.7em;"}

3개의 다른 은닉층 넓이를 활용한 ResNet-50을 활용하여 실험을 진행했습니다. 성능은 위와 같습니다.

## 7. Conclusion
이 논문에서는 시각적 표현 학습을 위한 대조 학습 프레임워크를 제안합니다. 각각의 요소에 대해 체계적인 연구를 통해 효과를 검증했습니다. 이러한 발견들을 조합하여 자기 지도 학습, 반 지도 학습, 그리고 전이 학습의 이전 방법론들을 뛰어넘는 성과를 얻었습니다. 
 