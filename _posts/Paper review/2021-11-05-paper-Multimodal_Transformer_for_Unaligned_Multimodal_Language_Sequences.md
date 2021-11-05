---
date: 2021-11-05
title: "[Paper Review] Multimodal Transformer for Unaligned Multimodal Language Sequences"
categories: 
  - Paper Review
tags: 
  - Transformer
  - Multimodal
  - NLP
  - 논문 리뷰
toc: true  
toc_sticky: true 
---
# Paper contents

Multimodal Transformer for Unaligned Multimodal Language Sequences

Yao-Hung Hubert Tsai, Shaojie Bai, Paul Pu Liang, J. Zico Kolter, Louis-Philippe Morency, Ruslan Salakhutdinov

ACL, 2019.

https://arxiv.org/abs/1906.00295

## 0. Abstract

인간의 언어는 자연어, 음향, 얼굴 제스쳐 등이 혼합되어 있는 *multimodal* 입니다. 이러한 언어의 시계열 데이터를 모델링하는 데에는 두 개의 주요한 과제가 있습니다. 첫 번째로 각 modality에서 샘플링 주기가 달라 데이터의 길이가 맞지 않아 데이터가 제대로 정렬되지  않습니다 **non-alignment**.  두 번째로 modality들 사이의 **장기 의존성** 문제가 발생합니다. 이러한 문제를 해결하기 위해 **Multimodal Transformer** *MulT*를 도입했습니다. 데이터를 명시적으로 정렬하지 않고 일반적으로 해결 가능하도록 하는 end-to-end 모델입니다. 이 모델의 핵심은 쌍방향성 교차 모델 어텐션 *directional pairwise crossmodal attention* 으로, 각 타임 스텝에서 여러 모델의 시퀀스 간의 상호작용에 집중합니다. 



## 1. Introduction

인간의 언어는 단어의 의미뿐만 아니라 시각적, 음향적 modality들을 포함하고 있습니다. 이러한 modality들은 많은 정보를 담고 있지만 서로 이질적이여서 분석이 어렵습니다. 가령 각 데이터의 수집 빈도가 다르기 때문에 이 데이터들을 올바르게 맵핑하는 것이 쉽지 않습니다. 혹은 찡그린 얼굴은 지금이 아닌 과거에 내뱉은 비관적인 말과 연결지어져야 합니다. 즉 multimodal 언어 시퀀스는 정렬되지 않은 특성을 나타내며 여러 modality 사이의 장기 의존성을 추론해내야 합니다. 

![image](https://user-images.githubusercontent.com/35906602/140474799-8445cb21-57b4-4619-a760-2ce5666cd6b9.png){: width="600"}{: .align-center} 

Figure 1. Alignment 예시
{: style="text-align: center; font-size:0.7em;"}

이러한 문제를 해결하기 위해 이 논문에서는 **Multimodal Transformer** *MulT*를 도입하였습니다. 기존 Transformer Network를 정렬되지 않은 multimodal stream에서 직접 representation을 추출하도록 확장시킨 end-to-end 모델입니다. 핵심은 **crossmodal attention module**로, 전체 발화의 규모에서 여러 modality간의 상호 작용에 집중합니다. 보통의 경우 단어 단위로 수동으로 modality들을 alignment를 해주어서 학습을 진행합니다. 이러한 방법은 사람의 노력이 들어가고, 도메인 지식이 필요할 뿐만 아니라 시간차가 있는 여러 modal들의 상호작용을 반영하기 힘듭니다.  이 논문에서 제안한 방법은 별도의 alignment가 필요하지 않으며, 피쳐 엔지니어링도 필요하지 않습니다. 좋은 성능을 거둬 SOTA를 달성했습니다.  

## 2. Proposed Method

![image](https://user-images.githubusercontent.com/35906602/140474880-ebb06134-5f16-45cd-ad12-30223283fc18.png){: width="600"}{: .align-center} 

Figure 2. MulT 아키텍쳐
{: style="text-align: center; font-size:0.7em;"}

각 *crossmodal transformer*는 다른 *source modality*의 낮은 레벨의 피쳐를 활용하여 *target modality*를 반복적으로 강화하며, 이 과정은 두 *modality*들의 피쳐 간의 어텐션을 학습함으로써 이루어집니다. **MulT** 아키텍쳐는 이렇게 *modality*들의 모든 쌍을 학습합니다.  그 후에는 이렇게 합쳐진 피쳐를 학습하는 transformer로 이루어집니다.

### 2.1 Crossmodal Attention

두 *modality*를 $\alpha$와 $\beta$라고 하고, 각 *modality*로부터 나온 시퀀스를 $X_\alpha \in \mathbb{R}^{T_\alpha\times d_\alpha}$, $X_\beta \in \mathbb{R}^{T_\beta \times d_\beta}$ 라고 하겠습니다. $T_{(.)}$와 $d_{(.)}$는 각각 시퀀스의 길이와 피쳐 차원을 의미합니다. 이 논문에서 각 *modality*들은 얼굴 표정, 음의 높낮이 등으로 서로 매우 이질적입니다. 여기서는 *modality* 간의 latent adaptation을 활용해 여러 *modal*간의 정보를 합치는 것이 목적입니다.

여기서는 *Query*를 $Q_\alpha = X_\alpha W_{Q_\alpha}$, *Key*를 $K_\beta = X_\beta W_{K_\beta}$, *Values*를 $V_\beta = X_\beta W_{V_\beta}$로 정의합니다. *modal* $\beta$ 에서부터 *modal* $\alpha$로 향하는 *crossmodal attention* $Y_\alpha$는 다음과 같습니다.

$$\begin{aligned}
Y_\alpha &= \text{CM}_{\beta \rightarrow \alpha}(X_\alpha, X_\beta) \\
&= \text{softmax}\left( \frac{Q_\alpha K_\beta^\text{T}}{\sqrt{d_k}}\right)V_\beta \\
&= \text{softmax}\left(\frac{X_\alpha W_{Q_\alpha}W_{K_\beta}^\text{T}X_\beta^\text{T}}{\sqrt{d_k}}\right)X_\beta W_{V_\beta}
\end{aligned}$$

![image](https://user-images.githubusercontent.com/35906602/140479377-42e63625-7aa6-403e-94f8-c58a4143ddba.png){: width="600"}{: .align-center} 

Figure 3. Crossmodal attention
{: style="text-align: center; font-size:0.7em;"}


기존의 Transformer에서는 Encoder Output과 Decoder Input의 연관성을 반영한 learned representation을 산출하게 되는데, 여기서는 이와 비슷하게 Source Modality $\beta$와 Target Modality $\alpha$의 연관성을 반영한 fused representation을 산출하게 됩니다. 

각 Modality에서 Linear Transformation을 통해 *Query*, *Key*, *Value*를 얻어내고, $\alpha$의 *Key*와 $\beta$의 *Query*의 *Scaled Dot-Product Attention Score*를 얻은 뒤, 이 *Attention Score*를 가중치로 활용하여 $\beta$의 *Value*의 가중합을 얻습니다. 이것이 바로 $\beta$에서 $\alpha$로 향하는 *Crossmodal Attention*이 됩니다. 낮은 레벨의 피쳐에서 이를 수행함으로써 각 *modality*의 낮은 레벨의 정보를 얻을 수 있습니다.

![image](https://user-images.githubusercontent.com/35906602/140480222-ec34be81-2b25-4042-8c89-e556eea7dc8d.png){: width="400"}{: .align-center} 

Figure 4. Crossmodal Transformer
{: style="text-align: center; font-size:0.7em;"}

이러한 작업을 모든 *modality*의 쌍에 대해서 각각 수행합니다. *Crossmodal Transformer*는 *Crossmodal Attention* 블럭들이 여러개 모인 모델이 됩니다.

### 2.2 Overall Architecture

#### Temporal Convolutions

각 *Modality*들의 시퀀스의 길이가 각각 다르기 때문에 Transformer에 Input으로 넣기 위해서 Conv1D를 수행하게 됩니다.

$$\hat{X}_{\{L,V,A\}} = \text{Conv1D}(X_{\{L,V,A\}},k_{\{L, V , A\}}) \in \mathbb{R}^{T_{\{L,V,A\}}\times d}$$

$k_{\{L,V,A\}}$는 커널의 사이즈를 의미하며 $d$는 차원입니다. 이렇게 만들어진 시퀀스는 시퀀스의 지역적인 구조를 가지고 있을 것이라고 기대할 수 있습니다.

#### Positional Embedding

각 시퀀스들이 시간적 정보를 지닐 수 있도록 **positional embedding**을 활용했습니다.

$$Z^\text{[0]}_{\{L,V,A\}}= \hat{X}_{\{L,V,A\}} + \text{PE}(T_{\{L,V,A\}},d)$$

#### Crossmodal Transformers

위에서 설명한 *crossmodal attention block* 들을 활용해 Figure 4의 **Crossmodal Transformer**를 구성하게 됩니다. 각 *crossmodal transformer*들은 $D$개의 *crossmodal attention block* 레이어로 구성됩니다. 

![image](https://user-images.githubusercontent.com/35906602/140482804-1aa5f1ec-2a2b-40fa-a199-5ae77b56f399.png){: width="500"}{: .align-center} 

Figure 5. example of visualizing alignment using attention matrix from modality $\beta$ to $\alpha$
{: style="text-align: center; font-size:0.7em;"}

각 *modality*들은 *multi-head crossmodal attention module*을 통해 낮은 레벨의 외부 정보를 활용해서 계속해서 업데이트 됩니다. *crossmodal attention block*의 모든 레벨에서 *source modality*로부터 나온 낮은 레벨의 신호들은 *target modality*와 상호작용하는 *Key/Value* 셋이 됩니다. 이를 통해 *transformer*는 *modality* 사이의 의미있는 원소를 학습하게 됩니다. 최족적으로 MulT는 모든 *modality* 쌍의 상호작용을 모델링합니다. 

#### Self-Attention Transformers and Prediction

마지막 단계로 같은 *target modality*를 지닌 모든 *crossmodal transformer*의 각 output을 이어붙인 뒤 sequence model 등에 넣어서 예측을 수행합니다. 이 논문에서는 *self-attention transformer*가 활용됐습니다. 

### 2.3 Discussion about Attention & Alignment

**MulT**는 따로 Alignment 작업을 수행하지 않고 *attention block*에 의존합니다. 즉 *modality* 사이에서 강한 신호 혹은 관련깊은 정보들이 서로 연결 지어지도록 합니다. 결과적으로 MulT는 기존의 방법으로 힘들었던 먼 거리의 *modality* 간의 관령성도 잡아낼 수 있게 되었습니다. 

## 3. Experiments

## 4. Discussion

**MulT**의 핵심은 *crossmodal attention mechanism*으로 *modality* 간의 정보를 함께 고려해 *latent crossmodal adaptation*을 산출해냅니다. 기존의 방법은 직접 align을 해주어야 했지만, 이 모델은 alignment의 가정없이 먼 거리의 연관성을 잡아낼 수 있습니다. 실제로 가장 좋은 성능을 보여주고 있습니다. 이 논문에선 인간의 언어에 대해 다루었지만, 더 많은 적용이 가능할 것이라고 기대됩니다. 