---
date: 2021-11-02
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

인간의 언어는 자연어, 음향, 얼굴 제스쳐 등이 혼합되어 있는 *multimodal* 입니다. 이러한 언어의 시계열 데이터를 모델링하는 데에는 두 개의 주요한 과제가 있습니다. 첫 번째로 각 modality에서 샘플링 주기가 달라 데이터의 길이가 맞지 않아 데이터가 제대로 정렬되지  않습니다 **non-alignment**.  두 번째로 modality들 사이의 **장기 의존성** 문제가 발생합니다. 이러한 문제를 해결하기 위해 **Multimodal Transformer** *MulT*를 도입했습니다. 데이터를 명시적으로 정렬하지 않고 일반적으로 해결 가능하도록 하는 end-to-end 모델입니다. 이 모델의 핵심은 쌍방향성 교차 모델 어텐션으로, 각 타임 스텝에서 여러 모델의 시퀀스 간의 상호작용에 집중합니다. 



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

### 2.1 Crossmodal Attention

### 2.2 Overall Architecture

### 2.3 Discussion about Attention & Alignment

## 3. Experiments

### 3.1 Datasets and Evaluation Metrics

### 3.2 Baseline

### 3.3 Quantitative Analysis

### 3.4 Qualitative Analysis

## 4. Discussion