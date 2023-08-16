---
date: 2022-08-05
title: "[Paper Review] TS2Vec: Towards Universal Representation of Time Series"
categories: 
  - Paper Review
tags: 
  - Time Series
  - Representation Learning
  - Contrastive Learning
toc: true  
toc_sticky: true 
---
## Reference

TS2Vec: Towards Universal Representation of Time Series

Yue, Z., Wang, Y., Duan, J., Yang, T., Huang, C., Tong, Y., & Xu, B. 

Proceedings of the AAAI Conference on Artificial Intelligence (2022)

https://ojs.aaai.org/index.php/AAAI/article/view/20881

## 0. Abstract

이 논문은 시계열의 표현을 학습하기 위한 프레임워크인 **TS2Vec** 를 제안합니다. TS2Vec는 증강된 맥락 뷰에 대해 계층적 방식의 대조 학습을 수행하므로 각 시점에 대한 강력한 컨텍스트 표현이 가능합니다. 다양한 데이터셋에서 비지도 시계열 표현학습 분야에서 SOTA를 달성했습니다. 특히 학습된 표현을 활용한 선형 회귀로 기존의 시계열 예측의 SOTA를 갱신했습니다. 이에 더해 비지도 이상탐지에 해당 방법론을 활용할 방안을 제안합니다.  

## 1. Introduction

시계열 데이터 연구의 활용 분야는 매우 다양합니다.  시계열 데이터의 보편적인 표현을 학습하는 것은 중요하지만 매우 어려운 과제입니다. 다양한 연구가 진행되었지만 많은 한계가 있습니다.

첫 번째로, instance 단계에서 표현을 학습하는 것은 시계열 예측 혹은 시계열 이상탐지에 적합하지 않습니다. 이러한 분야에서는 주로 시계열 일부분에서의 추론이 필요하지만, 전체 시계열로부터 구해진 표현은 좋은 성능을 보여주지 못 합니다.

두 번째로, 여러 스케일의 맥락 정보를 구분해낼 수 있는 방법론은 거의 없습니다. 특정한 길이로 데이터를 나누거나 혹은 원본 시계열의 무작위 샘플을 사용하는 방식으로 이루어지며, 이러한 방식은 다른 스케일의 입력에는 강건한 성능을 보여주지 못 합니다. 다양한 스케일의 특징을 고려할 수 있다면 다른 수준의 semantic을 볼 수 있고 따라서 일반화 능력이 향상됩니다.

세 번째로 현존하는 대부분의 비지도 시계열 표현 학습 방법론은 컴퓨터 비전 혹은 자연어 분야에서 영감을 받았으며, 이러한 방법들은 잘라내기 혹은 변환에 대해 강한 귀납적 bias를 가집니다. 하지만 시계열의 경우 이런 특징이 적합하지 않을 수 있습니다. 가령 시계열에서 잘라낸 일부분은 전체 시계열과 다른 특징을 가질 가능성이 큽니다.

이러한 문제점들을 개선하고자 이 논문에서는 모든 semantic 수준에서 가능한 일반적인 시계열 표현 학습 방법론인 **TS2Vec**을 제안합니다. 이 방법론은 instance 단계와 시간적 차원에서 positive 혹은 negative sample을 계층적으로 구별하며, 특정 부분에 대해서는 max pooling을 통해 전체적인 표현을 얻을 수 있습니다. 이러한 방법을 통해 여러 해상도의 맥락적 정보를 잡아낼 수 있고, 대부분의 시간 단위에 대한 표현을 얻을 수 있습니다. 또한 대조학습을 기반으로 하기 때문에 같은 샘플에 대해 증강된 두 맥락에 대한 표현이 같도록 학습되며, 그렇기 때문에 각 샘플에 대한 강력한 맥락적 표현을 얻을 수 있습니다.

이 논문의 기여를 정리하면 다음과 같습니다.

* 다양한 시계열의 부분에 대한 맥락적 표현을 학습할 수 있는 프레임워크를 제안했습니다. 모든 종류의 시계열에 활용할 수 있는 유연한 방법론을 제안한 것은 이 논문이 최초입니다.
* 대조 학습 프레임워크에 대한 두 가지 새로운 방법을 제안했습니다. 여러 스케일의 맥락에 대한 정보를 잡아내기 위해 instance 수준 및 시간적 차원에 대한 계층적 대조 학습 방법을 제안했습니다. 더해서, positive 쌍에 대한 *contextual consistency*를 제안했습니다.
* 시계열 분류, 예측, 그리고 이상 탐지에 대해 SOTA를 달성했습니다. 

## 2. Method

### 2-1. Problem Definition

시계열 $\mathcal{\chi} = \{x_1, x_2, ..., x_N\}$의 $N$개의 인스턴스들에 대해서, 각각의 $x_i$에 대해 표현 $r_i$로 맵핑시키는 최선의 비선형 임베딩 함수 $f_\theta$를 찾는 것을 목적으로 합니다. 입력된 시계열 $x_i$는 $T \times F$ 차원으로 이루어져 있으며, $T$는 시퀀스의 길이, $F$는 특징의 차원을 의미합니다. 표현 $r_i = \{r_{i,1}, r_{i,2}, ... r_{i,T}\}$는 각각의 시점 $t$마다 표현 벡터인 $r_{i, t} \in \mathbb{R}^K$을 포함하며 여기서 $K$는 표현 벡터의 차원을 의미합니다.

### 2-2. Model Architecture

![image](https://user-images.githubusercontent.com/35906602/183008256-ba5a4884-acca-418e-a80f-7b7216d1a2a8.png){: width="600"}{: .align-center} 

Figure 1. The proposed architecture of TS2Vec. Although this figure shows a univariate time series as the input example, the framework supports multivariate input. Each parallelogram denotes the representation vector on a timestamp of an instance. 
{: style="text-align: center; font-size:0.7em;"}

입력된 시계열 $x_i$로부터 서로 겹쳐지는 두 부분을 무작위로 샘플링하여, 공통 부분에서 맥락적 표현의 일관성을 촉진합니다. 데이터는 시간적 대조 손실과 인스턴스 대조 손실을 결합하여 최적화 한 인코더에 입력됩니다. 총 손실은 계층적 프레임워크의 여러 스케일의 손실들을 더해서 계산합니다.

인코더 $f_\theta$는 3가지 구성요소로 이루어져 있습니다. 

* **input projection layer** <br> 입력 투사층은 각 입력 $x_i$에 대해 각 시점 $t$의 관측치 $x_{i,t}$를 고차원의 잠재 벡터 $z_{i,t}$로 맵핑하는 FC층 입니다.
* **timestamp masking module** <br> 시점 마스킹 모듈은 임의로 선택된 시점의 잠재 벡터를 마스킹해 증강된 맥락 뷰를 생성합니다. 
* **dilated CNN module** <br> 확장된 CNN 모듈은 10개의 잔차 블럭으로 구성되어 있으며, 각 시점에서 맥락적 표현을 추출합니다. 각 블럭은 1D 합성곱 층으로 이루어져 있으며, 확장 파라미터 ($l$번째 블락은 $2^l$)를 포함합니다. 확장된 합성곱은 각기 다른 도메인에 대한 커다란 큰 receptive field를 가능하도록 합니다.   

### 2.3 Contextual Consistency

![image](https://user-images.githubusercontent.com/35906602/183021101-c1908124-f4d0-419a-b83a-3bbec1d44402.png){: width="600"}{: .align-center} 

Figure 2. Positive pair selection strategies.
{: style="text-align: center; font-size:0.7em;"}

![image](https://user-images.githubusercontent.com/35906602/183019470-0c1b494d-b972-4773-a7c3-d50fd1c941e6.png){: width="600"}{: .align-center} 

Figure 3. Two typical cases of the distribution change of time series, with the heatmap visualization of the learned representations over time using subseries consistency and temporal consistency respectively.
{: style="text-align: center; font-size:0.7em;"}

대조 학습에서 positive 쌍을 만드는건 매우 중요합니다. 현재까지의 대부분의 방법들은 데이터의 분포에 대한 강한 가정을 포함하고 있으며, 따라서 시계열에는 적합하지 않습니다. 특히 level shift가 존재하거나 이상치가 존재하는 등의 경우에 잘 작동하지 않습니다. 이러한 문제를 극복하기 위해 본 논문에서는 **contextual consistency**라는 방법을 제안합니다. 

*맥락적 일관성*은 같은 시점에서부터 비롯된 두 증강된 맥락들에 대한 표현들을 positve 쌍으로 다루게 됩니다. 맥락은 입력 시계열에 대한 시점 마스킹 혹은 무작위 잘라내기를 통해 생성됩니다. 이러한 방법은 크게 두가지 장점이 있습니다.
* 마스킹이나 잘라내기는 시계열의 크기를 변화시키지 않으며, 시계열에서는 중요한 요소입니다.
* 각 시점에 대해 각각의 맥락에서 자체적으로 재구성되도록 하여 학습된 표현에 대한 강건함을 향상시킵니다.

#### Timestamp Masking
인스턴스의 무작위 시점에 대해 마스킹을 하여 새로운 맥락적 뷰를 제공합니다. 구체적으로 *입력 투사층* 을 통과한 잠재 벡터 $z_i = \{z_{i,t}\}$를 시간 축을 통하여 마스킹하며, 이진 마스크 $m \in \{0,1\}^T$를 사용합니다. 이때 $p=0.5$인 베르누이 분포로부터 독립적으로 샘플링 됩니다. 마스크는 인코더의 모든 길에서 독립적으로 샘플링 됩니다.

#### Random Cropping
무작위 자르기 역시 새로운 맥락을 만들기 위해 적용됩니다. 입력 $x_i \in \mathbb{R}^{T\times F}$에 대해 두 겹쳐진 시간 부분인 $[a_1, b_1], [a_2, b_2]$를 $0 < a_1 \le a_2 \le b_1 \le b_2 \le T$ 범위에서 무작위로 합니다. 겹쳐진 부분인 $[a_2, b_1]$의 맥락적 표현은 두 가지 맥락 뷰에 대해 일관적이여야 합니다. 위 두가지 방법은 훈련 단계에서만 사용됩니다.

### 2.4 Hierarchical Contrasting

![image](https://user-images.githubusercontent.com/35906602/183024060-60823528-d876-4f01-a223-ca7ed78b5c14.png){: width="600"}{: .align-center} 

$$\mathcal{L}_{\text{dual}} = \frac1{NT}\sum_i\sum_t(\ell^{(i,t)}_\text{temp} + \ell^{(i,t)}_\text{inst})$$
계층적 대조 손실은 인코더로 하여금 다양한 스케일의 표현을 학습하도록 합니다. 시점 수준의 표현을 기반으로 하여, 시간축을 통해 학습된 표현에 *Max pooling*을 적용한 후 위 식을 통해 재귀적으로 계산했습니다. 특히 가장 위의 semantic 레벨의 대조는 인스턴스 레벨의 표현을 학습하도록 돕습니다. 기존의 방법은 특정 수준에 한정된 표현 학습을 하는 반면, 이 방법은 전체적인 표현 학습이 가능합니다.

시계열의 맥락적 표현을 더 잘 잡아내기 위해 인스턴스 레벨과 시간적 레벨의 대조 손실을 함께 활용하여 시계열 분포를 인코딩 했습니다. 손실 함수는 모든 세분화 단계에 적용됩니다.

#### Temporal Contrastive Loss
시간의 흐름에 따른 차별화 된 표현을 학습하기 위해 같은 시점의 표현의 두가지 관점을 positive 쌍으로 고려하고, 같은 시계열의 다른 시점을 negative 쌍으로 봅니다. $i$는 입력된 시계열 샘플의 인덱스, $t$는 시점이라고 할 때, $r_{i,t}$와 $r'_{i,t}$는 같은 시점 $t$에서의 $x_i$의 두 증강을 의미합니다. $i$번째 시계열에 대한 $t$시점의 시간적 손실은 다음과 같이 계산됩니다.

$$\ell^{(i,t)}_\text{temp} = -\log\frac{\exp(r_{i,t}\cdot r'_{i,t})}{\sum_{t' \in \Omega}(\exp(r_{i,t}\cdot r'_{i,t'})+\mathbf{1}_{[t \neq t']}\exp(r_{i,t}\cdot r_{i, t'}))}$$ 

이때 $\Omega$는 두 샘플이 겹쳐지는 시점의 집합, $\mathbf{1}$는 지시 함수를 의미합니다.

### Instance-wise Contrastive Loss

$$\ell^{(i,t)}_\text{inst} = -\log\frac{\exp(r_{i,t}\cdot r'_{i,t})}{\sum^B_{j=1}(\exp(r_{i,t}\cdot r'_{j,t})+\mathbf{1}_{[i \neq j]}\exp(r_{i,t}\cdot r_{j, t}))}$$ 

여기서 $B$는 배치 사이즈를 나타냅니다. 같은 배치에 속해있는 서로 다른 시계열의 $t$ 시점에서의 표현을 negative 샘플로 사용합니다.

두 손실은 서로 상호보완적입니다. 예를 들면, 여러 사용자의 전기 사용량 데이터가 주어진다면, 인스턴스 대조는 유저의 특성을 파악할 수 있으며, 시간적 대조는 시간의 흐름에 따른 동적인 트렌드를 목적으로 합니다. 

## 3. Experiment

학습된 표현을 시계열 분류, 시계열 예측, 그리고 시계열 이상 탐지에서 평가했습니다. 

### 3.1 Time Series Classification

시계열 분류에서는 전체 시계열 (인스턴스)에 라벨이 붙어 있습니다. 따라서 모든 시점에 대한 표현들에 *Max Pooling*을 활용하여 인스턴스 수준의 표현을 얻었습니다. UCR의 128개 단변량 시계열 데이터셋과 UEA의 30개 다변량 데이터셋을 활용하여 진행하였고, 다른 방법에 비해 많은 성능의 향상을 보였습니다. 또한 학습 시간도 가장 짧았습니다. TS2Vec는 하나의 배치에서 다양한 세분성을 통한 대조 손실을 적용하였으므로 매우 효율적인 학습 시간을 보였습니다.

![image](https://user-images.githubusercontent.com/35906602/183038155-67ea58f0-f4a6-449a-9ad5-f9a32d8474bb.png){: width="600"}{: .align-center} 

Table 1. Time series classification results compared to other time series representation methods. The representation dimensions of TS2Vec, T-Loss, TS-TCC, TST and TNC are all set to 320 and under SVM evaluation protocol for fair comparison.
{: style="text-align: center; font-size:0.7em;"}

### 3.2 Time Series Forecasting

마지막 관측치들 $T_l$인 $x_{t-T_{l+1}}, ..., x_t$가 주어졌을 때, 시계열 예측은 미래의 $H$개의 관측치 $x_{t+1}, ..., x_{t+H}$ 관측치를 예측하는 것을 목표로 합니다. 마지막 시점의 표현인 $r_t$를 사용하여 미래의 관측치를 예측하게 됩니다. 구체적으로 $r_t$ 를 입력으로 받아 직접적으로 미래의 값 $\hat{x}$를 예측하며, $L_2$ 페널티를 활용한 선형 회귀 모델을 활용 했습니다. $x$가 단변량 시계열일 때 $\hat{x}$는 $H$ 차원을 가지며 $x$가 $F$ 종류의 특징을 가진 다변량 시계열일 때는 $FH$ 차원을 가집니다. 

![image](https://user-images.githubusercontent.com/35906602/183040237-d96b2d7d-5e54-41a5-99e6-d8b604e32b89.png){: width="600"}{: .align-center} 

Table 2. Univariate time series forecasting results on MSE.
{: style="text-align: center; font-size:0.7em;"}

![image](https://user-images.githubusercontent.com/35906602/183040869-6c167ab9-2216-4de8-ac04-7cdb43e8a794.png){: width="600"}{: .align-center} 

Figure 4: A prediction slice (H=336) of TS2Vec, Informer and TCN on the test set of ETTh$_2$.
{: style="text-align: center; font-size:0.7em;"}

대부분의 경우에서 SOTA를 달성했습니다. 또한 위 그림에서 장기적인 트렌드와 지역적인 패턴 둘 다 잘 잡아낸 것을 확인할 수 있습니다.

### 3.3 Time Series Anomaly Detection

잘려진 시계열 $x_1, x_2, ..., x_t$가 주어질 때, 시계열 이상 탐지는 마지막 시점 $x_t$가 이상인지 아닌지를 결정하는 문제로 정의합니다. 학습된 표현에서 이상 포인트는 정상 포인트들에 비해 분명한 차이점을 보입니다. 이 논문에서는 이상치 점수를 마스크 된 입력과 마스크 되지 않은 입력의 차이를 이상치 점수로 정의합니다. 

$$\alpha_t = \lVert r_t^u - r_t^m\lVert_1$$

또한 drift 유형을 잘 발견하기 위해, 연속된 $Z$ 시점들의 지역적인 평균을 활용하여, $ \bar\alpha_t = \frac{1}{Z} \sum^{t-1}_{i=t-Z} \alpha_i $를 구했습니다.

이후 이상치 점수를 $ \alpha_t^{\text{adj}} = \frac{\alpha_t - \bar\alpha_t}{\bar\alpha_t}$로 조정했으며, 인퍼런스 단계에서는 시점 $t$에서 $\alpha_t^\text{adj} > \mu+\beta\sigma$ 일 경우 이상치로 판정하며, $\mu$와 $\sigma$는 여태까지 점수들의 평균과 표준편차고 $\beta$는 하이퍼 파라미터 입니다.

![image](https://user-images.githubusercontent.com/35906602/183044427-196aa3ae-4f36-46f6-aef5-6f3431a61d79.png){: width="500"}{: .align-center} 

Table 3: Univariate time series anomaly detection results.
{: style="text-align: center; font-size:0.7em;"}

또한 사전 학습이 필요하지 않은 모델도 있기 때문에, 이러한 경우에는 TS2Vec은 다른 데이터셋인 *FordA*을 활용해서 학습하고, 테스트는 *Yahoo*와 *KPI*로 진행했습니다. 결과는 위의 표와 같습니다. 또한 $\mu$와 $\sigma$는 훈련 데이터셋 전체를 활용해서 계산했으며, 전이학습의 경우 해당 데이터포인트 이전의 모든 데이터들을 활용해 계산했습니다.

## 4. Analysis

### 4.1 Ablation Study

![image](https://user-images.githubusercontent.com/35906602/183068674-10624093-bce0-45e0-a939-59762076dcf1.png){: width="600"}{: .align-center} 

Table 4: Ablation results on 128 UCR datasets.
{: style="text-align: center; font-size:0.7em;"}

제시하는 프레임워크의 성능을 검증하기 위해 다양한 실험을 진행했습니다. 결과는 위의 표와 같고, 모든 방법론이 유의미한 차이를 보였다는 내용입니다. 

### 4.2 Robustness to Missing Data

현실의 시계열 데이터에는 결측치가 흔하게 발생합니다. 일반적으로 적용 가능한 프레임워크를 위해서는 결측치에 대한 대응도 가능해야 합니다. 이때 본 논문에서 제시되는 계층척 대조와 시점 마스킹이 큰 역할을 합니다. 시점 마스킹은 완전하지 않은 문맥에 대한 표현을 해석력을 늘려주며, 계층척 대조는 장기적인 정보를 제공하여 결측된 시점 주변의 정보가 불완전한 경우에도 잘 대응하도록 돕습니다. 

![image](https://user-images.githubusercontent.com/35906602/183070088-ce855fd3-8ecd-41f7-bb21-a88fee726f73.png){: width="600"}{: .align-center} 

Figure 5: Accuracy scores of the top 4 largest datasets in UCR archive with respect to the rate of missing points.
{: style="text-align: center; font-size:0.7em;"}

UCR 데이터셋의 가장 큰 4가지 데이터를 통해 실험이 진행되었습니다. 훈련 데이터와 시험 데이터 양쪽에 무작위로 마스킹을 했습니다. 결과는 위의 그래프와 같습니다. 특히 계층적 대조가 없는 경우 결측치 비율이 높아질수록 급격하게 성능이 감소하는 모습을 볼 수 있는데, 결측치의 양이 많을 경우 장기간의 정보가 필요하다는 점을 알 수 있습니다. 이러한 결과로 TS2Vec는 결측치에 매우 강건하다고 결론 지을 수 있습니다.

### 4.3 Visualized Explanation

![image](https://user-images.githubusercontent.com/35906602/183071703-ce2c3994-3593-415b-b724-a94d8390091a.png){: width="600"}{: .align-center} 

Figure 6: The heatmap visualization of the learned represen- tations of TS2Vec over time.
{: style="text-align: center; font-size:0.7em;"}

UCR의 세 데이터셋을 활용해 학습된 표현을 시각화 했습니다. 테스트 데이터에서 샘플을 선택하고 분산이 제일 컸던 차원들을 선택했습니다. 첫번째 그림은 이진 디지털 신호와 비슷하며, 높은 값과 작은 값을 잘 구분합니다. 두번째 그림은 변동성이 점점 작아지는 데이터에 대한 것이며 학습된 표현이 시간이 지날수록 변하는 트렌드를 잘 반영합니다. 세번째 그림은 갑작스럽게 튀는 값들이 있는 데이터로 학습된 표현 역시 노말 데이터와 명백한 차이점을 보입니다.

## 5. Conclusion
이 논문은 시계열에 대한 일반적인 표현 학습 프레임워크 **TS2Vec** 를 제안했습니다. 특히 스케일에 무관한 표현을 학습하기 위해 계층적 대조를 적용했습니다.  평가는 시계열 분류, 시계열 예측 그리고 시계열 이상 탐지로 진행했으며 범용성과 효과성을 검증했습니다. 또한 결측치에도 굉장히 강건하게 대응함을 보였습니다. 이 프레임워크는 일반적으로 적용할 수 있으며, 다른 도메인에도 충분히 적용할 수 있을 것입니다.