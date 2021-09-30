---
date: 2021-10-05
title: "Chapter 2. Probability Distribution (3) - 가우시안 분포"
categories: 
  - 머신러닝과 패턴인식
tags: 
  - 머신러닝
  - 패턴인식
  - 확률
toc: true  
toc_sticky: true 
---
*본 글은 책 '패턴 인식과 머신 러닝'의 국내 출판본을 바탕으로 작성되었습니다.*

# 2.3 가우시안 분포

## 가우시안 분포의 특징

### 식

단일 변수 $x$에 대한 가우시안 분포는 다음과 같습니다.

$$\mathcal{N}(x\vert\mu,\sigma^2) = \frac{1}{(2\pi\sigma^2)^\frac12}\exp\left\{-\frac1{2\sigma^2}(x-\mu)^2\right\}$${: .notice}

여기서 $\mu$는 평균, $\sigma^2$는 분산입니다. $D$차원 벡터 $\mathbf{x}$에 대한 다변량 가우시안 분포는 다음과 같습니다.

$$\mathcal{N}(\mathbf{x}\vert\boldsymbol\mu, \boldsymbol{\Sigma}) = \frac1{(2\pi)^{D/2}}\frac1{\left\vert\boldsymbol{\Sigma}\right\vert^{1/2}}\exp\left\{-\frac12(\mathbf{x}-\boldsymbol{\mu})^\text{T}\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right\}$${: .notice}

여기서 $\boldsymbol\mu$는 $D$차원 평균 벡터를, $\boldsymbol\Sigma$는 $D \times D$ 공분산 행렬을, $\left\vert\boldsymbol\Sigma\right\vert$는 $\boldsymbol\Sigma$의 행렬식을 의미합니다.

여러 확률 변수의 합에 대해 고려할 때 가우시안 분포를 사용할 수 있습니다.  **중심 극한 정리** *central limit theorem*에 따르면 여러 개의 확률 변수들의 합에 해당하는 확률 변수는 몇몇 조건하에서 합해지는 확률 변수의 숫자가 증가함에 따라서 점점 가우시안 분포가 되어갑니다. 여기서 이항 분포 역시 $N \rightarrow \infty$이 됨에 따라 가우시안의 평태를 띈다는 점을 알 수 있습니다.

<div>
 <img src="/assets/images/ml/Figure2.6a.png" width="250" alt=""  /> 
 <img src="/assets/images/ml/Figure2.6b.png" width="250" alt=""  />
 <img src="/assets/images/ml/Figure2.6c.png" width="250" alt="" />
</div>

Figure 2.6 균일하게 분포된 $N$개의 값의 평균에 대한 히스토그램
{: style="text-align: center; font-size:0.7em;"}

### 기하학적 형태

$\mathbf{x}$에 대한 가우시안 분포의 함수적 종속성은 지수상에서 나타납니다. 이는 다음의 이차식 형태를 띱니다.

$$\Delta^2 = (\mathbf{x}-\boldsymbol\mu)^\text{T}\boldsymbol\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)$$

여기서 $\Delta$값은 $\boldsymbol\mu$로부터 $\mathbf{x}$까지의 **마할라노비스 거리** *Mahalanobis distance*라고 합니다. 마할라노비스 거리는 $\boldsymbol\Sigma$가 항등 행렬일 경우 유클리디안 거리가 됩니다. 이 이차식이 상수가 되는 $\mathbf{x}$ 공간의 표면에서는 가우시안 분포 역시 상수가 됩니다. $\boldsymbol\Sigma$ 행렬은 비대칭 원소들이 지수로부터 사라지기 때문에 대칭성을 띱니다.

$$\boldsymbol\Sigma \mathbf{u}_i = \lambda_i\mathbf{u}_i$$

여기서 $i = 1, ..., D$입니다. $\boldsymbol\Sigma$가 실수 대칭 행렬이므로 고윳값 역시 실수입니다. 정규직교 집합을 이루도록 고유 벡터들을 선택한다고 하겠습니다.

$$\mathbf{u}_i^\text{T}\mathbf{u}_j = I_{ij}$$ 

여기서 $I_{ij}$는 항등 행렬의 $i$와 $j$번째 원소입니다. 이때 고유 벡터를 이용해서 공분산 행렬 $\boldsymbol\Sigma$를 전개할 수 있으며 이는 다음 형태를 띱니다.

$$\begin{aligned}
\boldsymbol\Sigma &= \sum^D_{i=1}\lambda_i\mathbf{u}_i\mathbf{u}_i^\text{T} \\
\boldsymbol\Sigma^{-1}&=\sum^D_{i=1}\frac1{\lambda_i}\mathbf{u}_i\mathbf{u}_i^\text{T}
\end{aligned}$$

이 식을 위의 $\Delta^2$의 식에 대입하면 다음의 형태를 띱니다.

$$\begin{aligned}
\Delta^2 = \sum^D_{i=1}\frac{y_i^2}{\lambda_i} \\
y_i = \mathbf{u}_i^\text{T}(\mathbf{x}-\boldsymbol{\mu})
\end{aligned}$$

$\{y_i\}$를 정규직교 벡터 $\mathbf{u}_i$들로 정의되는 새로운 좌표계라고 해석할 수 있습니다. 원래의 $x_i$ 좌표계로부터 이동되고 회전된 것입니다. 벡터 $\mathbf{y} = (y_1,...,y_D)^\text{T}$이라 하면 다음과 같습니다.

$$\mathbf{y} = \mathbf{U(x-\boldsymbol\mu})$$

여기서 $\mathbf{U}$는 각각의 행이 $\mathbf{u}_i^\text{T}$로 주어지는 행렬입니다. $\mathbf{U}$는 **직교** *orthogonal*한 행렬임을 알 수 있습니다.

## 조건부 가우시안 분포

## 주변 가우시안 분포

## 가우시안 변수에 대한 베이지안 정리

## 가우시안 분포의 최대 가능도

## 순차 추정

## 가우시안 분포에서의 베이지안 추론

## 스튜던트 t 분포

## 주기적 변수

## 가우시안 분포의 총합

