---
date: 2022-08-12
title: "[Paper Review] Adversarially Learned Anomaly Detection"
categories: 
  - Paper Review
tags: 
  - Anomaly Detection
  - Generative Adversarial Networks
toc: true  
toc_sticky: true 
---

# Paper contents

Adversarially Learned Anomaly Detection

Zenati, H., Romain, M., Foo, C. S., Lecouat, B., & Chandrasekhar, V.

2018 IEEE International conference on data mining (ICDM) (2018)

https://ieeexplore.ieee.org/abstract/document/8594897

## 0. Abstract

복잡하고 고차원의 데이터에서 효과적인 이상 탐지를 하는 것은 여전히 어려운 일입니다. *GAN (Generative Adversarial Networks)* 은 복잡하고 고차원의 데이터 분포를 학습하는 것이 가능합니다. 이 논문에서는 양방향 GAN을 바탕으로 한 **Adversarially Learned Anomaly Detection (ALAD)** 를 제안합니다. ALAD는 이상 탐지를 위해 특징을 적대적으로 학습하고, 이렇게 학습된 특징들을 활용하여 재구성 오차를 계산해 이상을 탐지합니다. 특히 최근에 GAN에 데이터 공간 및 잠재 공간의 주기적 일관성과 학습을 안정화 시키는 발전이 있었으며, 이러한 발전을 바탕으로 하였습니다. ALAD는 여러 이미지 및 테이블 형태의 데이터에서 sota를 달성했습니다. 뿐만 아니라 기존에 공개된 GAN 기반의 방법론에 비해 압도적으로 빠른 테스트 속도를 보였습니다.

## 1. BACKGROUND

기본적인 GAN은 두 신경망으로 구성되어 있습니다. 생성자 $G$와 판별자 $D$ 입니다. 이 신경망들은 M개의 라벨이 없는 데이터 $[x^{(i)}]^M_{i=1}$를 통해 훈련됩니다. 생성자  $G$는 잠재 분포로부터 샘플된 무작위 변수들 $z$에서부터 입력 공간으로 맵핑하도록 훈련됩니다. 판별자 $D$는 실제 데이터 샘플인 $x^{(i)}$와 $G$로부터 생성된 $G(z)$를 구분하도록 학습됩니다. 따라서 $G$와 $D$는 경쟁적으로 학습됩니다. $G$는 $D$를 속이도록 학습되며, $D$는 $G$에게 속지 않도록 학습됩니다. 	

$p_{\chi}(x)$는 데이터 공간 $\chi$에서의 데이터 $x$의 분포이며, $p_{\mathcal{Z}}(z)$는 잠재 공간 $\mathcal{Z}$에서의 잠재 생성 변수들 $z$에 대한 분포를 의미합니다. GAN을 훈련 시키는 것은 $D$와 $G$에 대한 문제 $\min_G \max_D V(D,G)$를 푸는 것을 의미합니다. 이는 곳 실제 데이터 분포 $p_\chi(x)$와 유사한 분포 $p_G(x)$를 만드는 것을 의미합니다.

## 2. ADVERSERIALLY LEARNED ANOMALY DETECTION

앞에서 설명한 기본적인 GAN을 이상 탐지 문제에 적용하려는 연구들이 있었습니다. 예를 들면 데이터 $x$에 대해 샘플링을 통해 $x$의 가능도를 추정하고 이상치에 해당되는지 결정하는 방법이 있습니다. 하지만 가능도의 추정은 매우 많은 데이터 샘플이 필요하여 많은 계산이 필요합니다. 다른 방법은 $G$를 뒤집어 재구성 오차를 최소화하는 잠재 변수 $z$를 찾는 방법입니다. 이 방법 역시 계산적으로 비효율적입니다.

### 2-1 GAN architecture

계산적 효율성을 위해 이 논문에서 제안된 모델은 인코더 네트워크 $E$가 동시에 데이터 샘플 x를 잠재 공간 z에 맵핑하도록 학습합니다. 데이터 포인트 $x$를 인코더 네트워크에 입력해 잠재 표현을 계산하도록 합니다. 또한 cycle-consistency를 위해 추가적인 판별자를 더해 인코더 네트워크의 성능을 높히도록 합니다. 이를 간단하게 $G(E(x)) \approx x$ 로 표현 할 수 있습니다.

BiGAN과 AliGAN은 x와 z를 입력으로 받는 적대적 판별자 $D_{xz}$를 통해 결합 분포 $p_G(x,z) = p_\mathcal{Z}(z)p_G(x\vert z)$와 $p_E(x,z) = p_\chi(x)p_E(z\vert x)$ 를 같게 합니다. 이론적으로는 결합 분포 $p_E(x,z)$와 $p_G(x,z)$는 동일하지만 실제로는 훈련으로 인해 반드시 수렴 하지는 않습니다. 이로 인해 cycle-consistency를 보장하지 못 해 $G(E(x)) \not\approx x$가 되어 재구성 기반의 이상 탐지 방법론에 적합하지 않게 됩니다.

이 문제를 해결하기 위해 ALICE 프레임워크에서는 조건부 엔트로피 $H^\pi(x\vert z) = -\mathbb{E}_{\pi(x,z)}[\log\pi(x\vert z)]$ 를 적대적 조건에서 근사 시키는 것을 제안했습니다. 이를 다시 정리하면 다음과 같습니다.

$$\min_{G,E}\max_{D_{xz}V_\text{ALICE}(D_{xz},E,G)}$$ 

위 식은 조건부 엔트로피 정규화를 포함하고 있습니다.

$$V_\text{ALICE}(D_{xz},E,G)=V(D_{xz},E,G)+V_{CE}(E,G)$$

조건부 엔트로피 정규화 $V_{CE}$는 추가적인 판별자 네트워크 $D_{xx}(x,\hat{x})$ 로 근사 할 수 있으며 이는 cycle-consistency를 지킬 수 있음을 의미합니다.

### 2-2 Stabilizing GAN training

베이스라인으로 사용 된 ALICE 모델의 훈련을 안정화 시키기 위해, 조건부 엔트로피 제약을 추가함으로써 조건부 분포를 정규화하고 L2 정규화를 적용하는 법을 제안합니다. 잠재 공간인 조건부 $H^\pi(z\vert x) = -\mathbb{E}_{\pi(x,z)}[\log\pi(z\vert x)]$ 를 추가적인 적대적으로 학습된 판별자를 통해 정규화하며, 식으로 정리하면 다음과 같습니다.

$$
\begin{aligned}
V(D_{zz},E,G) &= \mathbb{E}_{z\sim p_\mathcal{Z}}[\log D_{zz}(z,z)] \\
&+ \mathbb{E}_{z\sim p_\mathcal{Z}}[1-\log D_{zz}(z,E(G(z)))]
\end{aligned}
$$

이를 모두 고려하면, **ALAD**의 방법론은 다음과 같은 문제를 푸는 것을 의미합니다.

$$\min_{G,E}\max_{D_{xz},D_{xx},D_{zz}}V(D_{xz},D_{xx},D_{zz},E,G)$$

여기서 $V(D_{xz},D_{xx},D_{zz},E,G) = V(D_{xz},E,G)+V(D_{xx},E,G)+V(D_{zz},E,G)$ 입니다. 이를 그림으로 정리하면 다음과 같습니다.

![image](https://user-images.githubusercontent.com/35906602/184474341-308ec7d9-f908-4675-9bbd-00249ae29ec7.png){: width="500"}{: .align-center} 

Figure 1. The GAN used in ALAD. $D_{zz}$, $D_{xz}$, and $D_{xx}$ denote discriminators(white), $G$ the generator (orange), and $E$ the encoder (orange); these networks are simultaneously learned during training.
{: style="text-align: center; font-size:0.7em;"}

이 논문에서 제안되는 추가적인 L2 정규화는 GAN의 판별자에 Lipschitz 제약을 추가하는 최근 연구의 효율성에서 영감을 받았습니다. 특히 가중치들의 간단한 re-parametrization이 매우 좋은 효과를 보였다는 연구가 있었습니다. 가령 판별자의 은닉층에서 가중치들의 행렬을 가장 큰 고유값으로 L2 정규화를 적용하는 방법이 있습니다. 이 방법은 계산적으로 효율적이며 훈련을 안정화 시킵니다. 이 논문에서는 이 방법이 인코더의 정규화에도 좋은 효과를 보인다는 점을 실험으로 밝혀냈습니다. 베이스라인이 되는 ALICE 모델에는 가중치의 re-parametrization 방법이 포함되어 있지 않습니다.

### 2-3 Detection anomalies

ALAD는 재구성 기반의 이상 탐지 기법으로, 샘플이 GAN의 재구성 된 값들과 얼마나 큰 차이가 있는지를 바탕으로 평가를 진행합니다. 정상 데이터들은 잘 재구성 될 것이고 비정상 데이터들은 재구성이 잘 되지 않을 것이라는 아이디어에서 출발했습니다. 
이를 위해서는 우선 데이터 분포를 효과적으로 모델링 해야 합니다. 이는 생성자 $G$가 정상 데이터의 분포를 잘 학습해 $p_G(x) = p_\chi(x)$가 될 때 가능합니다. 이때 $p_G(x) = \int p_G(x\vert z)p_\mathcal{Z}(z)dz$ 입니다. 또한 데이터의 매니폴드를 잘 학습하여 잠재 표현을 잘 복원하도록 해야합니다. 이것은 곧 정상 샘플들의 신뢰할 수 있는 재구성으로 이어집니다. 이 논문에서 제안되는 두 개의 대칭적인 조건부 엔트로피 cycle-consistency 정규화 항이 이를 보장합니다.
ALAD의 또 다른 핵심 요소는 이상치 점수입니다. 정상 샘플과 재구성 된 값의 유클리안 거리는 단점이 존재합니다. 예를 들면, 이미지의 경우, 시각적으로 같은 특징을 가졌더라도 유클리안 거리는 멀 수 있기 때문에 적합하지 않습니다.
이 논문에서는 대신에 cycle-consistency 판별자인 $D_{xx}$의 특징 공간으로부터 얻어진 샘플들을 통해 거리를 계산합니다. 구체적으로 정상 데이터로부터 $E, G, D_{xz}, D_{xx}, D_{zz}$를 학습시키고, 점수 함수 $A(x)$를 특징 공간의 $L_1$ 재구성 오차로부터 계산합니다.

$$A(x) = \lVert f_{xx}(x,x) - f_{xx}(x,G(E(x)))\rVert_1$$

여기서 $f(\cdot, \cdot)$은 $D_{xx}$ 네트워크에서 로지스틱 직전의 은닉층 활성화를 의미합니다. $A(x)$는 샘플이 잘 인코딩 되어 있는지, 그리고 생성자를 통해 잘 재구성 되어 있는지에 대한 판별자들의 신뢰성을 의미합니다. 

![image](https://user-images.githubusercontent.com/35906602/184475138-2d5aae37-6637-4dcc-8a90-2590dbcd0790.png){: width="400"}{: .align-center} 

기존 연구는 일반적인 GAN의 판별자를 통해 계산된 특징들을 사용하여 feature-matching 손실을 계산하였습니다. ALAD에서는 대신 $D_{xx}$ 판별자를 사용합니다. 더해서 feature-matching 손실이 학습 단계에서 사용되지 않습니다.

$D_{xx}$는 실제 샘플과 재구성 된 샘플을 구분하도록 학습됩니다. 하지만 이 논문에서 제안된 구조에서는 생성자와 인코더가 완벽하게 학습됩니다. 따라서 $D_{xx}$가 실제 데이터와 재구성 된 데이터를 구분 할 수 없으므로, $D_{xx}$의 출력이 아닌 다른 방법으로 이상 점수를 계산 했습니다.

## 3. Experiments

실험은 테이블 형태의 데이터 및 이미지 데이터를 통해 진행 되었습니다. 실험 설정 및 결과는 다음과 같습니다.

![image](https://user-images.githubusercontent.com/35906602/184475421-ffaabf4c-0c1b-4517-9a4a-df6cdb47ba1e.png){: width="300"}{: .align-center} 

![image](https://user-images.githubusercontent.com/35906602/184475470-89d53945-6607-40c6-a583-b59f0df529bd.png){: width="500"}{: .align-center} 

![image](https://user-images.githubusercontent.com/35906602/184475506-e53f3467-6186-4f77-a047-34f2d8dde1af.png){: width="500"}{: .align-center} 

![image](https://user-images.githubusercontent.com/35906602/184475545-c283805e-c36e-4176-a608-5e2d82319769.png){: width="700"}{: .align-center} 

![image](https://user-images.githubusercontent.com/35906602/184475587-a12764dc-81a8-42c2-91a8-122281aa1f4f.png){: width="400"}{: .align-center} 

기타 실험 결과 및 구체적인 설명은 논문을 참고해주세요.

## 4. Conclusion

이 논문에서는 GAN 기반의 이상 탐지 기법인 ALAD를 제안했습니다. 데이터 공간에서부터 잠재 공간으로 맵핑하는 인코더를 학습하여 계산적으로 효율적입니다. 더해서 인코더에 추가적인 판별자를 적용하였으며, GAN의 훈련을 안정적이도록 돕는 L2 정규화를 적용했습니다. 실험은 이러한 방안들이 이상 탐지 성능을 높였음을 보입니다. 