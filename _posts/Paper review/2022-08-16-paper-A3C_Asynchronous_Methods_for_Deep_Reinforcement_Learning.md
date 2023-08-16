---
date: 2022-08-16
title: "[Paper Review] A3C: Asynchronous Methods for Deep Reinforcement Learning"
categories: 
  - Paper Review
tags: 
  - Reinforcement Learning
toc: true  
toc_sticky: true 
---

## Reference

Asynchronous Methods for Deep Reinforcement Learning

Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., ... & Kavukcuoglu, K.

International conference on machine learning (ICML) (2016)

http://proceedings.mlr.press/v48/mniha16.html?ref=https://githubhelp.com

## 0. Abstract

이 논문에서는 신경망의 최적화에 비동기적 경사 하강법을 활용한 간단한 심층 강화학습 프레임워크를 제안합니다. 기존 강화학습 알고리즘을 바탕으로 하여 네 가지의 비동기적 방법을 제안하고, 이러한 방법이 신경망을 학습시키는데 효과적임을 보입니다. 특히 actor-critic 방법의 비동기적 방법은 가장 좋은 성능을 보이며 훈련 시간도 빠릅니다. 

## 1. Instroduction

심층 신경망은 강화학습 알고리즘이 잘 작동하도록 도와주는 표현을 학습하도록 돕습니다. 이전에는 강화학습 알고리즘에 이러한 신경망을 함계 사용하는 것이 불안정하다고 여겨졌고, 따라서 알고리즘을 안정화 할 수 있도록 하는 여러 방법들이 제안 되었습니다. 이러한 방법들은 **실시간으로 작동하는 강화학습 에이전트로부터 관측되는 데이터들의 연속은 비정상성을 띄며, 따라서 강화학습 알고리즘의 학습도 강한 상관성을 가진다** 라는 아이디어를 공유합니다. 이러한 문제를 해결하기 위해 *experience memory*를 활용하여 이전의 경험으로부터 데이터를 랜덤하게 샘플링하는 방법이 제안 되었습니다.

*experience memory*를 기반으로 한 **DQN**과 같은 심층 강화학습 방법론은 좋은 성과를 보였습니다. 하지만 이 방법도 단점이 있습니다. 더 많은 저장 공간 및 연산이 필요합니다. 또한 이전의 정책으로부터 생성된 데이터를 활용 가능한 off-policy 방법에만 적용 가능합니다.

본 논문에서는 상기된 문제를 해결하기 위해 다른 접근 방법을 제시하고 있습니다. *replay memory* 대신, 여러 에이전트를 평행으로 비동기적으로 실행합니다. 이렇게 하면 각 에이전트는 같은 시점에서 서로 다른 상황의 state를 경험하고 있기 때문에, 에이전트들의 데이터의 상관성을 낮춰 더 정상성을 띄도록 돕습니다. 이 방법은 on-policy 기반의 알고리즘인 Sarsa, n-step methods, actor-critic methods 등에도 적용 가능하며, Q-learning과 같은 off-policy 알고리즘에도 역시 적용 가능합니다.

이 밖에도 GPU 및 CPU를 더 효율적으로 사용 가능하다는 실용적인 장점 역시 존재합니다.

## 2. Reinforcement Learning Background

우선, 기본적인 강화 학습을 환경 $\mathcal{E}$ 에서 에이전트가 일정 시점동안 상호작용 하는 것이라고 설정합니다. 각 시점 $t$에서 에이전트는 상태 $s_t$를 받게 되고 가능한 행동 집합 $\mathcal{A}$에서 정책 $\pi$에 따라 행동 $a_t$를 선택합니다. 이때 정책 $\pi$는 시점 $s_t$에서의 행동 $a_t$로의 맵핑을 의미합니다. 이로 인해 에이전트는 다음 상태 $s_{t+1}$과 보상 $r_t$를 받습니다. 이 과정은 에이전트가 터미널 상태에 도달 할 때까지 반복합니다. 결과적으로 $R_t = \sum^\infty_{k=0}\gamma^kr_{t+k}$가 시점 $t$에서 얻는 최종 결과이며, 이때 $\gamma$는 할인율 $\gamma \in (0,1]$ 를 의미합니다. 에이전트의 목표는 각 상태 $s_t$에서 예상되는 결과 $R_t$를 최대화 하는 것입니다.

행동의 가치 $Q^\pi(s,a) = \mathbb{E}[R_t\vert s_t = s,a]$는 상태 $s$에서 정책 $\pi$에 따라 행동 $a$를 취했을 때의 예상되는 결과를 의미합니다. 최적의 가치 함수 $Q^*(s,a) = \max_\pi Q^\pi(s,a)$는 모든 정책에서 상태 $s$와 행동 $a$를 위한 최대 행동 가치를 제공합니다. 비슷하게, 정책 $\pi$에서의 상태 $s$의 가치는 $V^\pi(s) = \mathbb{E}[R_t\vert s_t=s]$로 정의되며 상태 $s$에서 정책 $\pi$를 따랐을 때의 기대되는 결과 입니다. 

가치 기반의 model-free 강화학습 방법에서, 행동 가치 함수는 신경망과 같은 근사 함수를 통해 표현됩니다. $Q(s,a;\theta)$를 파라미터 $\theta$를 사용하여 근사한 행동 가치 함수라고 가정합니다. $\theta$의 갱신은 여러 강화학습 알고리즘을 통해 가능합니다. 예를 들면 *Q-learning*에서는 직접적으로 최적의 근사 함수$Q^*(s,a) \approx Q(s,a;\theta)$에 근사하는 것을 목적으로 합니다. *one-step Q-learning*에서는 행동 가치 함수 $Q(s,a;\theta)$의 파라미터 $\theta$가 손실 함수의 시퀀스로 인해 반복적으로 최소화되어 학습되며, 이때 $i$번째 손실 함수는 다음과 같이 정의됩니다. 여기서 $s'$는 상태 $s$ 다음에 만나게 되는 상태를 의미합니다.

$$L_i(\theta_i) = \mathbb{E}\left(r+\gamma\max_{a'}Q(s',a';\theta_{i-1})-Q(s,a;\theta_i)\right)^2$$

위와 같은 방법을 *one-step Q-learning*으로 정의합니다. 이 방법은 행동 가치 $Q(s,a)$를 one-step 결과인 $r + \gamma\max_{a'}Q(s',a';\theta)$를 통해 업데이트 합니다. 이러한 one-step 방법을 사용하면 보상 $r$를 얻는 것이 오직 보상으로 이어지는 상태 행동 쌍인 $s, a$에만 영향을 미친다는 단점이 있습니다. 다른 상태 행동 쌍은 업데이트된 가치인 $Q(s,a)$를 통해 간접적으로만 영향을 받습니다. 따라서 환경에는 무수한 상태와 액션 쌍이 존재하기 때문에 학습 속도가 느립니다.

보상을 더 빠르게 전파하는 방법은 $n-step$ 결과를 사용하는 것입니다. *n-step Q-learning*에서는 $Q(s,a)$가 $r_t + \gamma r_{t+1} + ... + \gamma^{n-1}r_{t+n-1} + \max_a\gamma^nQ(s_{t+n},a)$로 정의되는 $n$단계의 결과를 사용합니다. 하나의 보상 $r$이 연속된 $n$개의 상태 행동 쌍에 직접적으로 영향을 미치게 됩니다. 이를 통해 더 효과적인 학습이 가능합니다.

가치 기반의 방법과는 다르게, 정책 기반의 model-free 방법은 정책 $\pi(a\vert s;\theta)$를 직접적으로 파라미터화하여 근사하며, 대부분의 경우 $\mathbb{E}[R_t]$의 경사상승법을 활용합니다. 이 방법의 한가지 예는 *REINFORCE*입니다. 기본적인 *REINFORCE*는 정책 파라미터 $\theta$를 방향 $\nabla_\theta\log\pi (a_t\vert s_t;\theta)R_t$로 학습하며, 이는 $\nabla_\theta\mathbb{E}[R_t]$의 추정을 의미합니다. 추정의 분산을 결과에서 학습된 상태 함수 $b_t(s_t)$를 뺌으로써 줄일 수 있으며, 이를 *baseline*이라고 부릅니다. 결과적으로 경사는 $\nabla_\theta\log\pi(a_t\vert s_t;\theta)(R_t-b_t(s_t))$가 됩니다. 

학습된 가치 함수의 추정은 일반적으로 *baseline* $b_t(s_t) \approx V^\pi(s_t)$로 활용되며, 정책 경사*policy gradient*의 추정의 분산을 낮춰줍니다. 근사된 가치 함수가 *baseline*으로 사용될 때, $R_t - b_t$는 정책 경사를 스케일합니다. 이는 곧 상태 $s_t$에서의 행동 $a_t$의 *advantage*의 추정, 혹은 $A(a_t, s_t) = Q(a_t, s_t) - V(s_t)$을 의미합니다. $R_t$는 $Q^\pi(a_t,s_t)$의 추정이며 $b_t$는 $V^\pi(s_t)$의 추정이기 때문입니다. 이러한 접근법은 **actor-critic** 구조로 볼 수 있으며, 정책 $\pi$는 *actor*, *baseline* $b_t$는 *critic*으로 볼 수 있습니다. 

## 3. Asynchronous RL Framework

본 논문에서는 *one-step Sarsa*, *one-step Q-learning*, *n-step Q-learning*, *advantage actor-critic* 각각의 비동기적 변형이 소개됩니다. 이 글에서는 *advantage actor-critic*의 비동기적 변형인 **A3C** *Asynchronous Advantage Actor-Critic*을 중점적으로 다루며, 다른 방법들에 대한 자세한 설명은 논문을 참조해주세요.

위 방법들의 설계는 많은 자원의 필요 없이 심층 신경망 정책을 효과적으로 학습 가능하느냐에 중점을 두었습니다. 각각의 방법은 서로 많은 차이점을 가지고 있지만, 두 가지 핵심 아이디어를 활용해 네 알고리즘을 실용적으로 설계했습니다.

우선 비동기적 *actor-learners*를 사용하였으며, 하나의 머신을 활용해 CPU의 쓰레드들을 활용하여 학습을 진행했습니다. 하나의 머신을 활용했기 때문에 파라미터 및 경사를 서로 전달하는데 소요되는 불필요한 비용을 줄일 수 있습니다.

두번째로, 복수의 *actor*를 평행으로 실행되도록 하여 각 *actor*들이 환경의 서로 다른 부분을 탐색하도록 했습니다. 더해서 각 *actor-learner*에 서로 다른 탐험 정책을 사용하여 이런 다양성을 극대화 할 수 있습니다. 각각의 쓰레드에서 각각의 서로 다른 탐험 정책을 사용함으로써, 여러 *actor-learner*들이 실시간 업데이트를 동시에 수행하므로 하나의 에이전트가 실시간 업데이트를 하는 것에 비해 시간적 상관성이 낮을 가능성이 높습니다. 그러므로 여기서는 *experience memory*와 같은 방법을 사용하지 않았습니다.

안정적인 학습에 더해서 여러 실용적 장점이 있습니다. 우선 학습 속도가 빨라졌습니다. 또한 *on-policy* 기반의 강화 학습 방법론을 사용할 수 있습니다. 

### Asynchronous advantage actor-critic

**A3C** *Asynchronous Advantage Actor-Critic*는 정책 $\pi(a_t\vert s_t;\theta)$과 가치 함수의 근사 $V(s_t;\theta_v)$를 가집니다. 정책과 가치 함수는 매 $t_{max}$ 행동 혹은 터미널 상태에 도달했을 때 업데이트 됩니다. 식으로 나타내면 다음과 같습니다.

$$\nabla_{\theta'}\log\pi(a_t\vert s_t;\theta')A(s_t,a_t;\theta,\theta_v)$$

여기서 $A(s_t,a_t;\theta,\theta_v)$는 advantage 함수의 추정이며, 다음과 같이 나타낼 수 있습니다.

$$\sum^{k-1}_{i=0}{\gamma^ir_{t+i}+\gamma^kV(s_{t+k};\theta_v)-V(s_t;\theta_v)}$$

알고리즘으로 표현하면 다음과 같습니다.

![image](https://user-images.githubusercontent.com/35906602/185798475-40b934ac-dc2d-41f9-ae80-591db342fef1.png){: width="600"}{: .align-center}

 여기서 훈련의 안정성을 위해 평행의 *actor-learners*에 의존합니다. 정책의 파라미터 $\theta$와 가치 함수의 파라미터 $\theta_v$가 분리되어 있는 것 처럼 보이지만, 실제로는 파라미터의 일부를 공유하도록 했습니다. *CNN*을 사용했으며, 각각 정책 $\pi(a_t \vert s_t;\theta)$를 의미하는 하나의 *softmax* 출력을 가지는 층과 가치 함수 $V(s_t;\theta_v)$를 의미하는 하나의 선형 출력을 가지는 층이 존재하며 나머지 모든 비선형 층의 파라미터는 공유됩니다.

또한 정책 $\pi$의 엔트로피를 목적 함수에 더하는 것이 탐험을 더 효과적으로 하도록 돕는 것을 발견했습니다. 이를 통해 *local optimum* 에 일찍 수렴해버리는 현상을 완화할 수 있습니다. $\nabla_{\theta'}\log\pi(a_t\vert s_t;\theta')(R_t-V(s_t;\theta_v))+\beta\nabla_{\theta'}H(\pi(s_t;\theta'))$로 나타낼 수 있으며, $H$는 엔트로피를 의미합니다. 하이퍼 파라미터 $\beta$는 얼마나 강하게 엔트로피 정규화를 할 것인지를 조절합니다.

## 4. Experiments

실험 결과는 다음과 같습니다. 자세한 내용은 논문을 참조해주세요.

![image](https://user-images.githubusercontent.com/35906602/185799051-858053a7-c2c6-47ed-9456-4b7e0768f1c4.png){: width="800"}{: .align-center}

![image](https://user-images.githubusercontent.com/35906602/185799072-01882a8d-b962-4160-bbd5-ad4395695aa3.png){: width="400"}{: .align-center}

![image](https://user-images.githubusercontent.com/35906602/185799111-f8ed467a-5e84-4287-b7bb-c824771636f8.png){: width="400"}{: .align-center}

![image](https://user-images.githubusercontent.com/35906602/185799144-eee58b52-0507-4108-a21f-8a09b495c28f.png){: width="800"}{: .align-center}

![image](https://user-images.githubusercontent.com/35906602/185799191-c56a0c23-8495-473f-901b-de07780ed721.png){: width="800"}{: .align-center}

![image](https://user-images.githubusercontent.com/35906602/185799214-e851cc29-b5eb-4b3a-b172-5c61475815b6.png){: width="800"}{: .align-center}

## 5. Conclusion and Discussion

몇몇 강화학습 알고리즘의 비동기적 형태를 제안했으며, 안정적인 학습이 가능함을 보였습니다. 가치 기반의 방법 및 정책 기반의 방법 모두에서 사용 가능하며, 마찬가지로 *on-policy* 방법 및 *off-policy* 방법 양쪽에서 사용 가능합니다. 복수의 *actor-learner*가 공유된 모델을 업데이트하도록 하는 것이 학습 단계에서의 안정성을 높여줌을 보였습니다. 특히 *Q-learning*에서 *experience memory*를 사용하지 않고도 안정적인 학습이 가능함을 보였습니다. 기존의 여러 강화학습 방법에 본 논문에서 제안된 방법을 활용한다면 성능의 향상을 기대할 수 있습니다. 