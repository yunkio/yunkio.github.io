---
date: 2021-11-11
title: "[Paper Review] A Fault Detection and Diagnosis System for Autonomous Vehicles Based on Hybrid Approaches"
categories: 
  - Paper Review
tags: 
  - 머신러닝
  - Fault Detection
  - Forecasting
  - 논문 리뷰
toc: true  
toc_sticky: true 
---

# Paper contents

A Fault Detection and Diagnosis System for Autonomous Vehicles Based on Hybrid Approaches

Yukun Fang, Haigen Min, Wuqi Wang, Zhigang Xu, Xiangmo Zhao

IEEE Sensors Journal, 2020.

https://ieeexplore.ieee.org/abstract/document/9066934

## 0. Abstract

이 논문에선 고장 감지 및 진단을 위한 복합적인 접근 방식을 제안합니다. 먼저 자율주행차량의 고장을 감지하기 위해 **One-Class Support Vector Machine** *SVM*을 적용시켜 안전한 상태와 안전하지 않은 상태를 구분합니다. 동시에 선형 운동학적 *linear kinematic* 차량 자전거 모델을 기본으로 한 **Kalman filter observer**를 설계하여 차량의 현재 위치를 예측하고, 예측과 실제값의 잔차를 구한 후, 잔차 확률 분포의 정규성을 확인하기 위해 **Jarque-Bera test**를 적용해 경로 이탈 여부를 모니터링합니다. 그리고 고장의 유형을 구분하기 위해 입력층 다음에 **membership function layer**가 추가된 인공 신경망을 제안합니다. 이를 통해 각 고장 유형의 확률을 나타낼 수 있습니다. 자율주행치 플랫폼 'Xinda'를 통해 다른 방법론과의 성능 비교로 제안된 방법의 유용성을 입증했습니다. 

## 1. Introduction

자율주행차의 안전성은 상업화를 막고 있는 주요 이슈 중 하나입니다. 따라서 고장 감지 및 진단 시스템은 필수적입니다. 고장은 일반적인 조건에서 하나 이상의 속성 혹은 매개변수의 허용되지 않은 편차로 정의할 수 있습니다. 고장은 크게 센서 고장, 엑추에이터 고장, 그리고 구성 요소 및 프로세스 고장으로 분류됩니다. 센서 고장은 입력 모듈, 엑추에이터 고장은 출력 모듈의 고장과 관련 있습니다. 구성 요소 및 프로세스 고장은 기타 다른 모듈 혹은 장치의 고장을 의미합니다. 자율주행차의 경우 고장은 센서로부터 발생할 수 있으며 이런 경우 시스템의 다른 요소에 비정상성이 반영됩니다.

고장 진단은 크게 세 가지로 구성됩니다. **고장 감지**는 시스템에 오작동 혹은 고장이 있는지 점검하고, **고장 격리**는 고장난 부품의 위치를 찾고, **고장 식별**은 어떤 종류의 고장인지 찾게 됩니다. 다중화는 시스템의 신뢰성을 높이기 위한 중요 개념이며 하드웨어 다중화, 그리고 분석적 다중화가 필요합니다. **하드웨어 다중화**는 중복된 장치에서 들어온 데이터를 비교하는 것이며, 믿을 수 있고 중요 부품에는 필요하지만 전체 시스템에 적용하기엔 비용이 지나치게 많이 듭니다. **분석적 다중화**는 통계적 방법을 활용하여 고장을 진단하는 추정 기법이며 모델 기반 접근법, 신호 기반 접근법, 지식 기반 접근법으로 나뉩니다.

**모델 기반 접근법**에서는 실제 출력값과 모델이 예측한 출력값 사이의 일관성을 모니터링하는 알고리즘이 사용됩니다. **신호 기반 접근법**에서는 고장이 신호에 반영된다고 가정합니다. 측정된 신호에서 속성을 추출하고 증상 분석 및 해당 증상에 대한 사전 지식을 활용하며, 기대치에 맞지 않는 패턴을 데이터에서 찾아냅니다. **지식 기반 접근법**은 모델이나 신호 패턴이 존재하지 않는 경우가 있기 때문에 필요합니다. 데이터 기반 방법이라고 불리며, 시스템에 대한 속성을 학습하기 위해 많은 양의 과거 데이터가 필요합니다. 학습된 속성을 기반으로 관측되는 행동과 과거의 행동 사이의 일관성으로 고장을 감지합니다. 

각 접근법은 각각의 장점과 한계점이 있습니다. 모델 기반 접근법은 적은 수의 실시간 데이터만 필요로 하지만 입출력 관계를 나타내는 모델의 명시성에 크게 의존하며 현실에서는 적용하기 매우 힘든 경우가 많습니다. 신호 기반 접근법과 지식 기반 접근법은 완전한 모델을 필요로 하지 않습니다. 하지만 신호 기반 접근법은 알려지지 않은 입력 오류에 크게 영향받으며, 지식 기반 접근법은 많은 양의 데이터가 필요하고 계산량이 큽니다. 따라서 측정된 신호나 수집된 데이터의 질이 매우 중요합니다.

### 이 논문에서는...

복합적 접근 방식을 제안함으로써 각 접근법에서 장점을 취해 각각의 이슈에 다르게 처리합니다.  자율주행차에서는 **상황 인식**과 **모션 제어**가 두 필수적인 요소입니다. 자율주행 차량의 움직임은 차량 운동학 및 역학에 의해 모델될 수 있으며, 환경에 대한 지식은 센서를 통해 모아진 데이터에서 얻어집니다.  따라서 자율주행차의 고장 감지 및 진단을 위해서는 여러 감지 진단 방법을 합친 복합적 접근 방식이 합리적입니다. 

첫 번째로, 오류가 발생하는지 감지하기 위해 경계 곡선을 감지하는 **One-Class SVM**이 사용됐습니다. 자율주행차의 상태는 속도 $v$, 각속도 $w$로 나타나며 고장은 정상 상태에서의 $v$와 $w$와의 편차로 나타납니다. 상태 결함의 원인은 동적 시스템, 브레이크 시스템, 그리고 조향 시스템 같은 하위 시스템에서 비롯됩니다.

그 다음 운동학적 모델로 자율주행차의 궤적을 예측할 수 있기 때문에 차량 궤적의 이탈이 있는지 확인하기 위해 모델 기반 접근 방식이 적용될 수 있습니다. 차량 운동학적 모델에 기반하여 차량의 위치와 이상적인 궤적을 예측하기 위한 **Kalman filter observer**가 적용됩니다. 예측된 값과 측정된 값의 잔차를 구해 잔차 분포 추론을 활용하여 궤적 이탈을 검출합니다. 

그리고 감지된 고장의 타입을 식별하기 위해 입력층 바로 뒤에 **membership function layer** 추가된 수정된 신경망에 기반한 퍼지 시스템을 고안했습니다. 이를 통해 각 고장 유형에 대한 확률을 얻게 됩니다.

### Contribution

이 논문의 주요 기여는 다음과 같습니다. 첫 번째로 복합적 접근 방식을 통한 고장 감지 시스템을 고안했으며, 두 번째로 퍼지 시스템의 membership function을 신경망과 블랙박스 기법으로 최적화 했습니다.

## 2. System Framework and Methodology


![image](https://user-images.githubusercontent.com/35906602/141281639-4eb657ee-9355-479e-afb1-77c641a575a6.png){: width="600"}{: .align-center} 

Figure 1. System framework
{: style="text-align: center; font-size:0.7em;"}

### 2.1. System Framework

전체 시스템은 **고장 감지**와 **고장 진단** 두 부분으로 나눌 수 있으며, 고장 감지는 또 다시 상태 고장 감지와 궤적 편차 감지로 나누어집니다. **상태 고장 감지**는 속도나 각속도와 같은 상태들이 정상인지 판단하며 **궤적 이탈 감지**는 궤적의 이탈을 확인합니다. 고장이 감지되면 고장 진단 시스템이 작동하여 고장을 **Moving Alarm** 혹은 **Steering Alarm**로 분류합니다. 각 부분은 다른 접근법이 적용되었지만 서로 협업하게 됩니다. SVM은 상태 고장 감지에 적용되었고 편차 분포는 궤적 이탈 감지를 체크합니다. 그리고 고장 확률 표시기는 신경망을 통해 적용됩니다.

### 2.2. State Fault Detector

자율주행차의 수집되는 데이터는 대부분 정상 데이터입니다. 따라서 데이터 불균형 문제가 있습니다. 이를 해결하기 위해서 많은 One-Class 분류 모델이 제안되었으며 이 논문에서는 One-Class SVM이 사용됩니다. 다른 연구에서 One-Class SVM이 작은 샘플의 데이터에 대해서도 고장 감지를 잘 수행된다고 말하고 있습니다. 이 논문에서는 차의 상태 $X$를 $X = [v,w]^\text{T}$로 정의했으며 $v$와 $w$는 각각 속도와 각속도를 의미합니다. SVM의 커널 함수로는 RBF를 사용했습니다. 

### 2.3. Trajectory Deviation Detector

실제 적용에서 단일 측정은 노이즈 및 정확하지 않은 방해로 오류를 일으킬 수 있습니다. **Kalman filter**는 여러 측정 방법을 시간에 걸쳐 사용하며, Kalman filter를 통과한 알려지지 않은 변수들에 대한 측정은 더 정확한 경향을 보입니다. 고장 감지의 관점에서 Kalman filter는 모델 기반의 방법입니다. 이 논문에서는 차량 운동학적 모델에 기반해서 디자인되어 자율주행동안  차량의 현재 위치를 예측합니다. 실제 값과 예측 값의 잔차를 비교함으로써 궤적 이탈 여부를 확인할 수 있습니다.

#### Vehicle Kinematic Model

![image](https://user-images.githubusercontent.com/35906602/141429786-f8377118-a661-4f77-ab62-1d0051654ffc.png){: width="500"}{: .align-center} 

Figure 2. Schematic for a linear kinematic vehicle bicycle model
{: style="text-align: center; font-size:0.7em;"}


앞서 말한 것과 같이 차량 운동학적 모델은 Kalman filter를 디자인하기 위한 바탕이 됩니다. 여기서는 차량 자전거 모델을 선택하여 자율주행차의 운동학적 특징을 묘사합니다. 선형 운동학적 차량 저전거 모델을 위해서 차량의 가로*longitudinal*, 세로*lateral*, 그리고 요*yaw* 운동이 고려 되어야 합니다.

#### Kalman Filter Observer Designing

Kalman Filter는 시스템 상태 예측에서 널리 쓰이며, 더 복잡하게 확장된 많은 모델이 있습니다. 하지만 운동학적 모델이 선형이며 가우시안 노이즈를 가정하고 있기 때문에 더 복잡한 모델을 사용할 필요가 없으며 따라서 모델의 디자인을 간단하게 하고 계산량을 줄일 수 있습니다. Kalman filter는 재귀적 함수로, 이는 현재 상태에 대한 예측을 위해 오직 바로 전의 상태와 지금의 측정량만 필요하다는 것을 의미합니다. 이 알고리즘은 일반적으로 예측과 상태 갱신의 두 스텝으로 나누어집니다.

#### Residuals Distribution Inference

![image](https://user-images.githubusercontent.com/35906602/141432149-7f121b37-bb68-48ce-8bb4-42a1bf0fde29.png){: width="600"}{: .align-center} 

Figure 3. Explanation to different types of test period
{: style="text-align: center; font-size:0.7em;"}


상태 고장 감지와 비교해서 궤적 이탈 감지는 더 많은 샘플이 필요합니다. 상태 고장이 순간적인 비정상성을 강조한다면 궤적 고장은 움직이고 있는 상황의 고장을 강조합니다. 해당 분포의 추론을 위한 통계적 방법론은 많이 있으며, 이 논문에서는 **Jarque-Bera test**가 사용됩니다. Jarque-Bera test는 왜도 및 첨도 테스트를 기반으로 하는 접근 방식이기 때문에 이상치에 민감합니다. 또한 잔차 데이터는 대칭적인 긴 꼬리 형태를 보여주는데 Jarque-Bera test는 이에 적합한 방식입니다. 

### 2.4. Diagnosis System: Membership Function Training Using Neural Network

만약 고장이 이미 감지되었다면 그 원인을 규명해야 합니다. 이를 위해서 이 논문에서는 퍼지 시스템을 적용합니다. Membership function은 퍼지 시스템의 핵심입니다. 퍼지 시스템은 살펴보고 있는 변수가 변화함에 따라 확률 분포가 어떻게 움직이는지 묘사합니다. 퍼지 시스템의 목적은 각 요소들이 고장으로 이어질 확률을 나타내는 것입니다. 이를 통해 자율주행차의 고장을 **Moving Alarm** 혹은 **Steering Alarm**으로 나타냅니다. 

상태의 고장을 분류한 후에는 정확히 어떤 서브 시스템이 고장났는지를 살펴보게 됩니다. 서브 시스템은 동적 시스템, 브레이크 시스템, 조향 시스템, 혹은 인식 시스템 등이 있습니다. 하지만 이 논문에서는 여기까지 다루지는 않으며 어떤 유형의 고장이 일어났는지의 확률을 나타내는 것에 집중합니다.

![image](https://user-images.githubusercontent.com/35906602/141433889-b5647556-8cdc-4979-9340-085a7d335a78.png){: width="500"}{: .align-center} 

Figure 4. Shape of the membership function
{: style="text-align: center; font-size:0.7em;"}


$$\begin{aligned}
M(v) &= \frac{1}{1+e^{-\sigma_v(v-\mu_v)}} \\
M(w) &= \frac{1}{1+e^{-\sigma_w(w-\mu_w)}}
\end{aligned}$$

속도와 각속도의 Membership function은 각각 위와 같이 나타냅니다. 처음의 Membership function은 사전 지식을 활용해 주어졌으며 임계치는 속도 혹은 각속도의 변화에 따라 변화합니다. 즉 상태 벡터에 따라 역동적으로 변화함을 의미합니다. 

![image](https://user-images.githubusercontent.com/35906602/141433970-136712e4-5541-4aa7-8d4b-8d553f6c2312.png){: width="500"}{: .align-center} 

Figure 5. Structure schematic of the neural network
{: style="text-align: center; font-size:0.7em;"}

따라서 주관성을 줄이고 실제 상황을 더 잘 반영하도록 하기 위해 membership function을 업데이트하는 신경망을 설계했습니다. 구조는 위와 같습니다. 입력층은 상태 벡터를 Membership function으로 보내며, 나머지는 기본 신경망과 같습니다. 

### 2.5. Diagnosis System: Black Box Testing for Parameters Update

앞 단계의 신경망은 입력 데이터가 정상인지 아닌지만 판단할 수 있으며, 초기 membership function의 매개변수들을 갱신하지는 않습니다. 각 타입의 고장 확률을 얻기 위해서 membership function의 매개변수들을 갱신해야하며 이를 위해서 **black box test**가 적용 되었습니다. 학습된 네트워크는 black box로 여겨집니다. 앞서 살펴보았던 공식들을 활용해서 속도와 각속도를 각각 극단적인 상황을 가정하여 주고 최적의 파라미터로 membership function을 업데이트 했습니다. 자세한 과정은 논문을 참조해주세요. 

## 3. Methodology Validation and Analysis

제안된 방법을 평가하기 위해서 자율주행차 플랫폼 Xinda의 GPS 데이터를 사용했습니다. 데이터의 일부는 정상 데이터, 일부는 비정상 데이터입니다. 결과는 다음과 같습니다.

### 3.1. Data Preprocessing

데이터는 0.01초 간격으로 수집된 자율주행차의 경로, 속도, 각속도, 요 등으로 이루어 졌습니다.

### 3.2. Result and Analysis for One-Class SVM

![image](https://user-images.githubusercontent.com/35906602/141436167-827c6ed1-74b9-4682-9938-72a3ec8c33b9.png){: width="500"}{: .align-center} 

Figure 6. Decision boundary found by One-Class SVM
{: style="text-align: center; font-size:0.7em;"}

상태가 정상인지 감지하기 위해 One-Class SVM을 적용했으며 시각화한 결과는 위와 같습니다. 빨간색 영역이 학습된 경계이며, 과적합 됐기 때문에 계속해서 비정상 알람을 산출하게 됩니다. 따라서 더 부드러운 영역을 사용해 일반화 했으며 그림에선 노란색 점선으로 표시됩니다. 기준은 운동학적, 동적 지식을 사용해 임의로 정했습니다.

2초동안의 데이터가 경계선 밖으로 벗어나면 고장이라고 판단하게 됩니다. 노란색 점은 학습 데이터에 가우시안 노이즈를 부여한 결과인데 잘 측정하고 있습니다. 빨간색 점은 kaggle에 공개된 자율주행차의 데이터입니다. 이 모델이 과적합됐다고 볼 수도 있지만, 다른 시스템은 다르게 행동하게 되므로 Xinda에 맞춘 모델과는 안 맞을 수 있습니다.

![image](https://user-images.githubusercontent.com/35906602/141437151-b15715e3-70c0-4932-8348-1c17db0d9e5e.png){: width="500"}{: .align-center} 

Figure 7. Overfitting validation of the One-Class SVM method
{: style="text-align: center; font-size:0.7em;"}

분류기의 성능을 평가하기 위해 고장 데이터를 정상 데이터에 삽입했습니다. 결과는 위와 같습니다. 

![image](https://user-images.githubusercontent.com/35906602/141436980-a916df75-998f-482e-8e00-34877940f832.png){: width="500"}{: .align-center} 

Figure 8. Overfitting validation of the One-Class SVM method
{: style="text-align: center; font-size:0.7em;"}

One-Class SVM 분류기는 학습 데이터에 과적합된 결과일 수 있습니다. 결과는 위와 같습니다. 따라서 결정 경계를 최적화하기 위해 더 많은 다양한 데이터를 수집하거나 경계를 넓히는 등의 기법을 사용할 수 있습니다.

### 3.3. Result and Analysis for Residuals Distribution Inference

자율주행차의 현재 위치를 예측하기 위해 Kalman filter가 적용 되었습니다. 이론적으로는 현재 위치와 예측값의 잔차는 0을 중심으로 한 가우시안 분포를 가져야 합니다. 반복된 실험 결과 이 가정이 옳다는 점을 밝혔습니다. 만약 이 가정이 사실이 아니라면 다른 치명적인 노이즈가 있다는 의미이므로 이 방법을 적용할 수 없습니다.

### 3.4. Membership Function Parameters Update

![image](https://user-images.githubusercontent.com/35906602/141438689-abe1cd45-97cd-4ae4-8dde-5a7bb9db8ecf.png){: width="600"}{: .align-center} 

Table 1. Some test results of the faults probability for each factor
{: style="text-align: center; font-size:0.7em;"}

고장이 감지되면 각 고장의 확률을 밝혀내야 합니다. 안전의 관점에서 속도와 각속도의 값이 클수록 차량의 상태가 비정상일 확률이 높습니다. 그러므로 속도 $v$와 각속도 $w$의 membership function은 앞서 살펴봤던 형태를 보여야 합니다. Membership function의 초기 매개변수들은 사전 지식을 활용해 정해졌습니다. 하지만 주관적일 수 있기 때문에 신경망을 통해 이를 학습했습니다. 학습된 membership function을 통해 Xinda 데이터에서 구한 고장 확률은 위와 같습니다.

![image](https://user-images.githubusercontent.com/35906602/141438993-1e500258-f26d-4cf5-a334-b48b439aa35a.png){: width="600"}{: .align-center} 

Figure 9. Boundary Fault Probability Test
{: style="text-align: center; font-size:0.7em;"}

앞선 모델들을 공개된 데이터에 적용한 결과를 시각화하면 위와 같습니다. 

## 4. Conclusion

이 논문의 목적은 고장 감지 및 진단 시스템을 디자인하는 것입니다. 일반적으로는 이를 위해 모델 기반 접근법, 신호 기반 접근법, 그리고 지식 기반 접근법이 있습니다. 자율주행차의 실제 상황을 고려해 이 논문에서는 복합적 접근방식을 제안합니다. 차량의 상태 고장을 감지하기 위해서는 One-Class SVM이 사용됐으며, 실제 값과 예측 값 사이의 잔차를 나타내기 위해선 Kalman filter가 적용됐습니다. 그리고 마지막으로는 각 고장 유형의 확률을 나타내기 위해 membership function을 갱신하는 방식의 신경망을 포함한 퍼지 시스템이 적용됐습니다. 

이후에는 fault isolation을 고려할 수 있습니다. 이 논문에서는 'Moving Alarm'과 'Steering Alarm'으로 분류했는데, 어느 서브 시스템에서 고장이 발생했는 지는 알 수 없습니다. 