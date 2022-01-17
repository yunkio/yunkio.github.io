---
date: 2021-12-08
title: "[Paper Review] Fault Diagnosis of an Autonomous Vehicle With an Improved SVM Algorithm Subject to Unbalanced Datasets"
categories: 
  - Paper Review
tags: 
  - 딥러닝
  - Fault Detection
  - 논문 리뷰
toc: true  
toc_sticky: true 
---
# Paper contents

Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress

R. Wu, E. Keogh

IEEE Transactions on Knowledge and Data Engineering, 2021.

https://arxiv.org/abs/2009.13807

## 0. Abstract

시계열 데이터 이상 감지는 데이터 과학에 있어서 매우 중요한 영역 중 하나이고, 특히 최근 딥러닝의 발전으로 크게 주목받고 있습니다. 최근에 쓰여진 대부분의 논문들은 Yahoo, Numenta, NASA 등의 시계열 벤치마크 데이터셋을 활용했습니다. 이 논문에서는 앞서 제시된 벤치마크 데이터셋들의 문제점을 지적합니다. 각각의 예는 최소 1개에서 4개까지의 문제점을 갖고 있습니다. 따라서 최근에 이 데이터들을 활용해 제시된 대부분의 알고리즘들은 신뢰할 수 없으며 발전은 환상일 수 있다고 말하고 있습니다.

## 1. Introdution

최근 딥러닝의 발전으로 인해 5년간 시계열 데이터 이상 탐지에 대한 관심이 폭발적으로 증가하였으며, 각종 데이터 과학 관련 컨퍼런스에 매년 최소 1~2편 이상의 논문이 게재되었습니다. 이 논문들은 대부분 Yahoo, Numenta, NASA, OMNI 등의 공개된 벤치마크 시계열 데이터셋을 활용해 쓰여졌습니다. 하지만 이 데이터셋들은 1개에서 4개까지의 큰 문제점들을 가지고 있습니다. 그 문제점들은 *triviality*, *unrealistic anomaly density*, *mislabeled ground truth*, *run-to-failure bias* 입니다. 이 문제점들로 인해 현재 제시된 대부분의 이상 탐지 알고리즘들은 신뢰할 수 없으며, 이 분야의 발전 상황은 환상일 수 있습니다.

가령 *KPI-TSAD: A Time-Series Anomaly Detector for KPI Monitoring in Cloud Applications* 라는 논문에서 제시된 알고리즘은 굉장히 복잡한 구조를 사용하여 Yahoo 데이터셋의 일부에서 0.9가 넘는 정확도를 달성했다고 말하고 있지만, 몇 분의 노력과 한 줄의 코드만으로 비슷한 성과를 거둘 수 있습니다. 

여기서 말하는 '한 줄의 코드' 논쟁을 미리 다뤄보겠습니다. 최근 연구에서 딥러닝을 활용해 모기의 품종을 97.8%의 정확도로 분류해내는 알고리즘이 발표되었습니다. 하지만 만약 누군가가 1,185개 이미지의 원본 데이터를 내려받아서 한 줄의 코드로 100% 정확도로 이를 분류해낼 수 있다고 가정해봅시다. 그렇다면 우리는 두 가지 주장을 할 수 있습니다.

* 데이터 자체에 결함이 있을 것입니다. 가령 한 품종은 jpeg, 다른 품종은 gif의 포맷으로 되어있을 수 있습니다.
* 해당 논문의 방법론은 신뢰할 수 없습니다. 한 줄의 코드로 분류되지 않는 다른 데이터셋에 시험해서 비슷한 정도의 성능이 나온다면 그때는 신뢰할 수 있을 것입니다.

한 줄의 코드로 좋은 성능을 보일 수 있다는 것이 논문이 아무런 기여가 없다는 의미는 아닙니다. 하지만 최소한 한 줄로 해결이 되지 않는 새로운 데이터셋에 해당 논문의 방법론을 다시 시험해볼 필요가 있다고 주장 할 수는 있습니다.

또한 이 논문이 해당 데이터셋을 도입한 사람들을 비판하는 것은 아닙니다. 단지 학계는 이 데이터셋들의 한계에 대해 정확히 인지하고 이를 극복할 수 있는 연구를 끈임없이 진행해야 합니다.

## 2. A Taxonomy of Benchmark Flaws

### 2.1 Related Work

  이 논문의 연구는 대부분 공개된 시계열 벤치마크 데이터셋인 Yahoo, Numenta, NASA, OMNI 등을 대상으로 이루어졌습니다. 몇몇 논문은 공개된 데이터셋에 더해 비공개 데이터셋을 대상으로 진행되었으며 이 데이터셋들은 아무런 정보도 주어지지 않는 경우가 대부분입니다. 그러므로 이 논문에서는 그러한 비공개 데이터셋에 대한 논의는 이루어지지 않습니다.
  
  공개된 데이터셋 중 하나에 잘 적용이 되었다는 것은 곧 알고리즘이 유용하다는 강한 가정을 내포하고 있습니다. 실제로 많은 논문이 명시적으로 이 가정을 언급하고 있으며, 이 가정을 비판적으로 다룬 논문은 찾기 힘듭니다. 뒤의 4개 섹션들을 통해 공개 데이터셋이 알고리즘을 비교하거나 시게열 이상 탐지의 발전 상황을 측정하는 데 적합하다는 가정에 의문을 제기할 것입니다.

### 2.2 Trivaility

벤치마크 데이터셋의 가장 큰 문제는 너무 간단해서 좋은 성능을 보이는 것이 무의미하다는 것입니다. *Triviality* 라는 단어를 다시 정의해보겠습니다.

> 정의 1 : MATLAB의 기본 라이브러리를 활용해 한 줄로 해결이 가능하다면 그 문제는 *trivial* 합니다. 이때 사용되는 함수는 mean, max, std, diff 등의 기본 함수가 있습니다.

이 정의는 완벽하지는 않습니다. MATLAB은 복잡한 표현이 가능하므로 한 줄의 코드라고 해도 굉장히 복잡해질 수 있기 때문입니다. 게다가 모델 훈련을 통해 얻어낼 수 있는 상수가 포함될 수도 있습니다. 마지막으로 이상 탐지의 요점은 문제를 해결할 수 있는 자동화된 알고리즘을 제공하는 것이지만, 한 줄의 코드로 해결하기 위해서는 사람이 개입해야 합니다.

그럼에도 불구하고 이 간단한 정의는 이 논문의 주장의 요점을 잘 나타냅니다. 만약 이상치를 식별해낼 수 있는 간단한 코드를 빠르게 작성할 수 있다면 복잡한 알고리즘을 활용할 이유가 없다고 주장할 수 있습니다.

가령 RNA가 단백질로 코딩되었는지 아닌지, 혹은 영화 리뷰가 긍정적인지 부정적인지를 분류하는 작업을 가정해보면 이 문제가 명료해집니다. 이 문제들은 딥러닝의 발전과 함께 상당한 진전을 보인 문제들입니다. 하지만 이러한 생물학적 정보나 자연어 데이터는 아무리 복잡한 코드더라도 한 줄의 코드로는 결코 해결할 수 없습니다. 

![image](https://user-images.githubusercontent.com/35906602/149700332-3048b9b9-dd09-4e76-931a-d03316b04c73.png){: width="600"}{: .align-center} 

Figure 1. Dimension 19 from SDM3-11 dataset
{: style="text-align: center; font-size:0.7em;"}

요점을 잘 나타내기 위해 예시를 들어보겠습니다. OMNI 데이터셋의 한 예시이며 다차원 데이터이지만 여기서는 하나의 차원만을 예로 들겠습니다. 이 문제를 해결할 수 있는 한 줄의 코드는 매우 많습니다. 위에서는 3개의 예를 보이고 있습니다. 몇가지 제기할 수 있는 이의에 대해 답해보도록 하겠습니다.

* 모든 한 줄의 코드는 파라미터를 가지고 있다.
$\rightarrow$ 맞습니다. 하지만 대부분의 딥러닝 알고리즘 역시 다수의 파라미터들을 가지고 있습니다. 또한 위와 같은 결과는 설정한 파라미터에 민감하지 않습니다.

* 차원 선택이 편의적이다.
$\rightarrow$ 위의 예에서는 38개의 차원 중에 비교적 어려운 차원을 선택했습니다. 다른 차원은 해결이 더 쉽습니다.

* 문제 선택이 편의적이다.
$\rightarrow$ 이 데이터에 포함된 28개의 예 중에서 대부분은 저만큼 한 줄의 코드로 해결하기 쉽습니다.

* 문제를 한 줄로 해결할 수 있다는 것이 곧 다른 알고리즘이 유용하지 않다는 의미는 아닙니다.
$\rightarrow$ 맞습니다. 이 논문도 그런 주장을 의도하지는 않았습니다.

![image](https://user-images.githubusercontent.com/35906602/149732282-0326196e-0a69-4556-a033-b62718c936b4.png){: width="600"}{: .align-center} 

Figure 2. The Numenta Art Increase Spike Density datasets
{: style="text-align: center; font-size:0.7em;"}

Numenta, NASA 데이터셋에도 같은 문제가 존재합니다. 특히 NASA의 경우 반 이상의 데이터가 역동적인 시계열로 구성되어있기 때문에 *trivial*하지 않지만, 나머지의 경우 매우 간단하게 해결이 가능합니다. 다시 말하면 약 10%의 데이터만이 도전적이며, 그 예제들조차 딥러닝까지는 필요하지 않은 경우가 대부분입니다.

![image](https://user-images.githubusercontent.com/35906602/149736979-162409f4-2184-4271-a2b3-0cde86707acf.png){: width="600"}{: .align-center} 

Figure 3. Yahoo A1-Real1 dataset.
{: style="text-align: center; font-size:0.7em;"}

Yahoo의 경우 가장 많이 인용된 데이터입니다. 실제 데이터와 인공 데이터가 같이 있습니다. 그 중 첫번째는 현실의 데이터로 이루어져 있으며 사람의 눈으로 볼 때는 가장 난이도가 있어보이는 예제입니다. 하지만 이 역시도 한 줄의 코드로 해결이 가능합니다. Cherry-picked 된 결과가 아닌지 의심할 수 있으므로 전체 Yahoo 데이터에 대해서 살펴보면, 야후 데이터에는 367개의 시계열 데이터가 있으며, 이 중 316개가 3개의 파라미터를 포함한 한 줄의 코드만으로 해결이 가능합니다. 심지어는 반절 이상의 데이터가 단 하나의 숫자만으로 해결이 됩니다. 섹션 2.4에서 언급할 라벨링 오류를 고려하면 거의 완벽하게 해결이 가능하다는 의미가 됩니다.

### 2.3 Unrealistic Anomaly Density

이 문제는 세 가지의 오류를 포함합니다.

* 몇몇 예는 시험 데이터의 반 이상이 이상치로 라벨된 연속된 지역을 포함합니다. 예를 들면 NASA 데이터의 D-2, M-1, M-2가 있습니다. 
* 또 다른 예는 이상치로 라벨된 구역이 많습니다. 예를 들면 SDM exemplar machine-2-5의 경우 21개의 짧은 이상치 구간을 포함합니다.
* 몇몇 예에서는 이상치들이 서로 너무 가깝습니다. 예를 들면 위의 Figure 3에서 두 이상치사이에는 하나의 정상 데이터포인트만 존재합니다.

이러한 문제들로 인해 몇몇 이슈가 발생합니다. 첫 번째로 *분류* 와 *이상 탐지* 의 경계를 흐리게 합니다. 실제 환경에서는 이상치의 사전 확률은 0에 가깝습니다. 데이터셋의 반절이  이상치를 포함하고 있는건 실제 환경에 맞지 않으며, 심지어 많은 알고리즘은 사전 확률에 민감합니다. 

두 번째로 비현실적인 밀도는 평가 알고리즘을 결정하는 데에 혼동을 주기 쉽습니다. 가령 하나의 시계열에 총 10개의 이상치가 존재했다고 가정 했을 때, 뒤의 이상치보다 앞의 이상치를 더 좋게 평가하는 것이 현실에 더 적합합니다. 하지만 이러한 결과를 반영하는 연구는 거의 찾기 힘듭니다.


### 2.4 Mislabeld Ground Truth

![image](https://user-images.githubusercontent.com/35906602/149742303-c0f24dc6-fa79-425f-8dac-64712b19f824.png){: width="600"}{: .align-center} 

Figure 4. YAHOO A1-Real32
{: style="text-align: center; font-size:0.7em;"}

모든 벤치마크 데이터셋은 라벨이 잘못된 데이터가 존재합니다. 예를 들면 위 데이터에서 A를 이상치로 감지하면 정답이지만 B를 이상치로 감지하면 오답입니다. 같은 값으로 이루어진 저 선에서 무슨 기준으로 라벨을 구분했는지는 납득하기 어렵습니다. 

![image](https://user-images.githubusercontent.com/35906602/149742890-8169bd18-ee2d-4f60-b981-1adc44ea6a4c.png){: width="600"}{: .align-center} 

Figure 5. The Yahoo A1-Real46 dataset with its class labels and Overlaying two snippets allows a one-to-one comparison between the region of C and D.
{: style="text-align: center; font-size:0.7em;"}

다른 예에서도 마찬가지입니다. 위의 예에서 C와 D는 거의 같은 양상을 보이고 있지만 C는 이상치로, D는 정상으로 라벨되어 있습니다.

![image](https://user-images.githubusercontent.com/35906602/149742921-237bc1b6-094d-4cf9-8b3b-a6aa0344693b.png){: width="600"}{: .align-center} 

Figure 6. An excerpt from Yahoo A1-Real47
{: style="text-align: center; font-size:0.7em;"}

위 예의 경우 E와 F 둘다 이상치로 라벨되어 있지만 F가 진짜 이상치인지는 더 고민할 필요가 있어 보입니다. F와 같은 예는 위 그림에서도 많이 보이고 있습니다. 그 어떤 통계치를 살펴보더라도 F와 다른 경우를 구분할만한 통계값은 나타나지 않습니다.

![image](https://user-images.githubusercontent.com/35906602/149742970-7c712fb8-2524-4485-853d-747ab29371db.png){: width="600"}{: .align-center} 

Figure 7. An excerpt from Yahoo A1-Real67
{: style="text-align: center; font-size:0.7em;"}

위 Figure 7의 경우 합리적인 이유 없이 라벨링이 좁습니다. 자율주행차를 예로 들어 빠르게 움직이는 차가 부딪혀서 뒤집히는 경우를 생각해보겠습니다. 이러한 경우 충돌 이후의 모든 구간이 비정상으로 라벨링 되는 것이 옳습니다. 하지만 라벨링을 지나치게 정확하게 할 경우 이러한 점이 고려되지 않고 충돌 당시만 비정상으로 라벨링 될 수 있습니다. Figure 7에서 보이는 데이터 역시 50개의 반복되는 사이클 후에 저 시점에서 값의 움직임이 다르게 나타납니다. 이 경우 정상과 이상치를 반복하여 라벨링하는 것은 합리적이지 못 합니다. 위와 같은 문제로 인해 이상 탐지기는 이상치로 라벨 된 구역에서 벗어난 곳을 이상치로 감지할 때 오답으로 처리됩니다. 

![image](https://user-images.githubusercontent.com/35906602/149811707-f19ea64b-16d6-4662-b242-4e4cba09391e.png){: width="600"}{: .align-center} 

Figure 8. Numenta’s NY Taxi dataset.
{: style="text-align: center; font-size:0.7em;"}

라벨링이 지나치게 주관적인 경우도 있습니다. Numenta의 NY Taxi 데이터셋의 경우 기존에는 라벨이 오직 5개만 달려 있습니다. 하지만 그 밖에도 많은 이상치, 즉 택시의 수요가 비정상적으로 증가하는 날이 존재합니다. 알고리즘이 매우 잘 작동해서 저런 경우들을 잘 탐지해내는 경우 오히려 False-Positive로 판정되어 낮은 성능으로 평가받게 됩니다.

![image](https://user-images.githubusercontent.com/35906602/149812692-a03b540d-c4c3-4157-bbd2-8d1b8dc91709.png){: width="600"}{: .align-center} 

Figure 9. Three snippets from Mars Science Laboratory: G-1.
{: style="text-align: center; font-size:0.7em;"}

위 NASA 데이터의 경우 제일 위의 경우는 이상, 밑의 두 경우는 정상으로 라벨링 되어 있습니다. 오픈된 정보를 통해서 접근할 수 있는 숨겨진 정보로 인해 저렇게 라벨이 됐을 가능성도 있습니다. 하지만 맨 위의 경우만 이상치로 탐지해내는 알고리즘과 세 경우를 모두 탐지해내는 알고리즘 중 무엇이 더 성능 좋은 알고리즘인지는 이견의 여지가 있을 것입니다.

### 2.5 Run-to-failure Bias

![image](https://user-images.githubusercontent.com/35906602/149813210-52f78bb0-a489-4c82-b903-744b4d140e01.png){: width="600"}{: .align-center} 

Figure 10. The locations of the Yahoo A1 anomalies
{: style="text-align: center; font-size:0.7em;"}

이는 특히 Yahoo 데이터셋 (그리고 NASA 데이터셋)에서 발생하는 문제입니다.  Yahoo 데이터셋의 경우 대부분의 이상치가 끝 부분에 몰려 있습니다.  학습 데이터가 이와 같이 편향된 경우 이를 통해 학습하는 알고리즘 역시 편향된 결과를 학습하게 됩니다.

### 2.6 Ssummary of Benchmark Flaws

이와 같이 현재 존재하는 벤치마크 데이터셋은 심각한 결함이 존재합니다. 예를 들면 이러한 데이터셋을 바탕으로 F1 score를 1.0을 달성하는 알고리즘이 있다면 그 알고리즘은 완벽한지에 대해서는 이견의 여지가 있을 수 있습니다. 특히 라벨 자체가 잘못된 경우가 산재하므로 오히려 우리는 실험의 오류가 있지 않은지 의심해야 할 것입니다. 반면, F1 score 0.9를 달성한 알고리즘이 있다고 가정하겠습니다. 앞서 살펴본 *triviality*로 인해 낡은 알고리즘으로도 달성할 수 있는 점수일 것 같습니다. 따라서 현재 존재하는 벤치마크 데이터 셋을 통한 성능 검증으로는 알고리즘의 유용성을 입증할 수 없습니다.

## 3 Introducing the UCR Anomaly Archive

이 논문은 위와 같이 벤치마크에 존재하는 여러 문제점을 반면교사 삼아 새로운 데이터셋을 제안합니다. 하지만 여전히 일부 데이터는 한 줄의 코드로 해결이 가능합니다. 여기엔 두 가지 이유가 존재합니다. 첫 번째로 해결이 쉬운 문제에서부터 어려운 문제까지 넓은 범위의 데이터를 포함하고자 했습니다. 두 번째는 실제 데이터에도 한 줄의 코드로 해결이 간단한 문제가 많이 존재합니다. 

또한 이 논문에서는 하나의 시계열에는 하나의 이상치가 존재하는 것이 이상적이라고 주장합니다. 뒤에서는 데이터셋에 대해 소개할 것입니다.

### 3.1 Natural Anomalies Confirmed Out-of-Band

![image](https://user-images.githubusercontent.com/35906602/149815888-3fe368f6-a91a-460c-a23c-24a0894c0097.png){: width="600"}{: .align-center} 

Figure 11. UCR_Anomaly_BIDMC1_2500_5400_5600, a dataset from archive.
{: style="text-align: center; font-size:0.7em;"}

처음 2,500개의 데이터 포인트들은 학습을 위해 설계되었으며 이상치는 5,400부터 5,600까지 존재합니다. 이상치는 사소해보일 수 있으나 이 데이터는 심전도 데이터이기 때문에 저 사소한 이상도 치명적일 수 있습니다. 

### 3.2 Synthetic, but Highly Plausible Anomalies

![image](https://user-images.githubusercontent.com/35906602/149816564-dffc7c66-b18a-4eee-9136-4784babc997f.png){: width="600"}{: .align-center} 

Figure 12. UCR_Anomaly_park3m_60000_72150_72495, a dataset from Archive.
{: style="text-align: center; font-size:0.7em;"}

또한 그럴듯한 인공 데이터를 만들었습니다. 이상치가 존재하지 않는 데이터에 이상치를 삽입한 것입니다. 더욱 신뢰할 수 있는 데이터를 만들기 위해 절름발이 증상을 앓고 있는 개인의 데이터를 활용했습니다. 정상적인 오른발 사이클 중 한번을 약하게 걷는 왼발의 사이클로 대체했습니다. 실제로 환자가 아니더라도 보행 중에 한 걸음을 약하게 딛는 경우는 매우 많으므로 자연스러운 데이터의 형성이 가능합니다. 측정에 사용된 장치는 길이에 제한이 있어 보행자가 걷다가 방향을 바꿀 때 속도를 감소하게 되는데 이러한 현상은 데이터에도 나타나고 있습니다. 하지만 학습 데이터 및 시험 데이터 양 쪽에 이러한 데이터를 포함하고 있으므로 문제는 없습니다.
이러한 방식으로 데이터를 만들 때 너무 쉽거나 혹은 너무 어렵지 않도록 많은 주의를 기울였습니다. 실제로 이 예의 경우에도 10명 중에 9명은 육안으로 이상치를 식별할 수 있었습니다.

## 4 Recommendation

이 논문에서는 학계에 여러 의견을 제안하고 있습니다.

### 4.1 Existing Datasets should be Abandoned

현재 널리 쓰이는 벤치마크 데이터셋의 경우 결함이 너무 많기 때문에 사용이 중지되어야 합니다. 근본적인 문제로 인해 수정도 불가능에 가깝습니다. 또한 이 데이터셋만을 사용해서 알고리즘을 평가하거나 비교한 논문들 역시 다시 평가받아야 할 것입니다.

### 4.2 Algorithms should be Explained with Reference to their Invariances

최근 몇년간 시계열 **분류**가 더 많은 발전이 있었던 것으로 보입니다. 시계열 분류 학계에서는 **invariances** *불변성*에 대해 많은 논의가 이루어지고 있습니다.  여기에는 amplitude scaling, offset, occlusion, noise, linear trend, warping, uniform scaling 등이 포함됩니다. 이러한 불변성을 활용하는 것은 큰 도움이 될 수 있습니다. 
반면 이상 탐지 분야에서는 이러한 논의가 거의 이루어지지 않습니다. 저자들은 대부분 자신들의 알고리즘이 정확히 어떤 상황에서 사용되어야 하는지 언급하지 않습니다. 따라서 저자들은 중요한 통계적 불변성에 대해서는 언급해주어야 합니다.

![image](https://user-images.githubusercontent.com/35906602/149818671-80578dae-dcc4-447e-8be1-8a17860cd1a1.png){: width="600"}{: .align-center} 

Figure 13. One minute of an electrocardiogram with an obvious anomaly that is correctly identified by two very different methods
{: style="text-align: center; font-size:0.7em;"}

같은 시계열 데이터를 두 개의 다른 알고리즘으로 확인한 결과입니다. 위의 결과의 경우 두 알고리즘 모두 잘 식별해내고 있는 것으로 보입니다. 반면 같은 데이터에 가우시안 노이즈를 추가한 경우 Telemanom은 이상치를 탐지해내지 못 하고 있으며 Discord는 잘 탐지해내고 있습니다. 이는 곧 특정 알고리즘은 노이즈가 있는 환경에서 더 잘 작동한다는 의미가 됩니다. 

### 4.3 Visualize the Data and Algorithms Output

시계열 이상 탐지의 많은 논문들은 결과를 시각화한 그래프가 매우 적습니다. 시계열 분석은 본질적으로 시각적인 도메인입니다. 단순히 보이기 위해서가 아니라, 실제 연구에도 이를 반영해야 합니다. 많은 연구자들이 실제 시계열을 살펴보지 않고 그저 블랙 박스에 데이터를 집어넣고 평가 지표를 보는데 그칩니다. 이 논문에서 제시한 문제점들도 그저 실제 데이터 및 결과를 살펴보기만 하면 충분히 발견할 수 있는 문제들이였습니다.

### 4.4 A Possible Issue with Scoring Functions

이 논문에서는 데이터셋의 문제에 대해서만 다루었지만, 평가 지표에 문제를 제기한 연구자들도 있었습니다. 가령 위의 Figure 13에서 Telemanom은 Discord에 비해 더 앞선 시점에서 이상치 점수가 가장 높게 나옵니다. 이러한 경우를 똑같이 평가할 경우 알고리즘의 학습 및 평가에 편향이 발생할 수 있습니다. 

### 4.5 The “deep learning is the answer” Assumption should be Revisited

최근의 많은 논문들은 딥러닝이 이상치 탐지에 있어 최적의 방법론이라는 가정을 하고, 가장 최선의 딥러닝 방법을 찾는 것을 목적으로 합니다. 딥러닝이 다른 방법론에 비해 경쟁력 있다는 사실은 충분히 설득력 있습니다. 하지만 위에서 언급된 내용을 고려하면 딥러닝이 다른 더 간단한 알고리즘들을 능가한다는 강력한 재현 가능한 증거를 제시하는 논문은 아직 존재하지 않습니다. 연구자들은 존재하는 모든 방법들을 고려할 것을 권장합니다.

## 5 Conclusion

이 논문에서는 현재 일반적으로 사용되는 이상 탐지를 위한 벤치마크 데이터셋은 결함이 많기 때문에 알고리즘을 평가하거나 비교하기엔 적합하지 않다고 지적합니다. 이에 그치지 않고 언급된 문제점들을 해결한 새로운 벤치마크 데이터 셋을 제안합니다. 이에 대한 더욱 더 많은 논의가 필요할 것입니다.