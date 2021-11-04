---
date: 2021-11-04
title: "[Tensorflow] Chapter 8. 사전 훈련된 모델 다루기"
categories: 
  - 시작하세요! 텐서플로 2.0 프로그래밍
tags: 
  - Tensorflow
  - 딥러닝
  - Transfer Learning
toc: true  
toc_sticky: true 
---
*본 글은 '시작하세요! 텐서플로 2.0 프로그래밍' 을 바탕으로 작성되었습니다.*

# Chapter 8. Pre-trained

딥러닝이 발전함에 따라 네트워크도 점점 커졌습니다. 연구자들은 자신이 만든 **사전 훈련** *pre-trained* 된 모델을 공유해 다른 사람들이 쉽게 내려받을 수 있게 합니다. 이렇게 얻은 모델은 그대로 사용하거나 **전이 학습** *Transfer Learning*이나 **신경 스타일 전이** *Neural Style Transfer*같은 방법으로 재가공해서 사용할 수도 있습니다.

## 텐서플로 허브

텐서플로에선 **텐서플로 허브** *TensorFlow Hub*라는 재사용 가능한 모델을 쉽게 이용할 수 있도록 도와주는 라이브러리를 제공합니다. 텐서플로 허브 홈페이지에서 이미지, 텍스트, 비디오 등의 분야에서 사전 훈련된 모델을 검색해볼 수 있습니다. 

> 텐서플로 허브에서 사전 훈련된 MobileNet 불러오기

<script src="https://gist.github.com/yunkio/791331384a3b8d22473a2c98d55e52e2.js"></script>

~~~
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
keras_layer (KerasLayer)     (None, 1001)              3540265   
=================================================================
Total params: 3,540,265
Trainable params: 0
Non-trainable params: 3,540,265
_________________________________________________________________
~~~

**MobileNet**은 계산 부담이 큰 컨볼루션 신경망을 연산 성능이 제한된 모바일에서도 사용할 수 있도록 네트워크 구조를 경량화한 모델입니다. 허브에 올라와 있는 모델은 *hub.KerasLayer()* 명령으로 *keras*에서 사용 가능한 레이어로 변환할 수 있습니다. 이 모델은 1,000종류의 이미지를 포함하고 있는 ImageNet 데이터로 학습 시켰습니다. ImageNet의 데이터 중 일부만 모아놓은 ImageNetV2를 사용해서 얼마나 잘 분류하는지 알아보겠습니다. 여기서는 각 클래스에서 가장 많은 선택을 받은 이미지 10장씩 모아놓은 10,000장의 이미지가 포함된 TopImages 데이터를 사용합니다.

> ImageNetV2-TopImages 불러오기

<script src="https://gist.github.com/yunkio/a3e2f0fcb911149afae9b61dd2215bfb.js"></script>

~~~
Downloading data from [https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-top-images.tar.gz](https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-top-images.tar.gz)
1245929472/1245927936 [==============================] - 46s 0us/step 
1245937664/1245927936 [==============================] - 46s 0us/step
C:\ykio\Study_TF\Ch8_Pre-trained\data\imagenetv2-top-images-format-val
~~~

*path*에는 각자 경로를 설정해주시면 됩니다. *extract=True*를 주어 압축 파일이 자동으로 해제되도록 합니다. 각 라벨에 대한 숫자가 어떤 데이터를 뜻하는지는 따로 불러와야 합니다. 코드는 다음과 같습니다.

> ImageNet 라벨 텍스트 불러오기

<script src="https://gist.github.com/yunkio/1be69aed20d90e34eb9a463d739f9a76.js"></script>

~~~
Downloading data from [https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt](https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt)
16384/10484 [==============================================] - 0s 0us/step
24576/10484 [======================================================================] - 0s 0us/step
1001
['background', 'tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead', 'electric ray', 'stingray', 'cock', 'hen']
['buckeye', 'coral fungus', 'agaric', 'gyromitra', 'stinkhorn', 'earthstar', 'hen-of-the-woods', 'bolete', 'ear', 'toilet tissue']

[nltk_data] Downloading package wordnet to
[nltk_data]     C:\Users\ykio\AppData\Roaming\nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
~~~

제대로 불러온 것을 확인할 수 있습니다. 이제 이미지를 직접 확인해보겠습니다.

> 이미지 확인

<script src="https://gist.github.com/yunkio/131b13f599b3d34bc0591e2bb8077e73.js"></script>

~~~
image_count: 10000
~~~

![image](https://user-images.githubusercontent.com/35906602/140263432-6285a60d-678b-45e7-8123-04dca1fb1391.png)


실제로는 쓰이지 않는 background라는 라벨이 있어 실제 라벨보다 1 작은 값을 가지고 있어 1씩 더해서 처리했습니다. 무작위로 선택된 9장의 이미지를 확인할 수 있습니다. 라벨과 이미지를 보니 잘 매치가 되어있는 것을 확인할 수 있습니다. 

이제 이 이미지를 MobileNet이 얼마나 잘 분류하는지 확인해 보겠습니다. 상위 5개 이내에 데이터의 실제 분류가 포함되어 있는지를 보는 Top-5 정확도와 가장 높은 값만 보는 Top-1 정확도를 둘 다 보겠습니다.

> MobileNet의 분류 성능 확인

<script src="https://gist.github.com/yunkio/9fd8092e4c02e6abfab650ae1dedce88.js"></script>

~~~
Top-5 correctness: 83.52000000000001 %
Top-1 correctness: 59.06 %
~~~

정확도가 각각 83.5%, 59%가 나왔습니다. 임포트한 cv2라는 라이브러리는 OpenCV 라이브러리로, 이미지를 메모리에 불러오고 크기를 조정하는 등의 작업을 편하게 하도록 도와줍니다. *np.argsort()*는 인덱스를 정렬합니다. 이를 통해 예측 확률이 높은 순서대로 라벨을 불러오고 내림차순으로 정렬해 큰 순서대로 5개까지의 값만 불러왔습니다. 이를 통해 Top-5 정확도를 구할 수 있습니다. 이제 실제 이미지와 예측을 표시해보겠습니다.

> MobileNet의 분류 라벨 확인

<script src="https://gist.github.com/yunkio/2bd29c5820268a58e31285cd2d689e8f.js"></script>

![image](https://user-images.githubusercontent.com/35906602/140264460-2018c157-5e1b-417e-b6d9-bf4a466a4e54.png)

랜덤하게 3개의 이미지와 각 이미지에 대한 예측을 확인할 수 있습니다. 이렇게 별도의 훈련 과정 없이 미리 훈련된 모델을 불러와 네트워크를 그대로 사용할 수 있습니다.

## 전이 학습

**전이 학습** *Transfer Learning*은 미리 훈련된 모델을 다른 작업에 사용하기 위해 추가적인 학습을 시키는 것을 의미합니다. 이때 훈련된 모델은 유의미한 특징을 뽑아내기 위한 특징 추출기로 쓰이거나, 모델의 일부를 재학습시키기도 합니다.

### 모델의 일부를 재학습시키기

기존 모델에 레이어를 제거하거나 수정하고, 새로운 레이어를 추가하는 등의 작업을 할 수 있습니다. 이때 새로 추가된 레이어의 가중치만 훈련시키거나 미리 훈련된 모델의 일부 레이어만 다시 훈련 시킬 수도 있습니다. 본격적으로 데이터에 적용시켜보면서 살펴보겠습니다.

사용할 데이터는 스탠퍼드 대학의 *Dogs Dataset*으로 120가지 견종에 대한 2만여장의 사진으로 이루어져 있습니다. 여기서는 원본 데이터보다 접근이 쉬운 케글 *Kaggle*의 데이터세트를 사용하겠습니다.

> 데이터셋 불러오기

<script src="https://gist.github.com/yunkio/37d3655c3e42473de515f681f6fa90a5.js"></script>

~~~
Downloading data from [http://bit.ly/2GDxsYS](http://bit.ly/2GDxsYS)
483328/482063 [==============================] - 0s 1us/step
491520/482063 [==============================] - 0s 1us/step
Downloading data from [http://bit.ly/2GGnMNd](http://bit.ly/2GGnMNd)
25206784/25200295 [==============================] - 2s 0us/step
25214976/25200295 [==============================] - 2s 0us/step
Downloading data from [http://bit.ly/31nIyel](http://bit.ly/31nIyel)
361357312/361353329 [==============================] - 14s 0us/step
361365504/361353329 [==============================] - 14s 0us/step
Downloading data from [http://bit.ly/2GHEsnO](http://bit.ly/2GHEsnO)
362848256/362841195 [==============================] - 15s 0us/step
362856448/362841195 [==============================] - 15s 0us/step
~~~

데이터 셋을 불러와서 지정한 경로에 압축을 풀었습니다.  이제 데이터를 확인해보도록 하겠습니다

> labels.csv 확인하기

<script src="https://gist.github.com/yunkio/f88331c56feb2a4d3c68aa22ba456648.js"></script>

~~~
                                 id             breed
0  000bec180eb18c7604dcecc8fe0dba07       boston_bull
1  001513dfcb2ffafc82cccf4d8bbaba97             dingo
2  001cdf01b096e06d78e9e5112d419397          pekinese
3  00214f311d5d2247d5dfe4fe24b2303d          bluetick
4  0021f9ceb3235effd7fcde7f7538ed62  golden_retriever 

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10222 entries, 0 to 10221
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   id      10222 non-null  object
 1   breed   10222 non-null  object
dtypes: object(2)
memory usage: 159.8+ KB

120
~~~

총 10,222장의 사진이 데이터에 포함되어 있으며 데이터에 대한 대략적인 정보, 그리고 마지막 줄을 통해 총 몇 개의 클래스가 존재하는지 확인할 수 있습니다. 이제 실제로 이미지를 라벨과 함께 출력해보겠습니다.

> 이미지 확인

<script src="https://gist.github.com/yunkio/ed9cc227656d063fe64b09cfb6eaea72.js"></script>

![image](https://user-images.githubusercontent.com/35906602/140283837-16190df7-0fb8-4c7f-9ce0-77e876b47954.png)


이와 같이 각 사진과 라벨이 매치가 잘 되고 있음을 알 수 있습니다. 이제 데이터를 제대로 불러왔음을 확인했으니 전이 학습을 시도해보도록 하겠습니다. 기존의 가중치를 그대로 사용하고 일부 레이어의 가중치를 고정시킨 상태로 학습시킬 것입니다.

> 전이 학습 모델 정의

<script src="https://gist.github.com/yunkio/0dd4987c335bbc31e7bd5e2306353d0d.js"></script>

~~~
Downloading data from [https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5](https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5)
14540800/14536120 [==============================] - 1s 0us/step
14548992/14536120 [==============================] - 1s 0us/step
Model: "model_2"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_3 (InputLayer)            [(None, 224, 224, 3) 0                                            
__________________________________________________________________________________________________
Conv1 (Conv2D)                  (None, 112, 112, 32) 864         input_3[0][0]                    
__________________________________________________________________________________________________
bn_Conv1 (BatchNormalization)   (None, 112, 112, 32) 128         Conv1[0][0]                      
__________________________________________________________________________________________________
Conv1_relu (ReLU)               (None, 112, 112, 32) 0           bn_Conv1[0][0]                   
__________________________________________________________________________________________________

...

__________________________________________________________________________________________________
out_relu (ReLU)                 (None, 7, 7, 1280)   0           Conv_1_bn[0][0]                  
__________________________________________________________________________________________________
global_average_pooling2d_2 (Glo (None, 1280)         0           out_relu[0][0]                   
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 120)          153720      global_average_pooling2d_2[0][0] 
==================================================================================================
Total params: 2,411,704
Trainable params: 1,204,280
Non-trainable params: 1,207,424
__________________________________________________________________________________________________

~~~

뒤에서 20개까지의 레이어를 훈련 가능하게 하고 나머지 레이어의 가중치는 고정시켰습니다. 그럼 이제 학습을 위해 학습 데이터셋을 만들어보도록 하겠습니다.

> 학습 데이터셋 만들어주기

<script src="https://gist.github.com/yunkio/2ad7df32e9605927135b04444e562256.js"></script>

~~~
Found 7718 images belonging to 120 classes.
Found 2504 images belonging to 120 classes.
~~~

MobileNet V2의 입력형식에 맞도록 이미지의 크기를 고쳐주었고, 데이터를 한꺼번에 불러오기엔 차지하는 메모리가 크므로 감당이 가능한 만큼 가져올 수 있도록 generator를 정의해주었습니다. 또한 이 과정에서 앞에서 다뤘던 이미지 보강도 해 더 높은 성능을 기대할 수 있습니다. 이제 이렇게 만든 학습 데이터를 활용해 학습을 진행해보겠습니다.

> 모델 학습 및 결과 확인

<script src="https://gist.github.com/yunkio/7548b36e0543f7ba28f9097281be9d1f.js"></script>

~~~
Epoch 1/10
241/241 [==============================] - 127s 492ms/step - loss: 3.0941 - accuracy: 0.3259 - val_loss: 1.5182 - val_accuracy: 0.5739
Epoch 2/10
241/241 [==============================] - 93s 384ms/step - loss: 1.5425 - accuracy: 0.6175 - val_loss: 1.1866 - val_accuracy: 0.6514
Epoch 3/10
241/241 [==============================] - 92s 382ms/step - loss: 1.1832 - accuracy: 0.6854 - val_loss: 1.0762 - val_accuracy: 0.6725
Epoch 4/10
241/241 [==============================] - 91s 379ms/step - loss: 1.0203 - accuracy: 0.7273 - val_loss: 1.0182 - val_accuracy: 0.6977
Epoch 5/10
241/241 [==============================] - 91s 379ms/step - loss: 0.8952 - accuracy: 0.7579 - val_loss: 0.9667 - val_accuracy: 0.7081
Epoch 6/10
241/241 [==============================] - 92s 382ms/step - loss: 0.8159 - accuracy: 0.7786 - val_loss: 0.9657 - val_accuracy: 0.7069
Epoch 7/10
241/241 [==============================] - 92s 382ms/step - loss: 0.7379 - accuracy: 0.7991 - val_loss: 0.9484 - val_accuracy: 0.7181
Epoch 8/10
241/241 [==============================] - 92s 381ms/step - loss: 0.6847 - accuracy: 0.8143 - val_loss: 0.9415 - val_accuracy: 0.7220
Epoch 9/10
241/241 [==============================] - 91s 380ms/step - loss: 0.6326 - accuracy: 0.8300 - val_loss: 0.9533 - val_accuracy: 0.7129
Epoch 10/10
241/241 [==============================] - 91s 376ms/step - loss: 0.5833 - accuracy: 0.8461 - val_loss: 0.9307 - val_accuracy: 0.7292
~~~

![image](https://user-images.githubusercontent.com/35906602/140292559-0ce35231-31dd-47f7-a921-2adc392e16d3.png)

*val_accuracy* 가 약 73%를 보이고 있습니다. 이 글에서는 다루지는 않았지만 만약 기존에 저장된 가중치를 사용하지 않고 처음부터 학습시킨다면 정확도가 1%를 벗어나지 못 합니다.  또 *loss*는 감소하고 있고 정확도는 증가하고 있기 때문에 계속해서 성능이 향상될 것이라고 기대할 수 있습니다. 더해서 학습시킬 가중치의 숫자가 줄어들어서 속도도 더 빠릅니다. 적은 양의 데이터를 가지고 있을 경우 이렇게 미리 훈련된 모델의 가중치를 사용할 수 있습니다.

### 특징 추출기

미리 훈련된 모델에서 특징만 추출하고, 그 특징을 비교적 작은 네트워크에 통과시키는 방법도 있습니다. 학습할 때마다 전체 네트워크의 계산을 반복할 필요가 없어 계산량이 크게 줄어들고 메모리도 절약할 수 있습니다. 

텐서플로 허브에서 **Inception V3**를 구글에서 발표한 컨볼루션 신경망 모델입니다. 텐서플로 허브에는 Dense 레이어를 포함한 모델과 Dense 레이어가 없는 특징 추출기 네트워크가 있으므로 특징 추출기 네트워크를 불러오도록 하겠습니다.

> 텐서플로 허브에서 사전 훈련된 Inception V3의 특징 추출기 불러오기

<script src="https://gist.github.com/yunkio/7548b36e0543f7ba28f9097281be9d1f.js"></script>

~~~
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
keras_layer (KerasLayer)     (None, 2048)              21802784  
=================================================================
Total params: 21,802,784
Trainable params: 0
Non-trainable params: 21,802,784
_________________________________________________________________
~~~

앞서와 마찬가지로 네트워크 전체를 하나의 레이어로 불러 왔습니다. 마지막 레이어의 출력 크기가 2048이므로 해당 레이어를 통해서 2048 크기의 특징 벡터 *Feature Vector*를 추출할 수 있습니다.  *build()* 함수를 통해 299의 이미지 높이와 너비를 받고 RGB의 3차원을 받으며, 첫번째 인수로는 None을 넣어서 데이터의 수는 정해놓지 않습니다.

이제 이 특징 추출기를 이용해 훈련 데이터와 검증 데이터를 특징 벡터로 변환하겠습니다. 이렇게 변환한 특징 벡터를 작은 시퀀셜 모델에 넣어서 실제 라벨을 예측하게 됩니다.

> 훈련 데이터를 특징 벡터로 변환

<script src="https://gist.github.com/yunkio/a5b5bcbff7a38e953f6a7923ae3f5c42.js"></script>

~~~
0
100
...
700
(23058, 2048)
(23058, 120)
~~~

첫 줄에서는 *batch_step*을 지정했고, 3을 곱해주어 충분한 이미지 보강이 되도록 했습니다. x 값은 이미지 데이터에 해당하는 부분으로 특징 추출기를 통과하면 특징 벡터가 됩니다. 이미 학습이 완료된 특징 추출기를 사용하기 때문에 *predict()*를 사용합니다. 

최종 출력되는 값의 차원은 *train_features*가 (23084, 2048)로 의도했던 대로 나온 것을 알 수 있습니다. 기존 가로 세로 299픽셀, RGB 3차원의 이미지가 2048개의 차원으로 줄었습니다. 원본과 비교하면 약 0.76%의 차원으로 크게 줄었습니다. 이제 훈련 데이터를 변환했으니 검증 데이터도 변환하겠습니다.

> 검증 데이터를 특징 벡터로 변환

<script src="https://gist.github.com/yunkio/7bead471f08c5ccecbae27a3fa885598.js"></script>

~~~
0
100
200
...
2500
(2504, 2048)
(2504, 120)
~~~

검증 데이터는 훈련 데이터와 달리 한 번만 계산하면 되기 때문에 데이터 크기를 나타내는 *.n*을 넣었습니다. 이제 분류를 위한 간단한 네트워크를 만들겠습니다.

> 분류를 위한 작은 모델 정의

<script src="https://gist.github.com/yunkio/b54db5c48dad750cd0a97cf3276b4af8.js"></script>

~~~
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_3 (Dense)              (None, 256)               524544    
_________________________________________________________________
dropout (Dropout)            (None, 256)               0         
_________________________________________________________________
dense_4 (Dense)              (None, 120)               30840     
=================================================================
Total params: 555,384
Trainable params: 555,384
Non-trainable params: 0
_________________________________________________________________
~~~

첫 번째 Dense 레이어에선 입력으로 특징 추출기의 출력은 2048차원을 받습니다. 마지막에는 데이터의 클래스 수와 같은 128을 출력으로 내보냅니다. 그럼 이제 결과를 확인하겠습니다.

> 분류를 위한 작은 모델 학습

<script src="https://gist.github.com/yunkio/d0ea1a07dfdd5d4e92e7f144071f1f8a.js"></script>

~~~
Epoch 1/10
721/721 [==============================] - 2s 2ms/step - loss: 2.7438 - accuracy: 0.4108 - val_loss: 0.8665 - val_accuracy: 0.7823
Epoch 2/10
721/721 [==============================] - 2s 2ms/step - loss: 1.2111 - accuracy: 0.6879 - val_loss: 0.6669 - val_accuracy: 0.8035
...
Epoch 9/10
721/721 [==============================] - 2s 3ms/step - loss: 0.5792 - accuracy: 0.8255 - val_loss: 0.6284 - val_accuracy: 0.8151
Epoch 10/10
721/721 [==============================] - 2s 3ms/step - loss: 0.5490 - accuracy: 0.8309 - val_loss: 0.6331 - val_accuracy: 0.8135
~~~

![image](https://user-images.githubusercontent.com/35906602/140310547-91bce684-38a4-496e-b9ba-98441e2a39c6.png)


앞선 MobileNet V2에 비해 더 좋은 결과를 보여주고 있으며, 학습 속도는 훨씬 빨라졌습니다. 특징 추출기를 사용하면 더 많은 파라미터와 큰 이미지를 사용해도 학습 속도를 획기적으로 줄일 수 있습니다.

모델이 얼마나 예측을 잘 하는지 보기 위해 검증 데이터의 이미지에 대한 분류를 시각화해보겠습니다. *ImageDataGenerator*는 라벨을 인덱스로 저장할 때 알파벳 순으로 정렬된 순서로 저장하기 때문에 라벨 텍스트를 우선 정렬하고, 그 후 검증 데이터의 이미지에 대한 분류를 시각화 해보겠습니다.

> 특징 추출기 - 모델의 분류 라벨 확인

<script src="https://gist.github.com/yunkio/bd4b29cb4ab9047e313ad2a6a2feff2e.js"></script>

~~~
C:/ykio/Study_TF/Ch8_Pre-trained/data/dog/train_sub/brabancon_griffon\25573fa72e5b0052b27d7165a6da47e5.jpg
C:/ykio/Study_TF/Ch8_Pre-trained/data/dog/train_sub/brittany_spaniel\01b849a7e4fbc545f6b2806cb87ab371.jpg
C:/ykio/Study_TF/Ch8_Pre-trained/data/dog/train_sub/appenzeller\18c6389b08ab61f52298bfd1013e81fa.jpg
~~~

![image](https://user-images.githubusercontent.com/35906602/140310443-a632a47b-9629-4e86-b966-59cbf063490a.png)

이처럼 각 이미지에 대해 모델이 어떻게 예측하고 있는 지도 확인이 가능합니다. 테스트 데이터에 대한 예측은 각자 직접 해보면 좋을 것 같습니다.

## 신경 스타일 전이

**신경 스타일 전이** *Neural Style Transfer* 은 한 이미지에서 스타일을, 한 이미지에서 내용을 가져와 두 이미지의 스타일과 내용이 합성된 제3의 이미지를 만들어내는 모델입니다. 

### CNN을 사용한 텍스처 합성

**텍스처** *Texture*는 넓은 의미로는 이미지, 컴퓨터 비전에서 쓰는 좁은 의미로는 지역적으로는 비교적 다양한 값을 가지면서 전체적으로는 비슷한 모습을 보이는 이미지를 뜻합니다. **텍스처 합성** *Texture Synthesis*는 한 장의 이미지를 원본으로 삼아 해당 텍스처를 재생성 혹은 합성하는 작업입니다. 기존에는 공간적인 통곗값을 사람이 정교하게 만든 여러 개의 필터로 구하고, 필터를 통과하는 결과물이 같아질 때까지 타깃 텍스쳐를 변형시키는 방식을 사용했습니다. 현재는 이 필터를 인공 신경망으로 구할 수 있습니다. 한번 코드로 실습해보도록 하겠습니다.

> 원본 텍스처 이미지 불러오기

<script src="https://gist.github.com/yunkio/00d0456fbee36cdd5f859a764fa637d0.js"></script>

~~~
Downloading data from [http://bit.ly/2mGfZIq](http://bit.ly/2mGfZIq)
344064/337723 [==============================] - 0s 0us/step
352256/337723 [===============================] - 0s 0us/step
~~~

![image](https://user-images.githubusercontent.com/35906602/140313667-50f81dfe-7bed-4ce3-acf5-04a3f2e664f5.png)

이제 타깃 텍스쳐로 사용할 이미지를 만들겠습니다. 타깃 텍스처는 랜덤 노이즈 이미지에서 시작합니다.

> 타깃 텍스처 만들기

<script src="https://gist.github.com/yunkio/daae7bc8f8ad258538be8f6656c360b5.js"></script>

~~~
tf.Tensor([0.8026546  0.25072765 0.32582033], shape=(3,), dtype=float32)
~~~

![image](https://user-images.githubusercontent.com/35906602/140314022-87de15ce-20f5-4b0d-8914-61b70f7e2741.png)

이제 텍스처 합성에 사용할 네트워크를 불러오겠습니다. 여기서는 VGG-19를 사용하겠습니다. 각 레이어에 접근하기 쉽게 *tf.keras*에서 네트워크를 불러옵니다.

> VGG-19 네트워크 불러오기

<script src="https://gist.github.com/yunkio/aca121b6ddba1dc5824839f848631453.js"></script>

~~~
Downloading data from [https://storage.googleapis.com/tensorflow/keras-applications/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5](https://storage.googleapis.com/tensorflow/keras-applications/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5)
80142336/80134624 [==============================] - 1s 0us/step
80150528/80134624 [==============================] - 1s 0us/step
input_1
block1_conv1
...
block5_conv4
block5_pool
~~~

전체 네트워크를 불러올 필요는 없기 때문에 *include_top* 인수를 *False*로 지정해서 마지막 Dense 레이어를 제외한 나머지를 불러왔습니다. 이 네트워크는 특징 추출기의 역할을 하는 컨볼루션 레이어와 풀링 레이어를 포함하고 있습니다. 여기서 일부 레이어를 활용해 특징 추출 모델을 만들어 보겠습니다.

> 특징 추출 모델 만들기

<script src="https://gist.github.com/yunkio/698b5f1c24e2bf1b6726fafe376d225f.js"></script>

지역적인 구조와 전체적인 구조를 모두 잡아낼 수 있도록 앞쪽과 뒤쪽의 레이어를 모두 사용했습니다. 기존 가중치는 학습되지 않도록 얼려 놓고 선택된 다섯 개의 레이어를 출력으로 하는 모델입니다. 이미지를 입력하면 다섯 개의 레이어에서 출력되는 특징 추출값을 얻게 됩니다.

이제 Gram matrix를 계산해보겠습니다. Gram matrix는 각 뉴런의 특징 추출값을 1차원의 벡터로 변환한 다음에 벡터를 쌓아올린 행렬을 자신의 전치 행렬과 행렬곱해서 얻는 값으로, 특징 추출값의 상관관계를 나타내게 됩니다. 원본 텍스처와 타깃 텍스터 모두에 대해 구한 다음 두 Gram matrix의 평균 제곱 오차를 작게 하는 방향으로 학습이 진행됩니다.

> Gram matrix 계산

<script src="https://gist.github.com/yunkio/94d1b5b1eb1b3cdfbf5804cb6b764667.js"></script>

먼저 입력된 특징 추출값의 형태를 벡터로 변환해서 맨 뒤의 차원만 남기고 1차원의 벡터로 펴줍니다. 이렇게 만든 행렬을 자신의 전치행렬과 곱하면 [64, 64] 차원이 됩니다. 마지막 *return*문에서는 1차원의 벡터 길이로 나누어 줍니다. *style_output*은 다섯 레이어를 통과한 특징 추출값으로 구성됩니다. 그 중 하나를 출력해보겠습니다.

> 원본 텍스처의 첫 번째 특징 추출값 확인

<script src="https://gist.github.com/yunkio/8ad2acdb22efd1a632f3955fec38de4b.js"></script>

~~~
(1, 224, 224, 64)
~~~

![image](https://user-images.githubusercontent.com/35906602/140317021-f9d293a8-a4f2-4889-b63e-ec0e7194f765.png)

이제 원본 텍스처의 Gram matrix를 계산해서 값이 어떻게 나오는지 그래프로 분포를 확인해보겠습니다.

> 원본 텍스처의 Gram matrix 계산값 만들기, 분포 확인

<script src="https://gist.github.com/yunkio/b8e59be1751c1b9a1ac5c3b1ba949d0e.js"></script>

![image](https://user-images.githubusercontent.com/35906602/140317239-69f60412-57e4-48b4-8e21-1ce4d6be3b56.png)

레이어마다 다르게 나오고 최댓값도 크게 차이가 납니다. 

타깃 텍스처를 업데이트하기 위해서 몇 가지 함수를 설정하겠습니다. 먼저 Gram matrix를 구하는 함수가 필요하며, 원본 텍스처의 Gram matrix 값과 타깃 텍스처의 Gram matrix 사이의 MSE를 구해야 합니다. 또 나오는 값이 0에서 1 사이에 위치하도록 해야합니다.

이에 더해서 이미지를 업데이트 하는 함수가 필요합니다. 지금은 학습해야 할 가중치가 존재하지 않으며 2개의 이미지와 그 Gram matrix의 차이인 MSE만 존재합니다. 텐서플로의 *GradientTape*는 이런 상황의 해결책입니다. 어떤 식이 들어가더라도 자동 미분을 통해 입력에 대한 손실을 구한 뒤 다른 변수에 대한 Gradient를 계산합니다. 

> 함수 정의

<script src="https://gist.github.com/yunkio/b47e4abd35dc5a892843d336e458aeb1.js"></script>

첫 줄에서 최적화 함수를 정의했습니다. 더해서 *@tf.function*이라는 파이썬 문법 중 **장식자** *decorator* 라는 것을 사용했습니다. 장식자는 기존의 코드에 간편하게 기능을 추가할 수 있습니다. *GradientTape*는 계산에 관계되는 모든 변수와 연산을 추적하여 퍼포먼스를 개선하게 도와줍니다.  *tape.gradient(loss, image)*는 발생한 계산을 추적해서 입력값인 image에 대한 *loss*의 gradient를 계산합니다. 이제 텍스처를 실제로 합성해보겠습니다.

> 텍스처 합성 알고리즘 실행

<script src="https://gist.github.com/yunkio/ac2178a3ffec332c0b9a188d1c3f83eb.js"></script>

![image](https://user-images.githubusercontent.com/35906602/140318390-a1289f9d-a55d-4a46-8f37-7b2c0fcafdfd.png)

~~~
Total time: 73.8
~~~

이제 이미지에 생기는 노이즈를 개선해보겠습니다. 전체 손실에 *variation loss* 라는 것을 추가해서 어떤 픽셀과 그 옆에 인접한 픽셀의 차이를 최소화 해보겠습니다.

> variation loss 함수 정의

<script src="https://gist.github.com/yunkio/4454a92d471a7f2c59ecbaa9e2390604.js"></script>

*high_pass_x_y(image)* 함수를 통해 image의 x축 방향과 y축 방향의 차이를 구하고, *total_variation_loss(image)* 함수에서는 이렇게 구한 x, y축 방향의 차이를 제곱해서 평균을 낸 다음 합하게 됩니다. 이제 이 *loss*를 전체 손실 계산식에 추가하겠습니다.

> variation loss를 손실 계산식에 추가, 각 손실의 가중치 추가

<script src="https://gist.github.com/yunkio/79bdda7e67ec815e3fcff4303dbb114f.js"></script>

지금까지 구한 Gram matrix는 style loss라고 부르게 됩니다. 이 style loss와 새로 추가된 variation loss에 각각 가중치를 곱해서 전체 손실에 더합니다. 여기에 들어가는 가중치는 꾸준한 실험을 통해서 구해야합니다.

> variation loss를 추가한 알고리즘 실행

<script src="https://gist.github.com/yunkio/fcf02585173ed97f3b9af9ba114736b1.js"></script>

![image](https://user-images.githubusercontent.com/35906602/140319490-c1124e85-5d75-4a17-a36c-c54081d58d6a.png)


더 개선된 결과를 얻을 수 있습니다.

### 컨볼루션 신경망을 사용한 신경 스타일 전이

신경 스타일 전이는 위의 Gram matrix에 더해서 content 텍스처가 추가됩니다. content 텍스처는 픽셀 값의 차이를 구하게 됩니다. 이때 레이어는 서로 다른 것을 사용할 수 있습니다. content 복원에서 앞쪽 레이어의 특징 추출로 만든 복원 결과는 픽셀값을 복사하는 수준에 그칠 뿐이고, 뒤쪽 레이어의 특징 추출로 만든 복원 결과는 세부 픽셀은 깨지지만 전체적인 구조는 유지됩니다. 우리가 얻고 싶은 결과는 단순한 픽셀의 복사가 아니기 때문에 뒤쪽의 레이어를 사용합니다. 앞쪽의 레이어의 특징 추출을 같이 사용할 수도 있지만 이 경우 그냥 복사하는 결과를 불러올 수 있습니다. 그럼 먼저 content 원본 텍스처를 불러오겠습니다.

> content 텍스처 불러오기

<script src="https://gist.github.com/yunkio/96bbed6d0a2db567969b43d1ddcee490.js"></script>

~~~
Downloading data from [http://bit.ly/2mAfUX1](http://bit.ly/2mAfUX1)
761856/754420 [==============================] - 1s 1us/step
770048/754420 [==============================] - 1s 1us/step
~~~

![image](https://user-images.githubusercontent.com/35906602/140320752-906b54f7-84bf-4b31-a2db-859f9727b226.png)

이제 content의 특징을 추출하기 위한 모델을 만들고, output과 loss를 정의하겠습니다.

> content 특징 추출 모델, output, loss 함수 정의, loss 계산식 추가

<script src="https://gist.github.com/yunkio/18ed39094b81f745c88b35fd38477a36.js"></script>

우선 content 특징 추출을 위해 *block5_conv2* 레이어를 사용했습니다. style 특징을 추출하는 모델과 별도의 모델을 만들어서 model_content에 저장하고 이 모델을 사용해 content 텍스처에서 미리 특징을 추출해 따로 변수에 저장합니다. 

그 후 content의 output과 loss를 정의했습니다. loss에서는 타깃 텍스처와 content 텍스처의 픽셀 값의 MSE를 구합니다. 이 loss를 아까와 마찬가지로 계산식에 추가합니다. 이제 알고리즘을 실행해보겠습니다.

>  신경 스타일 전이 실행

<script src="https://gist.github.com/yunkio/fae52b97bd7042dfbd539385d831a0b3.js"></script>

![image](https://user-images.githubusercontent.com/35906602/140322357-cddde58d-b7c0-44a7-8a52-fc4369b05c44.png)

~~~
Total time: 133.4
~~~

학습률, 가중치 등의 하이퍼 파라미터를 변화시키면서 다른 결과를 얻을 수 있습니다. 또한 에포크가 진행됨에 따라 타깃 텍스처가 달라지기 때문에 적절한 에포크에서 멈추는 것도 중요할 수 있습니다.


