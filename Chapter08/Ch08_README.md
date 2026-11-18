# 혼공머신 8장 합성곱 신경망

> 『혼자 공부하는 머신러닝+딥러닝』 8장 정리노트임.
코드는 Keras 3 기준임. 책이랑 다른 부분은 따로 표시함.
> 

---

## 한눈에 보는 용어 정리

| 용어 | 쉬운 뜻 | 비유 |
| --- | --- | --- |
| 합성곱 (convolution) | 작은 창을 사진 위에서 한 칸씩 옮기며 곱하고 더하는 계산 | 돋보기로 사진을 한 칸씩 훑어보기 |
| 필터 (filter) | 그 작은 창. 숫자(가중치)가 들어 있는 작은 격자 | 특정 무늬만 찾는 돋보기 |
| 커널 (kernel) | 필터와 같은 말 | 돋보기의 다른 이름 |
| 특성 맵 (feature map) | 필터가 사진을 다 훑고 나서 나온 결과 | 무늬가 어디 있는지 표시한 지도 |
| 채널 (channel) | 사진이 몇 겹으로 되어 있는지 | 흑백은 종이 1장, 컬러는 빨강·초록·파랑 종이 3장 겹친 것 |
| 패딩 (padding) | 사진 테두리에 0을 둘러서 채우는 것 | 사진에 흰 액자 두르기 |
| 세임 패딩 (same padding) | 결과 크기가 원래 사진과 같아지게 0을 채움 | 액자를 둘러서 크기 유지 |
| 밸리드 패딩 (valid padding) | 0을 안 채움. 결과가 조금 작아짐 | 액자 없이 그대로 |
| 스트라이드 (stride) | 필터가 한 번에 옮겨 가는 칸 수 | 돋보기를 옮기는 보폭 |
| 풀링 (pooling) | 특성 맵 크기를 줄이는 것 | 사진을 축소 복사하기 |
| 최대 풀링 (max pooling) | 몇 칸씩 묶어서 가장 큰 값만 남김 | 반에서 1등만 뽑기 |
| 평균 풀링 (average pooling) | 몇 칸씩 묶어서 평균값을 남김 | 반 평균 점수 내기 |
| 합성곱 층 (Conv2D) | 합성곱을 하는 케라스 층 | 돋보기 여러 개를 든 팀 |
| 풀링 층 (MaxPooling2D) | 풀링을 하는 케라스 층 | 축소 복사기 |
| 합성곱 신경망 (CNN) | 합성곱 층을 쓰는 신경망 | 돋보기로 무늬를 찾고 그걸로 정답을 맞히는 모델 |
| 완전 연결 층 (Dense) | 앞의 모든 값과 연결된 층. CNN에서는 맨 끝에서 정답을 고름 | 모은 단서로 최종 판정하는 심사위원 |
| 함수형 API (functional API) | 층을 함수처럼 연결해서 모델을 만드는 방법 | 레고를 원하는 대로 이어 붙이기 |
| 가중치 시각화 | 필터 안의 숫자를 그림으로 그려 보기 | 돋보기 렌즈 모양 들여다보기 |
| 특성 맵 시각화 | 층을 지난 결과를 그림으로 그려 보기 | 돋보기가 찾은 무늬 지도 보기 |

---

## 8장 한 줄 요약

사진을 한 줄로 펴지 않고, 작은 돋보기(필터)로 무늬를 찾아가며 분류하는 신경망(CNN)을 만들고, 그 안에서 무슨 일이 일어나는지 그림으로 확인함.

> 💡 왜 필요함?
사진을 784칸 한 줄로 펴면 "이 픽셀 바로 아래에 어떤 픽셀이 있었는지" 정보가 사라짐.
CNN은 사진 모양 그대로 보기 때문에 선, 모서리, 무늬 같은 위치 정보를 살릴 수 있음.
> 

---

## 8-1 합성곱 신경망의 구성 요소

### 이 절에서 하는 일

- 합성곱이 뭔지 숫자 예시로 이해함
- 패딩, 스트라이드, 풀링이 크기를 어떻게 바꾸는지 봄
- 케라스로 합성곱 층과 풀링 층을 만드는 코드를 봄

### 합성곱

합성곱은 작은 창(필터)을 사진 위에 올려놓고 계산하는 것임.
겹친 칸끼리 곱하고, 그걸 전부 더해서 숫자 하나를 만듦.
그다음 창을 한 칸 옮겨서 또 계산함.

> 💡 비유
돋보기를 사진 왼쪽 위에 대고 "여기 세로선 있나?" 점수를 매김.
한 칸 옮겨서 또 점수를 매김. 이걸 사진 끝까지 반복함.
> 

**숫자로 보기**

4×4 사진에 3×3 필터를 올림.

```
사진(4×4)        필터(3×3)
1 2 0 1          1 0 1
0 1 3 1          0 1 0
2 1 0 0          1 0 1
1 0 1 2
```

왼쪽 위 3×3 칸에 필터를 겹치면 필터가 1인 자리만 남음.

```
사진 왼쪽 위 3×3     필터가 1인 자리의 값
1 2 0               1 . 0
0 1 3       →       . 1 .
2 1 0               2 . 0

1 + 0 + 1 + 2 + 0 = 4
```

이걸 오른쪽, 아래로 한 칸씩 옮기며 반복하면 결과가 나옴.

```
결과(2×2) = 특성 맵
4 7
6 4
```

- 실제로는 여기에 절편(따로 더하는 숫자 하나)을 더하고 활성화 함수를 거침
- 필터 안의 숫자는 사람이 정하지 않음. 학습하면서 스스로 정해짐

### 크기 계산

패딩이 없으면 결과가 조금 작아짐.

```
결과 크기 = 사진 크기 − 필터 크기 + 1

4×4 사진, 3×3 필터  →  4 − 3 + 1 = 2   →  2×2
28×28 사진, 3×3 필터 →  28 − 3 + 1 = 26 →  26×26
```

### 필터 여러 개

필터 하나는 무늬 하나만 찾음.
그래서 보통 필터를 여러 개 씀.
필터마다 특성 맵이 하나씩 나오고, 이걸 겹쳐서 쌓음.

```
28×28 사진 → 필터 10개 → 특성 맵 10장 → 모양 (26, 26, 10)
```

> 💡 비유
돋보기 10개가 각자 다른 무늬를 찾음.
하나는 세로선, 하나는 가로선, 하나는 동그라미 담당 같은 식임.
> 

### 패딩

패딩은 사진 테두리에 0을 한 줄 둘러 주는 것임.

```
패딩 전(4×4)        패딩 후(6×6)
                    0 0 0 0 0 0
1 2 0 1             0 1 2 0 1 0
0 1 3 1      →      0 0 1 3 1 0
2 1 0 0             0 2 1 0 0 0
1 0 1 2             0 1 0 1 2 0
                    0 0 0 0 0 0
```

6×6에 3×3 필터를 쓰면 6 − 3 + 1 = 4. 결과가 원래 크기 4×4 그대로임.

- **세임 패딩**: 이렇게 크기가 유지되게 0을 채움. CNN에서 많이 씀
- **밸리드 패딩**: 0을 안 채움. 결과가 작아짐. 케라스 기본값임
- 패딩을 하면 테두리 픽셀도 가운데 픽셀만큼 계산에 자주 쓰임

### 스트라이드

스트라이드는 필터가 한 번에 옮겨 가는 칸 수임.

- 기본값은 1칸
- 2칸씩 옮기면 결과 크기가 대략 절반이 됨
- 실제로는 1칸을 대부분 씀. 크기 줄이는 일은 풀링이 맡음

### 풀링

풀링은 특성 맵 크기를 줄이는 것임.
몇 칸씩 묶어서 숫자 하나만 남김.

```
특성 맵(4×4)       2×2씩 묶음
1 3 | 2 0
5 2 | 1 4
---------
0 1 | 7 2
3 2 | 1 6

최대 풀링 (가장 큰 값)   평균 풀링 (평균값)
5 4                     2.75 1.75
3 7                     1.5  4.0
```

- 2×2 풀링을 하면 가로·세로가 절반이 됨 (4×4 → 2×2)
- 특성 맵 장수(채널)는 그대로임
- 풀링은 고르기만 하니까 학습할 숫자(가중치)가 없음
- 최대 풀링을 더 많이 씀. 무늬가 "있다/없다"를 가장 뚜렷하게 남기기 때문임
- 풀링은 묶음끼리 겹치지 않게 옮김. 2×2 풀링이면 스트라이드도 2임

> 💡 비유
사진을 반으로 축소 복사하는 것과 비슷함.
세부는 조금 사라지지만 "여기 무늬가 있다"는 정보는 남음.
> 

### 컬러 사진이면

컬러 사진은 빨강·초록·파랑 3겹(채널 3개)임.
그래서 필터도 3겹짜리가 됨.

```
컬러 사진 (32, 32, 3)  ×  필터 (3, 3, 3)  →  특성 맵 1장
```

- 3겹을 한꺼번에 계산해서 특성 맵은 1장만 나옴
- 합성곱 층 두 개를 쌓을 때도 같음. 앞 층에서 특성 맵이 10장 나왔으면 다음 층 필터는 10겹짜리가 됨

### 코드 따라가기

```python
import keras

keras.layers.Conv2D(10, kernel_size=(3, 3), activation="relu")
```

- `Conv2D`: 합성곱 층을 만듦. 2D는 사진처럼 가로·세로 2방향으로 움직인다는 뜻임
- `10`: 필터 개수. 특성 맵이 10장 나옴
- `kernel_size=(3, 3)`: 필터 크기. `kernel_size=3`이라고만 써도 같음
- `activation="relu"`: 계산 결과에 ReLU를 적용함 (음수는 0, 양수는 그대로)

```python
keras.layers.Conv2D(10, kernel_size=(3, 3), activation="relu", padding="same")
```

- `padding="same"`: 세임 패딩. 안 쓰면 기본값 `"valid"`

```python
keras.layers.Conv2D(10, kernel_size=(3, 3), activation="relu", padding="same", strides=1)
```

- `strides=1`: 필터를 1칸씩 옮김. 기본값이라 보통 안 적음

```python
keras.layers.MaxPooling2D(2)
keras.layers.MaxPooling2D(2, strides=2, padding="valid")
```

- `MaxPooling2D(2)`: 2×2씩 묶어서 최대 풀링
- 두 번째 줄은 첫 줄의 기본값을 다 적은 것임. 결과는 같음
- 평균 풀링은 `keras.layers.AveragePooling2D`

### 결과 읽는 법

합성곱 층과 풀링 층의 결과 모양은 `(세로, 가로, 장수)`로 읽음.

```
(28, 28, 1)  →  Conv2D(32, 3, padding="same")  →  (28, 28, 32)
(28, 28, 32) →  MaxPooling2D(2)                →  (14, 14, 32)
```

- 합성곱 층은 장수를 필터 개수로 바꿈
- 풀링 층은 세로·가로를 줄이고 장수는 그대로 둠

### 자주 나는 에러

| 증상 | 원인 | 해결 |
| --- | --- | --- |
| `kernel_size`가 없다는 에러 | 필터 개수만 쓰고 크기를 안 씀 | `Conv2D(10, kernel_size=3)` |
| `input_shape`를 쓰지 말라는 경고 | Keras 3는 층 안에 입력 모양을 넣는 방식을 권하지 않음 | 모델 맨 앞에 `keras.Input(shape=(28, 28, 1))` |
| 크기가 예상보다 작게 나옴 | 패딩 기본값이 `"valid"`임 | 크기를 유지하려면 `padding="same"` |

### 이 절 핵심 3줄

1. 합성곱 = 작은 필터를 옮기며 곱하고 더하기. 필터 하나당 특성 맵 하나
2. 세임 패딩은 크기 유지, 밸리드 패딩은 크기가 작아짐
3. 최대 풀링은 2×2마다 가장 큰 값만 남겨서 가로·세로를 절반으로. 가중치 없음

---

## 8-2 합성곱 신경망을 사용한 이미지 분류

### 이 절에서 하는 일

- 패션 MNIST를 CNN에 맞는 모양으로 준비함
- 합성곱 층 2개 + 풀링 층 2개 + Dense 층으로 CNN을 만듦
- 학습하고, 사진 한 장을 예측해 보고, 테스트 세트로 마지막 평가를 함

### 전체 구조

```
사진 (28, 28, 1)
  ↓ 합성곱 층 1 (필터 32개, 세임 패딩)    → (28, 28, 32)
  ↓ 최대 풀링 (2×2)                       → (14, 14, 32)
  ↓ 합성곱 층 2 (필터 64개, 세임 패딩)    → (14, 14, 64)
  ↓ 최대 풀링 (2×2)                       → (7, 7, 64)
  ↓ Flatten (한 줄로 펴기)                → 3,136
  ↓ Dense 100 (ReLU)                      → 100
  ↓ Dropout 0.4
  ↓ Dense 10 (소프트맥스)                 → 10  (옷 10종류 확률)
```

> 💡 비유
앞부분(합성곱·풀링)은 "단서 찾는 탐정 팀"임. 선, 모서리, 무늬를 찾음.
뒷부분(Dense)은 "판사"임. 탐정이 모은 단서를 보고 무슨 옷인지 판정함.
> 

### 코드 따라가기

**① 데이터 준비**

```python
import keras
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()

train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)
```

- `load_data()`: 패션 MNIST를 불러옴. 28×28 흑백 옷 사진 6만 장 + 테스트 1만 장
- `reshape(-1, 28, 28, 1)`: 맨 끝에 채널 칸(1겹)을 붙임. 합성곱 층은 (세로, 가로, 채널) 모양을 받기 때문임
- `/ 255.0`: 픽셀 값 0~255를 0~1로 줄임. 학습이 안정적으로 됨
- `train_test_split(...)`: 20%를 검증 세트로 떼어 냄 → 훈련 (48000, 28, 28, 1), 검증 (12000, 28, 28, 1)

**② 모델 만들기**

```python
model = keras.Sequential()
model.add(keras.Input(shape=(28, 28, 1)))
model.add(keras.layers.Conv2D(32, kernel_size=3, activation="relu", padding="same"))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Conv2D(64, kernel_size=3, activation="relu", padding="same"))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10, activation="softmax"))
```

- `keras.Input(shape=(28, 28, 1))`: 사진 한 장의 모양을 알려 줌
- 첫 번째 `Conv2D(32, ...)`: 3×3 필터 32개. 세임 패딩이라 크기 유지 → (28, 28, 32)
- 첫 번째 `MaxPooling2D(2)`: 가로·세로 절반 → (14, 14, 32)
- 두 번째 `Conv2D(64, ...)`: 필터 64개. 더 복잡한 무늬를 찾음 → (14, 14, 64)
- 두 번째 `MaxPooling2D(2)`: → (7, 7, 64)
- `Flatten()`: 7 × 7 × 64 = 3,136칸 한 줄로 폄. Dense 층은 한 줄만 받기 때문임
- `Dense(100, relu)`: 찾은 무늬들을 조합하는 층
- `Dropout(0.4)`: 학습할 때 40%를 무작위로 꺼서 외우기(과대적합)를 막음
- `Dense(10, softmax)`: 옷 10종류 각각의 확률을 냄

> ⚠️ 책이랑 다른 부분 (Keras 3)
책은 첫 `Conv2D` 안에 `input_shape=(28, 28, 1)`을 넣음.
Keras 3에서는 경고가 뜨니까 맨 앞에 `keras.Input(shape=(28, 28, 1))`을 따로 둠.
> 

**③ 구조 확인**

```python
model.summary()
```

- 층별 결과 모양과 파라미터 개수를 표로 보여 줌

```python
keras.utils.plot_model(model, show_shapes=True)
```

- 층 구조를 그림으로 그려 줌
- `pydot`과 `graphviz`를 따로 설치해야 돌아감. 설치 안 하면 에러가 남. 이 줄은 건너뛰어도 학습에는 상관없음

**④ 학습**

```python
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

checkpoint_cb = keras.callbacks.ModelCheckpoint("best-cnn-model.keras",
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2,
                                                  restore_best_weights=True)

history = model.fit(train_scaled, train_target, epochs=20,
                    validation_data=(val_scaled, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])
```

- `optimizer="adam"`: 가중치를 고치는 방법. 무난하게 잘 됨
- `loss="sparse_categorical_crossentropy"`: 정답이 0~9 숫자일 때 쓰는 손실 함수
- `ModelCheckpoint`: 검증 손실이 가장 낮을 때 모델을 `best-cnn-model.keras` 파일로 저장
- `EarlyStopping(patience=2)`: 검증 손실이 2번 연속 안 좋아지면 자동으로 멈춤
- `restore_best_weights=True`: 멈출 때 가장 좋았던 상태로 되돌림
- `epochs=20`: 최대 20바퀴. 보통은 조기 종료로 그 전에 멈춤

> ⚠️ 책이랑 다른 부분 (Keras 3)
책 버전에 따라 저장 파일이 `best-cnn-model.h5`로 되어 있을 수 있음.
Keras 3에서는 `.keras`로 끝나야 함.
> 

**⑤ 손실 곡선**

```python
import matplotlib.pyplot as plt

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train", "val"])
plt.show()
```

- 에포크마다 훈련 손실과 검증 손실을 그림
- 검증 손실이 가장 낮은 지점 근처에서 학습이 멈춘 걸 볼 수 있음

**⑥ 검증 세트로 평가**

```python
model.evaluate(val_scaled, val_target)
```

- `[손실, 정확도]` 두 숫자가 나옴
- `restore_best_weights=True` 덕분에 가장 좋았던 상태로 평가됨

**⑦ 사진 한 장 예측해 보기**

```python
plt.imshow(val_scaled[0].reshape(28, 28), cmap="gray_r")
plt.show()
```

- 검증 세트 첫 번째 사진을 그림. `imshow`는 (28, 28) 모양을 받으니까 채널 칸을 빼고 그림
- `cmap="gray_r"`: 흑백을 뒤집어서 옷이 검게 보이게 함

```python
preds = model.predict(val_scaled[0:1])
print(preds)
```

- `val_scaled[0:1]`: 사진 1장이지만 "묶음(배치)" 모양을 유지하려고 `[0]`이 아니라 `[0:1]`로 자름 → (1, 28, 28, 1)
- 결과는 옷 10종류 각각의 확률 10개임

```python
plt.bar(range(1, 11), preds[0])
plt.xlabel("class")
plt.ylabel("prob.")
plt.show()
```

- 확률 10개를 막대그래프로 그림. 가장 높은 막대가 모델의 답임

```python
import numpy as np

classes = ["티셔츠", "바지", "스웨터", "드레스", "코트",
           "샌달", "셔츠", "스니커즈", "가방", "앵클 부츠"]
print(classes[np.argmax(preds)])
```

- `np.argmax(preds)`: 가장 큰 확률의 위치(0~9)를 찾음
- 그 위치로 `classes`에서 옷 이름을 꺼냄

**⑧ 테스트 세트로 마지막 평가**

```python
test_scaled = test_input.reshape(-1, 28, 28, 1) / 255.0
model.evaluate(test_scaled, test_target)
```

- 테스트 세트도 훈련 세트와 똑같이 준비해야 함 (채널 칸 추가 + ÷255)
- 테스트 세트는 모델을 다 만든 뒤 마지막에 한 번만 씀

### 층별 모양과 파라미터 수

| 층 | 결과 모양 | 파라미터 수 | 계산 |
| --- | --- | --- | --- |
| Conv2D (32) | (28, 28, 32) | 320 | (3 × 3 × 1 + 1) × 32 |
| MaxPooling2D | (14, 14, 32) | 0 | 고르기만 함 |
| Conv2D (64) | (14, 14, 64) | 18,496 | (3 × 3 × 32 + 1) × 64 |
| MaxPooling2D | (7, 7, 64) | 0 | 고르기만 함 |
| Flatten | 3,136 | 0 | 7 × 7 × 64 |
| Dense (100) | 100 | 313,700 | 3,136 × 100 + 100 |
| Dropout | 100 | 0 | 끄기만 함 |
| Dense (10) | 10 | 1,010 | 100 × 10 + 10 |
| **합계** |  | **333,526** |  |

```
합성곱 층 파라미터 = (필터 가로 × 필터 세로 × 들어오는 장수 + 1) × 필터 개수

첫 번째:  (3 × 3 × 1  + 1) × 32 = 10 × 32  = 320
두 번째:  (3 × 3 × 32 + 1) × 64 = 289 × 64 = 18,496
```

- "+ 1"은 필터마다 하나씩 있는 절편임
- 두 번째 합성곱 층의 필터는 32겹짜리임. 앞 층에서 특성 맵이 32장 들어오기 때문임
- 합성곱 층은 같은 필터를 사진 전체에 돌려 써서 파라미터가 적음. 파라미터 대부분은 Dense 층에 있음

### 결과 읽는 법

**summary()**

- `Output Shape`의 `(None, 28, 28, 32)`: None은 사진 장수(정해지지 않음), 나머지는 사진 한 장의 결과 모양
- `Param #`: 그 층의 파라미터 수. 위 표와 같아야 함
- 풀링, Flatten, Dropout 층은 파라미터가 0임

**fit 출력**

- 에포크마다 `accuracy`, `loss`, `val_accuracy`, `val_loss`가 찍힘
- 앞의 두 개는 훈련 세트, `val_`이 붙은 건 검증 세트 점수임
- 조기 종료가 되면 20에포크 전에 멈춤

**손실 곡선**

- 두 선이 같이 내려가면 잘 배우는 중임
- 훈련 손실은 내려가는데 검증 손실이 올라가기 시작하면 과대적합임. 그 근처에서 조기 종료가 멈춰 줌

**막대그래프**

- 막대 하나가 압도적으로 높으면 모델이 확신하고 있다는 뜻임
- 막대 몇 개가 비슷하면 모델이 헷갈리는 사진임

### 자주 나는 에러

| 증상 | 원인 | 해결 |
| --- | --- | --- |
| 입력이 4차원이어야 하는데 3차원이라는 에러 | 채널 칸을 안 붙임 (60000, 28, 28) | `reshape(-1, 28, 28, 1)` |
| `plot_model`에서 pydot 설치하라는 에러 | 그림 그리는 프로그램이 없음 | `pip install pydot`  • graphviz 설치, 또는 그 줄 건너뛰기 |
| `ModelCheckpoint` 파일 이름 에러 | Keras 3는 `.keras`로 끝나야 함 | `"best-cnn-model.keras"` |
| 사진 한 장 예측에서 모양 에러 | `val_scaled[0]`으로 잘라서 묶음 칸이 사라짐 | `val_scaled[0:1]` |
| 테스트 점수가 이상하게 낮음 | 테스트 세트에 ÷255나 reshape를 안 함 | 훈련 세트와 똑같이 준비 |

### 이 절 핵심 3줄

1. 입력은 `reshape(-1, 28, 28, 1)`로 채널 칸을 붙이고 ÷255
2. 구조: (합성곱 → 풀링) × 2 → Flatten → Dense → Dropout → Dense(10, softmax)
3. 파라미터 합계 333,526개. 대부분 Flatten 다음 Dense 층에 있음

---

## 8-3 합성곱 신경망의 시각화

### 이 절에서 하는 일

- 학습된 필터(가중치)를 그림으로 그려서 학습 전과 비교함
- 함수형 API로 "중간 층 결과만 꺼내는 모델"을 만듦
- 특성 맵을 그림으로 그려서 각 층이 무엇을 보는지 확인함

> 💡 비유
지금까지는 탐정 팀의 최종 판결만 봤음.
이 절에서는 돋보기 렌즈(필터)가 어떻게 생겼는지, 돋보기로 본 장면(특성 맵)이 어떤지 직접 들여다봄.
> 

### 코드 따라가기

**① 저장한 모델 불러오기**

```python
import keras

model = keras.models.load_model("best-cnn-model.keras")
print(model.layers)
```

- 8-2에서 저장한 가장 좋은 모델을 불러옴
- `model.layers`: 모델 안의 층 목록. 0번이 첫 번째 합성곱 층임

> ⚠️ 책이랑 다른 부분 (Keras 3)
`keras.Input`은 층 목록에 안 들어감.
그래서 `model.layers[0]`이 첫 번째 합성곱 층, `model.layers[2]`가 두 번째 합성곱 층임.
> 

**② 필터(가중치) 꺼내 보기**

```python
conv = model.layers[0]
print(conv.weights[0].shape, conv.weights[1].shape)
```

- `conv.weights[0]`: 필터 숫자들. 모양은 (3, 3, 1, 32) → 3×3 크기, 1겹, 32개
- `conv.weights[1]`: 절편. 모양은 (32,) → 필터마다 하나

```python
conv_weights = conv.weights[0].numpy()
print(conv_weights.mean(), conv_weights.std())
```

- `.numpy()`: 넘파이 배열로 바꿔서 계산하기 쉽게 함
- `mean()`, `std()`: 필터 숫자의 평균과 퍼진 정도(표준편차)

```python
import matplotlib.pyplot as plt

plt.hist(conv_weights.reshape(-1, 1))
plt.xlabel("weight")
plt.ylabel("count")
plt.show()
```

- 필터 숫자 288개(3 × 3 × 1 × 32)를 한 줄로 펴서 히스토그램(값의 분포 막대그래프)을 그림

**③ 필터 32개를 그림으로**

```python
fig, axs = plt.subplots(2, 16, figsize=(15, 2))
for i in range(2):
    for j in range(16):
        axs[i, j].imshow(conv_weights[:, :, 0, i*16 + j], vmin=-0.5, vmax=0.5)
        axs[i, j].axis("off")
plt.show()
```

- `plt.subplots(2, 16)`: 그림 칸을 2줄 × 16칸 = 32개 만듦
- `conv_weights[:, :, 0, i*16 + j]`: 필터 하나(3×3)를 꺼냄. `i*16 + j`는 0~31번 필터 번호
- `vmin=-0.5, vmax=0.5`: 색 기준을 모든 그림에 똑같이 맞춤. 그래야 필터끼리 비교할 수 있음
- `axis("off")`: 눈금을 숨김

**④ 학습 안 한 모델과 비교**

```python
no_training_model = keras.Sequential()
no_training_model.add(keras.Input(shape=(28, 28, 1)))
no_training_model.add(keras.layers.Conv2D(32, kernel_size=3, activation="relu",
                                          padding="same"))

no_training_conv = no_training_model.layers[0]
no_training_weights = no_training_conv.weights[0].numpy()
print(no_training_weights.mean(), no_training_weights.std())
```

- 같은 모양의 합성곱 층을 새로 만들고 학습은 안 함
- 이 필터 숫자도 ②, ③처럼 히스토그램과 그림으로 그려서 비교함
- 학습 전 필터는 무작위로 고르게 퍼져 있음. 학습한 필터는 값이 더 넓게 퍼지고 뚜렷한 무늬가 생김. 필터가 무언가를 배웠다는 뜻임

### 함수형 API

지금까지는 `Sequential`로 층을 순서대로 쌓기만 했음.
함수형 API는 층을 함수처럼 불러서 연결하는 방법임.
입력과 출력을 내가 직접 정할 수 있음.

```python
inputs = keras.Input(shape=(784,))
dense1 = keras.layers.Dense(100, activation="relu")
dense2 = keras.layers.Dense(10, activation="softmax")

hidden = dense1(inputs)
outputs = dense2(hidden)
model_f = keras.Model(inputs, outputs)
```

- `keras.Input(...)`: 입력 자리를 만듦
- `dense1(inputs)`: 층을 함수처럼 부르면 "입력을 이 층에 넣은 결과"가 나옴
- `keras.Model(inputs, outputs)`: 입력부터 출력까지 이어진 모델을 만듦

> 💡 비유
Sequential은 기차처럼 한 줄로만 연결됨.
함수형 API는 레고처럼 "여기서 시작해서 여기까지"를 내가 정해서 이어 붙일 수 있음.
> 

이걸 쓰면 "입력은 그대로, 출력은 첫 번째 합성곱 층 결과"인 모델을 만들 수 있음.
그 모델에 사진을 넣으면 특성 맵이 바로 나옴.

**⑤ 첫 번째 합성곱 층의 특성 맵**

```python
conv_acti = keras.Model(model.inputs, model.layers[0].output)
```

- `model.inputs`: 불러온 모델의 입력 자리
- `model.layers[0].output`: 첫 번째 합성곱 층의 결과
- 이 둘을 이은 새 모델 = "사진 넣으면 첫 번째 특성 맵이 나오는 모델"

> ⚠️ 책이랑 다른 부분 (Keras 3)
책은 `model.input`(단수)을 씀.
Keras 3에서는 모델에 따라 에러가 날 수 있어서 `model.inputs`(복수)가 더 안전함.
> 

```python
(train_input, train_target), (test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()

plt.imshow(train_input[0], cmap="gray_r")
plt.show()
```

- 훈련 세트 첫 번째 사진을 그림 (앵클 부츠)

```python
inputs = train_input[0:1].reshape(-1, 28, 28, 1) / 255.0
feature_maps = conv_acti.predict(inputs)
print(feature_maps.shape)
```

- 사진 1장을 학습 때와 똑같이 준비함 (묶음 모양 유지 + 채널 칸 + ÷255)
- `predict`: 특성 맵을 꺼냄
- 모양은 (1, 28, 28, 32) → 사진 1장, 28×28 특성 맵 32장

```python
fig, axs = plt.subplots(4, 8, figsize=(15, 8))
for i in range(4):
    for j in range(8):
        axs[i, j].imshow(feature_maps[0, :, :, i*8 + j])
        axs[i, j].axis("off")
plt.show()
```

- 특성 맵 32장을 4줄 × 8칸으로 그림

**⑥ 두 번째 합성곱 층의 특성 맵**

```python
conv2_acti = keras.Model(model.inputs, model.layers[2].output)
feature_maps = conv2_acti.predict(inputs)
print(feature_maps.shape)
```

- `model.layers[2]`: 두 번째 합성곱 층 (0 합성곱, 1 풀링, 2 합성곱)
- 모양은 (1, 14, 14, 64) → 풀링을 한 번 거쳐서 14×14, 필터 64개라 64장

```python
fig, axs = plt.subplots(8, 8, figsize=(12, 12))
for i in range(8):
    for j in range(8):
        axs[i, j].imshow(feature_maps[0, :, :, i*8 + j])
        axs[i, j].axis("off")
plt.show()
```

- 특성 맵 64장을 8줄 × 8칸으로 그림

### 결과 읽는 법

**필터 그림 (가중치 시각화)**

- 밝은 칸은 값이 큰 곳, 어두운 칸은 값이 작은 곳임
- 학습한 필터는 밝고 어두운 칸이 뚜렷하게 나뉨. 특정 방향의 선이나 경계를 찾는 모양임
- 학습 안 한 필터는 밝기가 뒤죽박죽임

**첫 번째 층 특성 맵**

- 앵클 부츠 모양이 그대로 보임
- 밝게 표시된 곳이 그 필터가 "찾았다"고 반응한 곳임
- 어떤 맵은 세로 경계가, 어떤 맵은 바닥 쪽 경계가 밝게 나옴

**두 번째 층 특성 맵**

- 사진 모양을 알아보기 어려워짐
- 첫 번째 층이 찾은 무늬들을 다시 조합해서 더 복잡한 특징을 보기 때문임
- 층이 깊어질수록 사람 눈에는 낯설고, 모델에게는 더 쓸모 있는 정보가 됨

### 자주 나는 에러

| 증상 | 원인 | 해결 |
| --- | --- | --- |
| `model.input` 관련 에러 | Keras 3에서 단수 `input`이 안 되는 경우가 있음 | `model.inputs` 사용 |
| 엉뚱한 층의 특성 맵이 나옴 | 층 번호를 잘못 셈 | `print(model.layers)`로 순서 확인 |
| 특성 맵이 이상하게 나옴 | 사진을 ÷255나 reshape 없이 넣음 | 학습 때와 똑같이 준비 |
| 모델 파일을 못 찾음 | 8-2에서 저장한 파일과 다른 폴더에서 실행함 | 파일이 있는 폴더에서 실행하거나 경로를 정확히 씀 |

### 이 절 핵심 3줄

1. 학습한 필터는 뚜렷한 무늬를 가짐. 학습 전과 비교하면 배운 게 보임
2. 함수형 API로 `keras.Model(model.inputs, 원하는 층.output)`을 만들면 중간 결과를 꺼낼 수 있음
3. 앞쪽 층은 선·경계 같은 단순한 무늬, 뒤쪽 층은 그걸 조합한 복잡한 특징을 봄

---

## 8장 전체 흐름 정리

| 단계 | 하는 일 | 핵심 코드 |
| --- | --- | --- |
| 1. 데이터 준비 | 채널 칸 붙이고 0~1로 | `train_input.reshape(-1, 28, 28, 1) / 255.0` |
| 2. 검증 세트 | 20% 떼어 냄 | `train_test_split(..., test_size=0.2)` |
| 3. 합성곱 층 | 필터로 무늬 찾기 | `Conv2D(32, kernel_size=3, activation="relu", padding="same")` |
| 4. 풀링 층 | 크기 절반으로 | `MaxPooling2D(2)` |
| 5. 한 줄로 펴기 | Dense에 넣을 준비 | `Flatten()` |
| 6. 판정 | 옷 10종류 확률 | `Dense(10, activation="softmax")` |
| 7. 학습 | 최고 모델 저장 + 조기 종료 | `model.fit(..., callbacks=[checkpoint_cb, early_stopping_cb])` |
| 8. 평가 | 마지막에 테스트 세트로 | `model.evaluate(test_scaled, test_target)` |
| 9. 필터 보기 | 필터 숫자를 그림으로 | `model.layers[0].weights[0].numpy()` |
| 10. 특성 맵 보기 | 중간 층 결과를 그림으로 | `keras.Model(model.inputs, model.layers[0].output)` |

**8-2 모델 모양 변화 한 줄로**

```
(28, 28, 1) → (28, 28, 32) → (14, 14, 32) → (14, 14, 64) → (7, 7, 64) → 3,136 → 100 → 10
```