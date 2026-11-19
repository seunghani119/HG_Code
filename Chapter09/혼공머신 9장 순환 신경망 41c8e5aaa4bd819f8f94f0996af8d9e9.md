# 혼공머신 9장 순환 신경망

> 『혼자 공부하는 머신러닝+딥러닝』 9장 정리노트임.
코드는 Keras 3 기준임. 책이랑 다른 부분은 따로 표시함.
> 

---

## 한눈에 보는 용어 정리

| 용어 | 쉬운 뜻 | 비유 |
| --- | --- | --- |
| 순차 데이터 (sequential data) | 순서가 중요한 데이터. 글, 날짜별 주가, 음성 같은 것 | 순서를 섞으면 뜻이 바뀌는 문장 |
| 피드포워드 신경망 (feedforward) | 데이터가 앞으로만 흐르는 신경망. 7장·8장 모델이 전부 이거임 | 한 번 보고 바로 답하는 사람 |
| 순환 신경망 (RNN) | 앞에서 본 내용을 기억하면서 다음 것을 보는 신경망 | 앞 문장을 기억하며 책 읽는 사람 |
| 순환층 (recurrent layer) | 순환 신경망에서 기억하며 계산하는 층 | 메모하며 읽는 사람 |
| 셀 (cell) | 순환층 안에서 실제로 계산하는 부분. 순환층과 거의 같은 말로 씀 | 메모하는 손 |
| 타임스텝 (timestep) | 순서대로 들어오는 한 조각. 글이면 단어 하나 | 문장 속 단어 한 개 |
| 은닉 상태 (hidden state) | 셀이 계산해서 다음 타임스텝으로 넘겨주는 값 | 지금까지 읽은 내용 요약 메모 |
| tanh (하이퍼볼릭 탄젠트) | 값을 -1~1 사이로 바꾸는 활성화 함수 | 너무 큰 값도 -1~1로 눌러 담기 |
| 말뭉치 (corpus) | 공부에 쓰는 글 모음 | 리뷰 5만 개 묶음 |
| 토큰 (token) | 글을 잘게 나눈 한 조각. 이 책에서는 단어 하나 | 문장을 단어로 자른 조각 |
| 어휘 사전 (vocabulary) | 토큰마다 번호를 붙여 둔 목록 | 단어 → 번호 사전 |
| 패딩 (padding) | 짧은 글 앞에 0을 채워 길이를 맞추는 것 | 줄 맞추려고 빈칸 채우기 |
| 원-핫 인코딩 (one-hot) | 번호를 "해당 자리만 1, 나머지 0"인 긴 줄로 바꾸는 것 | 수백 칸짜리 체크리스트에 한 칸만 체크 |
| 단어 임베딩 (word embedding) | 단어 번호를 짧은 실수 숫자 묶음으로 바꾸는 것. 비슷한 단어는 비슷한 숫자가 되도록 학습함 | 단어마다 특징 카드 만들기 |
| LSTM | 오래된 내용도 잘 기억하도록 만든 셀 | 긴 책도 중요한 건 노트에 따로 적어 두는 사람 |
| 셀 상태 (cell state) | LSTM이 따로 들고 다니는 장기 기억 | 따로 챙기는 요점 노트 |
| 게이트 (gate) | 정보를 얼마나 통과시킬지 정하는 문 | 기억을 넣을지, 지울지, 꺼낼지 정하는 문지기 |
| GRU | LSTM을 간단하게 줄인 셀 | 요점 노트 없이 메모 하나로 관리하는 사람 |
| 순환층 드롭아웃 | 순환층 안에서 일부 값을 끄는 것 | 메모 일부를 일부러 가리고 공부하기 |

---

## 9장 한 줄 요약

글처럼 순서가 중요한 데이터를 위해, 앞 내용을 기억하며 읽는 신경망(RNN, LSTM, GRU)을 만들고 영화 리뷰가 긍정인지 부정인지 맞힘.

> 💡 왜 필요함?
"재미없을 줄 알았는데 재미있다"와 "재미있을 줄 알았는데 재미없다"는 단어가 거의 같음.
순서를 무시하면 둘을 구별 못 함. 순환 신경망은 단어를 순서대로 읽으면서 앞 내용을 기억해서 이걸 구별함.
> 

---

## 9-1 순차 데이터와 순환 신경망

### 이 절에서 하는 일

- 순차 데이터가 뭔지, 왜 7·8장 모델로는 부족한지 이해함
- 순환 신경망이 "기억"을 어떻게 넘기는지 그림으로 봄
- 순환층의 입력·출력 모양과 파라미터 개수 세는 법을 익힘

### 순차 데이터

순차 데이터는 순서에 의미가 있는 데이터임.

- 글: 단어 순서가 바뀌면 뜻이 바뀜
- 날짜별 기온, 주가: 어제 값이 오늘 값에 영향을 줌

7·8장 모델(피드포워드 신경망)은 입력을 한 번에 다 넣고 바로 답을 냄.
앞에서 무엇을 봤는지 기억하는 장치가 없음.

### 순환 신경망

순환 신경망은 데이터를 **한 조각씩 순서대로** 넣음.
한 조각을 계산하고 나온 값(은닉 상태)을 **다음 조각 계산에 같이 넣음**.
그래서 앞 내용이 뒤로 계속 전달됨.

```
문장: "이 영화 정말 최고"

타임스텝 1: "이"     → 셀 → 은닉 상태 h1
타임스텝 2: "영화"   + h1 → 셀 → h2
타임스텝 3: "정말"   + h2 → 셀 → h3
타임스텝 4: "최고"   + h3 → 셀 → h4  → 이걸로 긍정/부정 판단
```

- 셀은 하나뿐임. 같은 셀을 타임스텝마다 반복해서 씀
- h4에는 "이 → 영화 → 정말 → 최고"를 다 읽은 요약이 들어 있음
- 맨 처음(타임스텝 1)에는 앞 내용이 없으니까 은닉 상태를 0으로 시작함

> 💡 비유
책을 한 문장씩 읽으면서 포스트잇에 요약을 적음.
다음 문장을 읽을 때 앞 포스트잇을 같이 보고 요약을 새로 고쳐 씀.
마지막 포스트잇을 보고 "이 책은 재밌다/재미없다"를 판단함.
> 

### 셀 안에서 하는 계산

셀은 두 가지를 받아서 계산함.

1. 지금 들어온 입력 × 입력용 가중치
2. 직전 은닉 상태 × 은닉 상태용 가중치

둘을 더하고 절편을 더한 뒤, **tanh**를 통과시켜서 새 은닉 상태를 만듦.

- tanh는 값을 -1~1 사이로 바꿔 줌
- 같은 계산을 계속 반복하니까, 값이 끝없이 커지지 않게 눌러 주는 역할임
- 순환층은 보통 ReLU 대신 tanh를 씀

### 입력과 출력 모양

순환층 입력은 숫자 3개로 된 모양임.

```
(샘플 수, 타임스텝 수, 특성 수)

예: 리뷰 20,000개, 리뷰마다 단어 100개, 단어 하나를 숫자 500개로 표현
→ (20000, 100, 500)
```

출력은 기본으로 **마지막 타임스텝의 은닉 상태만** 내보냄.

```
입력 (샘플, 100, 500) → 순환층 (뉴런 8개) → 출력 (샘플, 8)
```

- 순환층을 두 개 쌓을 때는 앞 층이 **모든 타임스텝**의 은닉 상태를 넘겨야 함
- 그때는 `return_sequences=True`를 씀 → 출력 (샘플, 100, 8)

### 파라미터 개수

```
순환층 파라미터 = 입력 특성 수 × 뉴런 수     (입력용 가중치)
               + 뉴런 수 × 뉴런 수          (은닉 상태용 가중치)
               + 뉴런 수                    (절편)
```

- 은닉 상태용 가중치가 "뉴런 수 × 뉴런 수"인 이유: 은닉 상태 크기가 뉴런 수와 같고, 이게 다시 같은 뉴런들로 들어가기 때문임

**숫자 예시** (입력 특성 4개, 뉴런 3개)

```
4 × 3 = 12   입력용 가중치
3 × 3 =  9   은닉 상태용 가중치
        3    절편
합계  = 24
```

### 결과 읽는 법

- 모양 `(샘플, 타임스텝, 특성)`에서 가운데 숫자가 "몇 조각을 순서대로 넣는지", 마지막 숫자가 "한 조각이 숫자 몇 개인지"임
- 순환층 출력이 `(샘플, 8)`이면 마지막 은닉 상태만 나온 것, `(샘플, 100, 8)`이면 모든 타임스텝이 나온 것임

### 자주 헷갈리는 것

| 헷갈리는 것 | 바른 이해 |
| --- | --- |
| 타임스텝마다 셀이 따로 있다 | 셀은 하나임. 같은 가중치를 반복해서 씀 |
| 은닉 상태 가중치는 "입력 수 × 뉴런 수" | "뉴런 수 × 뉴런 수"임. 은닉 상태 크기 = 뉴런 수 |
| 순환층은 모든 타임스텝을 출력한다 | 기본은 마지막 것만. 전부 필요하면 `return_sequences=True` |

### 이 절 핵심 3줄

1. 순환 신경망 = 한 조각씩 읽으면서 은닉 상태(기억)를 다음 조각으로 넘김
2. 입력 모양은 (샘플, 타임스텝, 특성). 출력은 기본으로 마지막 은닉 상태만
3. 파라미터 = 입력 특성 × 뉴런 + 뉴런 × 뉴런 + 뉴런

---

## 9-2 순환 신경망으로 IMDB 리뷰 분류하기

### 이 절에서 하는 일

- 영화 리뷰(IMDB)를 숫자로 바꾸고 길이를 맞춤
- 원-핫 인코딩으로 순환 신경망을 학습함
- 단어 임베딩으로 바꿔서 더 가볍고 좋은 모델을 만듦

### 글을 숫자로 바꾸는 방법

신경망은 글자를 못 읽음. 그래서 단어마다 번호를 붙임.

```
"this movie is great"  →  [14, 22, 9, 87]
```

- IMDB 데이터는 이미 번호로 바뀌어 있음
- 번호 몇 개는 정해진 뜻이 있음
    - 0: 패딩 (빈칸 채우기용)
    - 1: 문장 시작
    - 2: 어휘 사전에 없는 단어

### 패딩: 길이 맞추기

리뷰마다 길이가 다름. 신경망에 넣으려면 길이를 똑같이 맞춰야 함.

```
최대 길이 5로 맞추기

짧은 리뷰  [5, 12, 8]             →  [0, 0, 5, 12, 8]     앞에 0을 채움
긴 리뷰    [3, 9, 4, 7, 2, 6, 1]  →  [4, 7, 2, 6, 1]      앞부분을 잘라냄
```

- 기본은 **앞쪽**에 0을 채우고, **앞쪽**을 잘라냄
- 리뷰는 보통 끝부분에 결론(재밌다/별로다)이 있어서, 뒷부분을 남기는 게 유리하기 때문임

### 원-핫 인코딩

번호를 그냥 숫자로 넣으면, 번호가 큰 단어를 "더 큰 값"으로 오해함.
그래서 번호를 "그 자리만 1인 줄"로 바꿈.

```
어휘가 5개일 때

번호 0 → [1, 0, 0, 0, 0]
번호 3 → [0, 0, 0, 1, 0]
번호 4 → [0, 0, 0, 0, 1]
```

- 어휘가 500개면 단어 하나가 숫자 500개짜리 줄이 됨
- 대부분이 0이라서 낭비가 큼. 그래서 뒤에서 단어 임베딩으로 바꿈

### 코드 따라가기

**① 데이터 불러오기**

```python
import keras

(train_input, train_target), (test_input, test_target) = \
    keras.datasets.imdb.load_data(num_words=500)
```

- IMDB = 영화 리뷰 5만 개. 훈련 25,000개, 테스트 25,000개
- `num_words=500`: 자주 나오는 단어 500개만 씀. 나머지 단어는 전부 번호 2(사전에 없음)로 바뀜
- 정답(`train_target`)은 0(부정) 또는 1(긍정)

> ⚠️ 책에서 확인
`num_words` 값은 책 판에 따라 다를 수 있음. 이 노트는 500으로 계산함.
책 숫자가 다르면 아래 모양과 파라미터 계산에서 500 자리에 그 숫자를 넣으면 됨.
> 

**② 검증 세트 떼기**

```python
from sklearn.model_selection import train_test_split

train_input, val_input, train_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)
```

- 20%를 검증 세트로 뗌 → 훈련 20,000개, 검증 5,000개

**③ 리뷰 길이 살펴보기**

```python
import numpy as np
import matplotlib.pyplot as plt

lengths = np.array([len(x) for x in train_input])
print(np.mean(lengths), np.median(lengths))

plt.hist(lengths)
plt.xlabel("length")
plt.ylabel("frequency")
plt.show()
```

- `len(x)`: 리뷰 하나의 단어 수
- 평균과 중간값을 비교함. 평균이 중간값보다 크면 아주 긴 리뷰가 몇 개 섞여 있다는 뜻임
- 히스토그램을 보면 대부분 리뷰가 짧은 쪽에 몰려 있음. 그래서 최대 길이를 100 정도로 잘라도 됨

**④ 길이 100으로 맞추기**

```python
train_seq = keras.utils.pad_sequences(train_input, maxlen=100)
val_seq = keras.utils.pad_sequences(val_input, maxlen=100)
```

- `maxlen=100`: 모든 리뷰를 단어 100개로 맞춤
- 결과 모양: 훈련 (20000, 100), 검증 (5000, 100)

> ⚠️ 책이랑 다른 부분 (Keras 3)
책은 `from tensorflow.keras.preprocessing.sequence import pad_sequences`로 불러옴.
Keras 3에서는 `keras.utils.pad_sequences`로 바로 쓰면 됨.
> 

**⑤ 원-핫 인코딩**

```python
train_oh = keras.utils.to_categorical(train_seq)
val_oh = keras.utils.to_categorical(val_seq)
print(train_oh.shape)
```

- 단어 번호 하나를 숫자 500개짜리 줄로 바꿈
- 모양: (20000, 100) → **(20000, 100, 500)** = (샘플, 타임스텝, 특성)

> ⚠️ 주의
원-핫 결과는 숫자가 20,000 × 100 × 500 = 10억 개임. 컴퓨터 메모리를 많이 씀.
메모리가 부족하다는 에러가 나면 `num_words`나 `maxlen`을 줄여서 해 보기.
> 

**⑥ 순환 신경망 만들기**

```python
model = keras.Sequential()
model.add(keras.Input(shape=(100, 500)))
model.add(keras.layers.SimpleRNN(8))
model.add(keras.layers.Dense(1, activation="sigmoid"))
model.summary()
```

- `keras.Input(shape=(100, 500))`: 리뷰 하나 = 단어 100개 × 숫자 500개
- `SimpleRNN(8)`: 뉴런 8개짜리 기본 순환층. 활성화 함수는 기본으로 tanh
- `Dense(1, sigmoid)`: 긍정일 확률 하나를 냄 (이진 분류)

> ⚠️ 책이랑 다른 부분 (Keras 3)
책은 `SimpleRNN(8, input_shape=(100, 500))`처럼 층 안에 입력 모양을 넣음.
Keras 3에서는 맨 앞에 `keras.Input`을 따로 둠.
> 

**⑦ 학습**

```python
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rmsprop, loss="binary_crossentropy",
              metrics=["accuracy"])

checkpoint_cb = keras.callbacks.ModelCheckpoint("best-simplernn-model.keras",
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

history = model.fit(train_oh, train_target, epochs=100, batch_size=64,
                    validation_data=(val_oh, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])
```

- `learning_rate=1e-4`: 학습률 0.0001. 기본값(0.001)보다 작게 해서 천천히 안정적으로 배움
- `binary_crossentropy`: 정답이 0/1 두 가지인 이진 분류용 손실 함수
- `epochs=100`: 최대 100바퀴. 조기 종료가 있어서 보통 그 전에 멈춤
- `batch_size=64`: 한 번에 64개씩 보고 가중치를 고침 (기본 32)
- `patience=3`: 검증 손실이 3번 연속 안 좋아지면 멈춤

**⑧ 손실 곡선**

```python
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train", "val"])
plt.show()
```

- 훈련 손실과 검증 손실이 같이 내려가다가 벌어지는 지점을 보면 됨

### 단어 임베딩으로 바꾸기

원-핫은 단어 하나에 숫자 500개를 쓰고, 그중 499개가 0임.
단어 임베딩은 단어 하나를 **숫자 16개 정도의 짧은 줄**로 바꿈.

```
원-핫:    번호 3 → [0, 0, 0, 1, 0, 0, …, 0]       (500칸, 거의 다 0)
임베딩:   번호 3 → [0.12, -0.40, 0.05, …, 0.33]   (16칸, 전부 의미 있는 값)
```

- 임베딩 숫자는 처음엔 무작위인데, 학습하면서 바뀜
- 학습이 끝나면 "좋다"와 "훌륭하다"처럼 뜻이 비슷한 단어는 비슷한 숫자를 갖게 됨
- 입력은 원-핫으로 바꿀 필요 없이 **단어 번호 그대로** 넣으면 됨

> 💡 비유
원-핫은 학생 500명 출석부에서 한 명에만 동그라미 치는 것.
임베딩은 학생마다 "키, 성격, 취미…" 16개 특징 카드를 만들어 주는 것. 비슷한 학생은 카드도 비슷함.
> 

**⑨ 임베딩 모델**

```python
model2 = keras.Sequential()
model2.add(keras.Input(shape=(100,)))
model2.add(keras.layers.Embedding(500, 16))
model2.add(keras.layers.SimpleRNN(8))
model2.add(keras.layers.Dense(1, activation="sigmoid"))
model2.summary()
```

- `keras.Input(shape=(100,))`: 단어 번호 100개 (원-핫 안 함)
- `Embedding(500, 16)`: 단어 500종류를 각각 숫자 16개로 바꿈 → 출력 (샘플, 100, 16)
- 나머지는 ⑥과 같음
- 학습은 ⑦과 같은 방식으로 하되, 입력으로 `train_oh` 대신 `train_seq`를 넣음

> ⚠️ 책이랑 다른 부분 (Keras 3)
책은 `Embedding(500, 16, input_length=100)`처럼 씀.
Keras 3에서는 `input_length`가 없어졌음. 길이는 `keras.Input(shape=(100,))`으로 알려 줌.
> 

### 모양과 파라미터 수

**원-핫 모델**

| 층 | 결과 모양 | 파라미터 수 | 계산 |
| --- | --- | --- | --- |
| SimpleRNN (8) | (8,) | 4,072 | 500 × 8 + 8 × 8 + 8 |
| Dense (1) | (1,) | 9 | 8 × 1 + 1 |
| **합계** |  | **4,081** |  |

**임베딩 모델**

| 층 | 결과 모양 | 파라미터 수 | 계산 |
| --- | --- | --- | --- |
| Embedding (500, 16) | (100, 16) | 8,000 | 500 × 16 |
| SimpleRNN (8) | (8,) | 200 | 16 × 8 + 8 × 8 + 8 |
| Dense (1) | (1,) | 9 | 8 × 1 + 1 |
| **합계** |  | **8,209** |  |
- 임베딩 층 파라미터 = 단어 종류 수 × 줄 길이. 절편은 없음
- 순환층은 입력 특성이 500에서 16으로 줄어서 4,072개 → 200개로 확 줄었음
- 입력 데이터도 (100, 500)에서 (100,)으로 작아짐. 메모리를 훨씬 덜 씀

### 결과 읽는 법

- `fit` 출력의 `val_accuracy`가 검증 정확도임. 원-핫 모델과 임베딩 모델을 비교해 보면 됨
- 학습률을 작게 잡아서 에포크마다 조금씩 좋아짐. 조기 종료가 언제 멈추는지 보기
- 손실 곡선에서 검증 손실이 다시 올라가기 시작하는 지점 근처에서 멈췄으면 정상임

### 자주 나는 에러

| 증상 | 원인 | 해결 |
| --- | --- | --- |
| 메모리 부족으로 멈춤 | 원-핫 결과가 너무 큼 | `num_words`나 `maxlen`을 줄이거나 임베딩 모델 사용 |
| `input_length` 관련 에러 | Keras 3에서 없어진 인자 | 지우고 `keras.Input(shape=(100,))` 사용 |
| 입력 모양이 안 맞는다는 에러 | 원-핫 모델에 `train_seq`를 넣거나, 임베딩 모델에 `train_oh`를 넣음 | 원-핫 모델 ↔ `train_oh`, 임베딩 모델 ↔ `train_seq` |
| `pad_sequences`를 못 찾음 | 책의 불러오는 경로가 Keras 3와 다름 | `keras.utils.pad_sequences` |

### 이 절 핵심 3줄

1. 글 → 단어 번호 → `pad_sequences`로 길이 맞추기 (앞에 0 채우고 앞을 자름)
2. 원-핫은 단어 하나를 500칸으로, 임베딩은 16칸 정도의 의미 있는 숫자로 바꿈
3. 이진 분류라서 `Dense(1, sigmoid)` + `binary_crossentropy`

---

## 9-3 LSTM과 GRU 셀

### 이 절에서 하는 일

- 기본 순환층(SimpleRNN)이 긴 글에서 앞 내용을 잊어버리는 문제를 이해함
- 오래 기억하는 LSTM, 더 간단한 GRU를 써 봄
- 순환층 드롭아웃과 순환층 2개 쌓기를 해 봄

### 기본 순환층의 문제

기본 순환층은 은닉 상태 하나에 모든 걸 계속 덮어씀.
그래서 글이 길어지면 앞부분 내용이 점점 희미해짐.

```
"처음엔 지루했지만 … (단어 90개) … 결국 최고였다"
→ 끝에 갈수록 "처음엔 지루했지만"이 거의 안 남음
```

### LSTM

LSTM은 기억 통로를 **두 개** 들고 다님.

- **은닉 상태**: 지금 당장 쓰는 짧은 기억 (기본 순환층과 같음)
- **셀 상태**: 오래 간직하는 긴 기억. 다음 층으로 내보내지 않고 셀 안에서만 넘어감

그리고 **게이트(문)** 로 기억을 관리함.

| 게이트 | 하는 일 | 비유 |
| --- | --- | --- |
| 삭제 게이트 | 셀 상태에서 필요 없는 기억을 지움 | 요점 노트에서 지울 줄 고르기 |
| 입력 게이트 | 새 정보 중 기억할 걸 셀 상태에 더함 | 요점 노트에 새로 적을 것 고르기 |
| 출력 게이트 | 셀 상태에서 이번에 꺼내 쓸 걸 골라 은닉 상태로 내보냄 | 요점 노트에서 지금 볼 부분 펴기 |

> 💡 비유
기본 순환층은 포스트잇 한 장에 계속 고쳐 쓰는 사람.
LSTM은 포스트잇 + 요점 노트를 같이 쓰는 사람. 요점 노트에 중요한 건 오래 남겨 둠.
> 

**LSTM 파라미터 수**

LSTM 안에는 작은 셀 계산이 **4개**(게이트 3개 + 새 기억 후보 1개) 있음.
각각이 기본 순환층 하나와 같은 크기의 가중치를 가짐.

```
LSTM 파라미터 = 4 × (입력 특성 × 뉴런 + 뉴런 × 뉴런 + 뉴런)

입력 16개(임베딩), 뉴런 8개:
4 × (16 × 8 + 8 × 8 + 8) = 4 × 200 = 800
```

### GRU

GRU는 LSTM을 간단하게 줄인 셀임.

- 셀 상태가 없고 은닉 상태 하나만 씀
- 게이트 2개 + 새 기억 후보 1개 = 작은 셀 계산 **3개**
- LSTM보다 가볍고, 성능은 비슷한 경우가 많음

**GRU 파라미터 수**

```
GRU 파라미터 = 3 × (입력 특성 × 뉴런 + 뉴런 × 뉴런 + 뉴런 + 뉴런)

입력 16개, 뉴런 8개:
3 × (16 × 8 + 8 × 8 + 8 + 8) = 3 × 208 = 624
```

- 마지막에 "+ 뉴런"이 한 번 더 붙는 이유: 케라스 GRU는 절편을 두 벌 씀 (기본 설정)
- 그래서 "3 × 200 = 600"이 아니라 624가 나옴

### 코드 따라가기

앞의 데이터 준비(①~④)는 9-2와 같음. `train_seq`, `val_seq`를 씀.

**① LSTM 모델**

```python
model = keras.Sequential()
model.add(keras.Input(shape=(100,)))
model.add(keras.layers.Embedding(500, 16))
model.add(keras.layers.LSTM(8))
model.add(keras.layers.Dense(1, activation="sigmoid"))
model.summary()
```

- `LSTM(8)`: SimpleRNN 자리에 LSTM만 바꿔 넣으면 됨
- 학습 코드는 9-2 ⑦과 같음. 체크포인트 파일 이름만 `"best-lstm-model.keras"`처럼 바꿈

**② 순환층 드롭아웃**

```python
model2 = keras.Sequential()
model2.add(keras.Input(shape=(100,)))
model2.add(keras.layers.Embedding(500, 16))
model2.add(keras.layers.LSTM(8, dropout=0.3))
model2.add(keras.layers.Dense(1, activation="sigmoid"))
```

- `dropout=0.3`: 순환층에 들어오는 입력의 30%를 학습할 때 무작위로 끔
- 7장의 `Dropout` 층처럼 과대적합을 줄임. 다만 따로 층을 넣지 않고 순환층 안에 옵션으로 줌
- 파라미터 수는 안 바뀜 (800 그대로)

**③ 순환층 2개 쌓기**

```python
model3 = keras.Sequential()
model3.add(keras.Input(shape=(100,)))
model3.add(keras.layers.Embedding(500, 16))
model3.add(keras.layers.LSTM(8, dropout=0.3, return_sequences=True))
model3.add(keras.layers.LSTM(8, dropout=0.3))
model3.add(keras.layers.Dense(1, activation="sigmoid"))
model3.summary()
```

- 첫 번째 LSTM에 `return_sequences=True`: 모든 타임스텝의 은닉 상태를 내보냄 → (100, 8)
- 두 번째 LSTM은 그걸 받아서 마지막 은닉 상태만 내보냄 → (8,)
- 첫 번째에 `return_sequences=True`가 없으면 두 번째 LSTM이 받을 "순서"가 없어서 에러가 남

**④ GRU 모델**

```python
model4 = keras.Sequential()
model4.add(keras.Input(shape=(100,)))
model4.add(keras.layers.Embedding(500, 16))
model4.add(keras.layers.GRU(8))
model4.add(keras.layers.Dense(1, activation="sigmoid"))
model4.summary()
```

- `GRU(8)`: LSTM 자리에 GRU만 바꿔 넣음

**⑤ 테스트 세트로 마지막 평가**

```python
test_seq = keras.utils.pad_sequences(test_input, maxlen=100)
rnn_model = keras.models.load_model("best-2rnn-model.keras")
rnn_model.evaluate(test_seq, test_target)
```

- 테스트 세트도 훈련 세트와 똑같이 길이 100으로 맞춤
- 가장 좋았던 모델 파일을 불러와서 마지막에 한 번만 평가함
- 파일 이름은 ③ 모델을 학습할 때 체크포인트에 쓴 이름으로 맞추면 됨

### 모양과 파라미터 수

| 모델 | 층별 파라미터 | 합계 |
| --- | --- | --- |
| LSTM | Embedding 8,000 + LSTM 800 + Dense 9 | **8,809** |
| LSTM + 드롭아웃 | 위와 같음 (드롭아웃은 파라미터 없음) | **8,809** |
| LSTM 2개 | Embedding 8,000 + LSTM 800 + LSTM 544 + Dense 9 | **9,353** |
| GRU | Embedding 8,000 + GRU 624 + Dense 9 | **8,633** |

```
두 번째 LSTM은 입력이 8개 (첫 번째 LSTM의 뉴런 수)
4 × (8 × 8 + 8 × 8 + 8) = 4 × 136 = 544
```

**LSTM 2개 모델 모양 변화**

```
(100,) → Embedding → (100, 16) → LSTM(return_sequences=True) → (100, 8) → LSTM → (8,) → Dense → (1,)
```

### 결과 읽는 법

- 기본 순환층, LSTM, GRU 모델의 `val_loss`, `val_accuracy`를 비교함
- 드롭아웃을 넣으면 훈련 손실과 검증 손실 사이가 덜 벌어지는지 손실 곡선으로 확인함
- LSTM과 GRU는 SimpleRNN보다 학습이 조금 느림. 계산할 게 3~4배 많기 때문임

### 자주 나는 에러

| 증상 | 원인 | 해결 |
| --- | --- | --- |
| 두 번째 LSTM에서 입력이 3차원이어야 한다는 에러 | 첫 번째 LSTM에 `return_sequences=True`를 안 씀 | 앞쪽 순환층에 `return_sequences=True` |
| GRU 파라미터가 계산(600)과 다름 | 케라스 GRU는 절편을 두 벌 씀 | 3 × (입력 × 뉴런 + 뉴런 × 뉴런 + 뉴런 × 2)로 계산 |
| 저장한 모델 파일을 못 찾음 | 체크포인트 이름과 `load_model` 이름이 다름 | 같은 이름으로 맞추고 `.keras`로 끝내기 |
| 테스트 점수가 이상하게 낮음 | 테스트 세트에 `pad_sequences`를 안 함 | 훈련 세트와 똑같이 `maxlen=100` |

### 이 절 핵심 3줄

1. LSTM = 은닉 상태 + 셀 상태(긴 기억) + 게이트 3개. 파라미터는 기본 순환층의 4배
2. GRU = LSTM을 줄인 것. 셀 상태 없음, 계산 3개. 케라스에선 절편이 두 벌이라 3 × 208
3. 순환층을 쌓을 땐 앞 층에 `return_sequences=True`, 과대적합은 `dropout=` 옵션으로 줄임

---

## 9장 전체 흐름 정리

| 단계 | 하는 일 | 핵심 코드 |
| --- | --- | --- |
| 1. 데이터 불러오기 | 자주 나오는 단어만 남김 | `keras.datasets.imdb.load_data(num_words=500)` |
| 2. 검증 세트 | 20% 떼어 냄 | `train_test_split(..., test_size=0.2)` |
| 3. 길이 맞추기 | 앞에 0 채우고 앞을 자름 | `keras.utils.pad_sequences(train_input, maxlen=100)` |
| 4-1. 원-핫 (방법 1) | 단어 → 500칸 줄 | `keras.utils.to_categorical(train_seq)` |
| 4-2. 임베딩 (방법 2) | 단어 → 16칸 의미 있는 숫자 | `keras.layers.Embedding(500, 16)` |
| 5. 순환층 | 순서대로 읽으며 기억 | `SimpleRNN(8)` / `LSTM(8)` / `GRU(8)` |
| 6. 과대적합 막기 | 순환층 안 드롭아웃 | `LSTM(8, dropout=0.3)` |
| 7. 순환층 쌓기 | 앞 층이 모든 타임스텝 전달 | `LSTM(8, return_sequences=True)` |
| 8. 판정 | 긍정 확률 하나 | `Dense(1, activation="sigmoid")` |
| 9. 학습 | 이진 분류 손실 + 조기 종료 | `loss="binary_crossentropy"`, `EarlyStopping(patience=3)` |
| 10. 평가 | 마지막에 테스트 세트로 | `model.evaluate(test_seq, test_target)` |

**순환층 파라미터 공식 한눈에** (입력 특성 = i, 뉴런 = u)

```
SimpleRNN :     i × u + u × u + u
LSTM      : 4 × (i × u + u × u + u)
GRU       : 3 × (i × u + u × u + u + u)
Embedding : 단어 종류 수 × 줄 길이
```