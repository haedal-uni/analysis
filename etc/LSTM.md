# LSTM 시계열 예측 모델 정리
시계열 데이터란?
→ 시간에 따라 변하는 데이터
예: 매일의 주가, 매시간의 기온, 매초의 센서 값 등

100 → 101 → 103 → 105 → 108 → ...
이런 식으로 시간 순서대로 줄줄이 이어지는 데이터가 시계열.


## LSTM이란?

**LSTM(Long Short-Term Memory)** 은 순차적인 데이터를 처리하는 데 강력한 딥러닝 모델이다.

과거 정보(예: 주가 흐름)를 기억해서, 미래를 예측할 수 있도록 만들어졌다.

RNN의 일종인데, RNN보다 기억력이 더 좋고, 장기 의존성 문제를 해결했다.

> “이전 며칠간의 주가 흐름을 잘 기억하고, 내일의 주가를 맞춰보자!” 라고 말하는 모델

---

## 시퀀스(Sequence)란?

순서가 있는 데이터 묶음이다.

ex) 

- [100, 101, 103, 105, 106] → 주가 시퀀스
- [‘안’, ‘녕’, ‘하’, ‘세’, ‘요’] → 글자 시퀀스

---

## 코드 흐름 요약

1. CSV 파일 읽기 (날짜를 인덱스로 설정)
2. 결측치 처리, 문자열 → 숫자로 변환 (거래량, 변동%)
3. 종가 그래프 시각화
4. 정규화 (MinMaxScaler)
5. train/test 데이터 분할
6. y_train, y_test를 기반으로 LSTM 학습용 시퀀스 구성
7. 모델 정의 (Conv1D + LSTM + Dense)
8. 훈련 (EarlyStopping & Checkpoint 사용)
9. 예측 결과 시각화 및 해석

---

## 시퀀스 구성 방식
- WINDOW_SIZE = 20이면 20일치 종가 → 1일치 예측

예:X: `[100, 101, ..., 119]` Y: 120

---

## 핵심 개념 요약
- **MinMaxScaler**: 데이터 범위를 0~1로 정규화
- **train_test_split**: 학습용 / 테스트용 분할
- **windowed_dataset()**: 시계열 데이터를 (20개 입력 → 1개 출력) 형태로 변환
- **Conv1D**: 1차원 필터로 시계열 흐름의 특징 추출
- **LSTM**: 순서를 고려하여 시계열 패턴 학습
- **Dense**: 출력층
- **ModelCheckpoint / EarlyStopping**: 모델 저장 + 과적합 방지

---

## Hyperparameter 정리

| 하이퍼파라미터 | 설명 |
|----------------|------|
| WINDOW_SIZE     | 몇 일치의 데이터를 한 묶음으로 볼지 (ex: 20일) |
| BATCH_SIZE      | 몇 묶음을 한 번에 학습할지 (ex: 32개 시퀀스) |
| LSTM 노드 수    | LSTM 레이어의 뉴런 개수 (ex: 16) |
| learning rate   | 학습 속도 (ex: 0.0005) |
| epochs          | 학습 반복 횟수 (ex: 50) |
| optimizer       | 가중치 최적화 방법 (Adam 등) |

---

## 왜 Conv1D?

주가 데이터는 시간 순서대로 나열된 **1차원 시계열**이다.

예: [100, 102, 101, 103, …]

→ 시간 축의 패턴을 뽑기 위해 **1차원 Conv 사용**      
→ 음성, 센서, 주가 등 시계열에 많이 사용

---

## Huber Loss란?

Huber 손실 함수는 **MSE**와 **MAE**의 장점만 결합한 손실 함수다.

- MSE: 이상치에 매우 민감
- MAE: 이상치에 둔감하지만, 기울기 변화가 불연속
- Huber: 둘의 장점을 섞어 안정적

## 코드
이 프로젝트는 주가의 '종가'를 예측하는 문제를 다루며, LSTM 기반 시계열 예측 모델을 구성

모델의 성능은 val_loss, mse를 기준으로 평가되며 예측값과 실제값을 시각화하여 비교

데이터 로드 및 전처리: CSV 파일에서 데이터를 로드하고, 날짜를 인덱스로 설정하며, 필요한 경우 문자열 데이터를 수치형으로 변환합니다.

시각화: Matplotlib과 Seaborn을 사용하여 데이터의 추이를 시각화합니다.

정규화: MinMaxScaler를 사용하여 데이터의 값을 0과 1 사이로 스케일링하여 모델 학습

### 1. 데이터 로딩 및 전처리
```py
import pandas as pd
df = pd.read_csv('미국 철강 코일 선물 과거 데이터.csv', parse_dates=['날짜'], index_col="날짜", thousands=",") # 구분자 제거
df = df.sort_index()
df.head()
```
|날짜|종가|시가|고가|저가|거래량|변동 %|
|---|---|---|---|---|---|---|
|2015-01-02 00:00:00|605\.0|605\.0|605\.0|605\.0|NaN|0\.00%|
|2015-01-05 00:00:00|605\.0|605\.0|605\.0|605\.0|NaN|0\.00%|
|2015-01-06 00:00:00|605\.0|605\.0|605\.0|605\.0|NaN|0\.00%|
|2015-01-07 00:00:00|597\.0|597\.0|597\.0|597\.0|0\.05K|-1\.32%|
|2015-01-08 00:00:00|597\.0|597\.0|597\.0|597\.0|NaN|0\.00%|

#### 문자열을 숫자로 바꾸기 
```py
df['거래량'] = df['거래량'].apply(lambda x: float(x.replace('K', '')) * 1000 if isinstance(x, str) and 'K' in x else float(x))
df['변동 %'] = df['변동 %'].apply(lambda x: float(x.replace('%', '')) / 100 if isinstance(x, str) else x)
```

### 2. 데이터 시각화  
```py
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(16, 9))
sns.lineplot(y=df['종가'], x=df.index)
plt.xlabel('time')
plt.ylabel('price')
```

#### 시계열 구간별 시각화  
```py
df.index = pd.to_datetime(df.index)

time_steps = [['2015', '2018'], 
              ['2018', '2021'], 
              ['2021', '2024'], 
              ['2024', '2025']]

fig, axes = plt.subplots(2, 2)
fig.set_size_inches(16, 9)
for i in range(4):
    ax = axes[i//2, i%2]
    data = df.loc[(df.index > time_steps[i][0]) & (df.index < time_steps[i][1])]
    sns.lineplot(y=data['종가'], x=data.index, ax=ax)
    ax.set_title(f'{time_steps[i][0]}~{time_steps[i][1]}')
    ax.set_xlabel('time')
    ax.set_ylabel('price')
plt.tight_layout()
plt.show()
```

### 3. 데이터 정규화 (MinMaxScaler)
주가 데이터에 대하여 딥러닝 모델이 더 잘 학습할 수 있도록 정규화(Normalization)를 해준다.

**MinMaxScaler**로 0~1 범위로 스케일링  

```py
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# 스케일을 적용할 column을 정의
scale_cols = ['종가', '시가', '고가', '저가', '거래량']

# 스케일 후 columns
scaled = scaler.fit_transform(df[scale_cols])
scaled
```

### 4. Train/Test 분리

```py
df = pd.DataFrame(scaled, columns=scale_cols)

# train, test 분할
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.drop('종가', axis=1), df['종가'], test_size=0.2, random_state=0, shuffle=False)
x_train.shape, y_train.shape # ((2078, 4), (2078,))
x_test.shape, y_test.shape # ((520, 4), (520,))
```
x_train.shape = (2078, 4) : 2078개의 샘플, 4개의 feature (시가, 고가, 저가, 거래량)

y_train.shape = (2078,): 각 샘플의 예측 대상 종가 값 1개
→ 즉, y는 1차원 배열이며 각 값이 하나의 출력 (정답)

(2078,) vs (2078, 1)	LSTM은 최소 2D가 필요하므로 (2078,)은 (2078, 1)로 reshape

<br>

- Tensor의 차원
  - (2078,): 1D 텐서 → 값이 하나씩만 존재하는 배열 (예: y값). 쉼표를 사용하여 튜플임을 나타냄

  - (2078, 1): 2D 텐서 → y값을 LSTM에 넣기 위한 구조로 reshape한 형태 (2D 이상 필요). 두 개의 요소를 가진 튜플

   - (2078,)와 (2078, 1)은 서로 다른 구조

### 5. 시퀀스 데이터셋 구성
💡 “연속된 데이터 포인트를 윈도우로 묶는다”는 뜻?
→ 데이터를 일정한 간격(예: 20일)으로 잘라서 묶는다는 뜻입니다.

예를 들어, 100일치 주가가 있다고 가정  .
여기서 앞의 20일을 보고, 21번째 값을 예측하고 싶다면?

X: [1~20일 주가], y: 21일 주가  
X: [2~21일 주가], y: 22일 주가  
X: [3~22일 주가], y: 23일 주가  
...
이렇게 일정한 길이의 슬라이딩 윈도우를 만든다.  
이때 window_size = 20이면 20개의 데이터를 보고 1개를 예측하는 구조가 된다.  

💡 그럼 윈도우 객체를 하나의 텐서로 묶는다는 건?           
→ 처음에는 윈도우가 작은 덩어리로 나눠진 객체 상태.     

TensorFlow에서는 이를 window()로 만들고         
flat_map()과 batch()를 사용해서      
**하나의 텐서 (즉, 숫자 덩어리)**로 만들어준다.
 
py
복사
편집
ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
ds = ds.flat_map(lambda w: w.batch(window_size + 1))
즉,

window: 데이터를 잘라냄

batch: 잘라낸 조각을 하나의 텐서로 묶음

flat_map: 이 묶은 것들을 평평하게 펼침 (flatten)

배치(batch) : 학습할 때 한 번에 모델에 넣는 데이터 묶음.

예를 들어 1개의 데이터씩 넣으면 너무 느리니까, 한 번에 32개씩 넣는 식으로 처리.

BATCH_SIZE = 32 → 배치가 크면 학습 속도가 빨라지지만, 메모리도 많이 먹어요.




```py
import tensorflow as tf
def windowed_dataset(series, window_size, batch_size, shuffle):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series) # TensroFlow Dataset으로 변환 (한 줄씩 슬라이스)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True) 
    ds = ds.flat_map(lambda w: w.batch(window_size + 1)) # window 객체를 하나의 텐서로 묶음
    if shuffle:
        ds = ds.shuffle(1000) # 학습 데이터 섞기 
    ds = ds.map(lambda w: (w[:-1], w[-1])) # 입력 X: 앞 20개, 출력 Y: 마지막 1개 
    return ds.batch(batch_size).prefetch(1) # 배치 단위로 묶고, 성능 향상을 위해 prefetch.
    # prefetch() : 성능 개선을 위한 병렬 처리

WINDOW_SIZE=20
BATCH_SIZE=32

# Hyperparameter를 정의
# trian_data는 학습용 데이터셋, test_data는 검증용 데이터셋이다.
train_data = windowed_dataset(y_train, WINDOW_SIZE, BATCH_SIZE, True)
test_data = windowed_dataset(y_test, WINDOW_SIZE, BATCH_SIZE, False)

# 아래의 코드로 데이터셋의 구성을 확인해 볼 수 있다.
# X: (batch_size, window_size, feature)
# Y: (batch_size, feature)
for data in train_data.take(1):
    print(f'데이터셋(X) 구성(batch_size, window_size, feature갯수): {data[0].shape}')
    print(f'데이터셋(Y) 구성(batch_size, window_size, feature갯수): {data[1].shape}')
```
- windowed_dataset() : 시계열 데이터를 LSTM 학습에 적합한 형태로 바꾸는 함수

- 윈도우 객체를 하나의 텐서로 묶는다는 것의 의미와 관련 코드
  
시계열 데이터를 다룰 때, 연속된 데이터 포인트를 하나의 윈도우(window)로 묶어 입력과 출력으로 사용하는 것이 일반적이다.

TensorFlow의 flat_map과 batch를 사용하여 이를 구현할 수 있습니다:

```python
ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
ds = ds.flat_map(lambda w: w.batch(window_size + 1))
```
여기서 window는 데이터셋을 지정된 크기의 윈도우로 나누는 역할을 하며, 

flat_map은 이 윈도우들을 하나의 배치로 묶습니다. 

이를 통해 모델에 입력할 시퀀스 데이터를 생성할 수 있습니다.


- LSTM은 2D 이상 입력이 필요하므로 차원 추가(ex. 데이터를 `[100, 101, 102, ...]` → `[[100], [101], [102], ...]`)

- ds.window(window_size + 1, shift=1, drop_remainder=True) : 시퀀스를 생성

  - ex: window_size=20이면 21일 데이터를 한 묶음으로 생성 → 20개는 입력, 마지막 1개는 정답(Y)

### 6. LSTM 모델 설계
```py
# 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Lambda
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential([
    # 1차원 feature map 생성
    Conv1D(filters=32, kernel_size=5, # 5개 시점 간의 패턴을 학습하는 1D CNN 필터 32개 사용
           padding="causal", # 미래 정보 안 보도록 설정 (시계열 예측 필수)
           activation="relu",
           input_shape=[WINDOW_SIZE, 1]),
    # LSTM
    LSTM(16, activation='tanh'), # LSTM(16): 은닉 상태 차원 수가 16 (과거 정보 기억 용량)
    Dense(16, activation="relu"),
    Dense(1),
])

# Sequence 학습에 비교적 좋은 퍼포먼스를 내는 Huber()를 사용.
loss = Huber() # Huber : 이상치에 덜 민감한 손실 함수
optimizer = Adam(0.0005) # Adam: 잘 쓰이는 최적화 알고리즘 
model.compile(loss=Huber(), optimizer=optimizer, metrics=['mse'])

# earlystopping은 10번 epoch 동안 val_loss 개선이 없다면 학습을 멈춘다.
earlystopping = EarlyStopping(monitor='val_loss', patience=10) # EarlyStopping: 과적합 방지, val_loss 개선 없으면 조기 종료

# val_loss 기준 체크포인터도 생성. ModelCheckpoint: 가장 성능 좋은 모델 저장
filename = os.path.join('tmp', 'checkpointer.weights.h5')
checkpoint = ModelCheckpoint(filename, 
                             save_weights_only=True, 
                             save_best_only=True, 
                             monitor='val_loss', # 검증 손실 기준으로 가장 좋은 모델을 저장
                             verbose=1)

history = model.fit(train_data, 
                    validation_data=(test_data), 
                    epochs=50, 
                    callbacks=[checkpoint, earlystopping])
```

model 
- Conv1D: 시계열 데이터의 국소 패턴 추출

- padding='causal': 미래를 보지 않게끔 함 (시계열 예측에 중요)

- LSTM(16): 과거 정보를 기억하며 시계열 흐름을 학습

- Dense(16): 은닉층

- Dense(1): 출력층, 종가 1개 예측

- filters=32: 출력 채널 수, 즉 합성곱 필터의 수를 32개로 설정합니다.

- kernel_size=5: 각 필터의 크기를 5로 설정합니다.

- padding="causal": 인과적 패딩을 사용하여 시간 순서를 유지하며, 현재 시점의 출력이 미래 시점의 입력에 영향을 받지 않도록 합니다.

- activation="relu": 활성화 함수로 ReLU를 사용합니다.

- `input_shape=[WINDOW_SIZE, 1]` : 입력 데이터의 형태를 정의한다.

  여기서 WINDOW_SIZE는 타임스텝의 수를, 1은 특성(feature) 수를 나타낸다.

<br>  

val_loss는 검증 데이터셋(validation set)에 대한 손실(loss)을 의미

monitor='val_loss'는 검증 손실을 모니터링하며 save_best_only=True는 가장 낮은 검증 손실을 기록한 모델만 저장하도록 설정

과적합(overfitting)은 모델이 학습 데이터에 너무 맞춰져서 새로운 데이터에 대한 일반화 성능이 떨어지는 현상

이는 모델이 학습 데이터의 노이즈나 세부 사항까지 학습하여 발생하며, 새로운 데이터에 대한 예측 정확도가 낮아지는 결과를 초래한다.


### 7. 모델 평가 및 시각화
```py
# 저장한 ModelCheckpoint 를 로드
model.load_weights(filename)

# test_data를 활용하여 예측을 진행
pred = model.predict(test_data)
pred.shape

'''
예측 데이터 시각화
아래 시각화 코드중 y_test 데이터에 [20:]으로 슬라이싱을 한 이유는

예측 데이터에서 20일치의 데이터로 21일치를 예측해야하기 때문에 test_data로 예측 시 앞의 20일은 예측하지 않습니다.

따라서, 20번 째 index와 비교하면 더욱 정확합니다.
'''
plt.figure(figsize=(12, 9))
plt.plot(np.asarray(y_test)[20:], label='actual')
plt.plot(pred, label='prediction')
plt.legend()
plt.show()
```
 X는 과거 20일 종가, Y는 그 다음 날 종가가 된다.  
이걸 하나의 시퀀스라고 부르며, 수천 개의 시퀀스를 만들어 모델을 학습한다.


✔️ 두 선이 비슷하게 움직인다면 → 예측을 잘했다!

✔️ 너무 차이가 크면 → 모델이 학습을 잘 못했다.

📌 정규화된 값이기 때문에 소수점이 나오는 거예요. 원래 종가로 되돌리려면 inverse_transform()을 사용해야 합니다.





마무리 핵심 요약
질문	요약 답변
이 코드는?	LSTM으로 종가 예측하는 딥러닝 모델
예측 대상?	정규화된 종가 데이터
LSTM이란?	순서 있는 데이터(시계열)를 예측하는 데 강한 딥러닝 모델
시퀀스란?	순서 있는 데이터 묶음 (20개 종가 → 1개 종가 예측)
WINDOW_SIZE?	시퀀스의 길이 (입력 데이터 개수)
BATCH_SIZE?	한 번에 학습할 시퀀스 묶음 수
Huber()란?	이상치에 강한 손실 함수
Conv1D는 왜 쓰나요?	시계열 데이터의 패턴을 1차원 필터로 추출
예측 시 소수점 나오는 이유?	정규화된 값이기 때문 (원래 값 아님)
