## Google 주가 예측
- RNN, GRU, LSTM

- Date, Open, High, Low, Close, Volume

```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("Google_Stock_Price_Train.csv")
df.head()
```
| **Date** | **Open** | **High** | **Low** | **Close** | **Volume** |
| --- | --- | --- | --- | --- | --- |
| **0** | 1/3/2012 | 325.25 | 332.83 | 324.97 | 663.59 |
| **1** | 1/4/2012 | 331.27 | 333.87 | 329.08 | 666.45 |
| **2** | 1/5/2012 | 329.83 | 330.75 | 326.89 | 657.21 |
| **3** | 1/6/2012 | 328.34 | 328.77 | 323.68 | 648.24 |
| **4** | 1/9/2012 | 322.04 | 322.29 | 309.46 | 620.76 |

<br><br><br>

```
df.shape
df.info()
```
```
(1258, 6)

 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   Date    1258 non-null   object 
 1   Open    1258 non-null   float64
 2   High    1258 non-null   float64
 3   Low     1258 non-null   float64
 4   Close   1258 non-null   object 
 5   Volume  1258 non-null   object 
dtypes: float64(3), object(3)
```
null 값이 없는 것을 확인한다.

<br><br><br>


```py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU
from sklearn.preprocessing import MinMaxScaler

train = df.loc[:, ["Open"]].values  # 종가 대신 시가(Open)만 뽑아서 numpy 배열로 변환
scaler = MinMaxScaler(feature_range=(0, 1))  # 정규화(데이터를 0 ~ 1 범위로 바꾸기 위해 MinMaxScaler를 설정)
train_scaled = scaler.fit_transform(train)  # 0 ~ 1로 스케일링(데이터를 학습시킬 수 있도록 정규화(스케일링))
```
```
array([[325.25],
       [331.27],
       [329.83],
       ...,
       [793.7 ],
       [783.33],
       [782.75]])

array([[0.08581368],
       [0.09701243],
       [0.09433366],
       ...,
       [0.95725128],
       [0.93796041],
       [0.93688146]])
```

<br><br><br>

```py
# 타임스텝 데이터셋 생성 함수
def create_timesteps(ts=50): # ts: 타임스텝 → 과거 50일 사용
    X_train, y_train = [], []  # 입력(X), 출력(y) 배열 초기화
    for i in range(ts, 1258):  # 50번째 ~ 1257번째까지 반복 (총 1258개 데이터 기준)
        X_train.append(train_scaled[i-ts:i, 0])  # 과거 ts개의 데이터를 X로(과거 50일간 데이터를 X로)
        y_train.append(train_scaled[i, 0])       # 바로 다음 시점을 y로

    X_train = np.array(X_train).reshape(-1, ts, 1)  # (샘플 수, 타임스텝, 1)
    return X_train, np.array(y_train), ts
```
- i가 100이라고 하면, X_train에는 50~99번 데이터가 들어가고

    y_train에는 100번 데이터(=다음 날 시가)가 들어간다.

- 이런 방식으로 과거 데이터를 입력으로 다음 날 데이터를 출력으로 만들어 학습용 데이터셋을 구성한다.

- reshape(-1, ts, 1):

  - reshape은 데이터의 모양을 바꾸는 것.

  - -1: 샘플 수 (자동으로 계산됨. 예: 몇 개 만들지 몰라도 자동 계산)

  - ts: 한 샘플당 몇 개의 시간 데이터를 가지고 있는지 (예: 과거 50일)

  - 1: 특성 수 (여기서는 시가 하나만 쓰니까 1개)

  - 즉, 모양은 (샘플 수, 50, 1)이 된다.  

<br><br><br>


### 1) RNN 모델
```py
X_train, y_train, timesteps = create_timesteps(ts=50)

modelRNN = Sequential()
modelRNN.add(SimpleRNN(timesteps, activation="relu", input_shape=(50, 1)))
modelRNN.add(Dense(1))  # 다음 시점의 가격 1개 예측
modelRNN.compile(optimizer="ADAM", loss="mse")
modelRNN.fit(X_train, y_train, epochs=100)
```
loss: 2.8382e-04

- 입력: 과거 50일간의 시가

- 출력: 다음 날 시가 (회귀)

- 모델: SimpleRNN → Dense

- 용도: 단기 시계열 예측

<br><br><br>

### 2) LSTM 모델
```py
X_train, y_train, timesteps = create_timesteps(ts=10)

model = Sequential()
model.add(LSTM(timesteps, input_shape=(10, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=50)
```

loss: 8.4960e-04

- 입력: 과거 10일 시가

- 출력: 다음 날 시가

- 모델: LSTM → Dense

<br><br><br>

### 3) GRU 모델
```py
X_train, y_train, timesteps = create_timesteps(ts=15)

model_gru_ts = Sequential()
model_gru_ts.add(GRU(timesteps, return_sequences=True, input_shape=(50, 1)))
model_gru_ts.add(GRU(50))
model_gru_ts.add(Dense(1))
model_gru_ts.compile(optimizer="ADAM", loss="mse")
model_gru_ts.fit(X_train, y_train, epochs=100)
```

3.8390e-04

- 입력: 과거 50일 시가

- 모델: GRU 2층 + Dense

- 특징: return_sequences=True로 중간층 출력 유지 → 시계열 길이 유지

<br><br>

(추가 분석)

RNN이 세 가지 모델 중에서 가장 좋은 성능을 보였기 때문에

RNN 모델을 최종 선택해서 시각화와 예측 결과 분석을 진행한 코드

학습 데이터 전체에 대해 예측 → 시각화, MAE, MSE 출력해서 수치적으로 평가

```py
import matplotlib.pyplot as plt

# 학습 데이터의 마지막 50일을 기준으로 예측용 데이터 준비
input_data = train_scaled[-50:].reshape(1, 50, 1)  # (1개 샘플, 50일, 1개 특성)

# 예측 (정규화된 값)
pred_scaled = modelRNN.predict(input_data)

# 정규화 복원
predicted_price = scaler.inverse_transform(pred_scaled)

print("예측한 다음 날 시가:", predicted_price[0][0])
```

<br><br><br>


```py
# 학습 데이터 전체에 대해 예측
train_predict = modelRNN.predict(X_train)

# 정규화 복원
train_predict_original = scaler.inverse_transform(train_predict.reshape(-1, 1))
y_train_original = scaler.inverse_transform(y_train.reshape(-1, 1))

# 시각화
plt.figure(figsize=(12, 6))
plt.plot(y_train_original, label="실제 시가")
plt.plot(train_predict_original, label="예측 시가")
plt.legend()
plt.title("실제 시가 vs 예측 시가")
plt.xlabel("시간")
plt.ylabel("시가")
plt.show()
```
![image](https://github.com/user-attachments/assets/ae3ef32d-67ee-4d70-b320-843d83bb2168)

<br><br><br>

```py
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_train_original, train_predict_original)
mse = mean_squared_error(y_train_original, train_predict_original)

print("MAE:", mae)
print("MSE:", mse)
```
```
MAE: 5.447966969521508 
MSE: 70.77429343216302
```

#### 모델 성능 평가 지표(값이 낮을수록 좋다.)
- MAE (평균 절대 오차) : 실제 정답 값과 예측 값의 차이를 절댓값으로 변환한 뒤 합산하여 평균을 구한다.

- MSE (평균 제곱 오차) : 실제 정답 값과 예측 값의 차이를 제곱한 뒤 평균을 구한다.

- RMSE(평균 제곱근 오차) : MSE에 루트를 씌운 값

- MAPE(평균 절대 비율 오차) : MAE를 퍼센트로 표현


<br><br><br><br>


---


```py
df_train = pd.read_csv(...)

 # 'Close' 컬럼에서 쉼표 제거 → 문자열을 숫자(float)로 변환
df_closing = df_train['Close'].apply(lambda x : x.replace(',', '')).astype('float')

# 0~1 사이로 정규화할 스케일러 생성
scaler = MinMaxScaler(feature_range=(0,1))

# 종가 데이터를 (n, 1) 형태로 reshape → 정규화 진행
df_closing = scaler.fit_transform(df_closing.values.reshape(-1, 1))

# 시퀀스 만들기
def create_dataset(dataset, time_step=1):
    x_data, y_data = [], []  # 입력 시퀀스와 정답 리스트 초기화
    for i in range(len(dataset)-time_step-1):  # 전체 길이에서 타임스텝만큼 슬라이딩
        x_data.append(dataset[i:(i+time_step), 0])  # 과거 time_step개 데이터를 입력으로 저장
        y_data.append(dataset[i + time_step, 0])    # 다음 값을 정답으로 저장
    return np.array(x_data), np.array(y_data)  # 넘파이 배열로 반환

# 65% 학습, 35% 테스트로 나눔
training_size = int(len(df_closing)*0.65)
train_data = df_closing[:training_size]  # 학습 데이터
test_data = df_closing[training_size:]   # 테스트 데이터

time_step = 100  # 과거 100일 사용(시퀀스 길이를 100으로 지정)  
X_train, y_train = create_dataset(train_data, time_step)  # 학습 시퀀스 생성
X_test, ytest = create_dataset(test_data, time_step)      # 테스트 시퀀스 생성

X_train = X_train.reshape(-1, 100, 1)  # (샘플 수, 시퀀스 길이, 1개의 특성)
X_test = X_test.reshape(-1, 100, 1)
```

<br><br><br>

### 하이퍼파라미터 튜닝 모델 정의
```py
def build_model(hp):  # hp: 하이퍼파라미터를 조정할 수 있게 해주는 객체
    model = Sequential()  # 순차 모델 생성

   # 첫 번째 LSTM 레이어: 유닛 수를 10~100 중 선택, 시퀀스 출력 유지
    model.add(LSTM(units = hp.Choice('layer1_units', [10~100]), return_sequences=True, input_shape=(100,1)))

    # 두 번째~여러 번째 LSTM 레이어: 층 수를 2~15 사이로 지정
    for i in range(hp.Int('num_layers', 2, 15)):
        # 각 층의 유닛 수를 10~150 사이 10단위로 조절        
        model.add(LSTM(units = hp.Int('units'+str(i), 10, 150, step=10), return_sequences=True))

    # 마지막 LSTM 층: 시퀀스 출력하지 않음 (마지막 예측만 출력)
    model.add(LSTM(units = hp.Choice('last_lstm_units', [50, 100, 150])))

    # Dropout: 과적합 방지용, 비율은 0.3~0.7 중 선택
    model.add(Dropout(rate = hp.Choice('rate', [0.3~0.7])))

    # 출력층: 다음 날 가격 1개 예측
    model.add(Dense(1))

    # 손실함수와 옵티마이저 설정
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
```
- 튜닝 가능한 하이퍼파라미터: LSTM 유닛 수, 층 수, 드롭아웃 비율 등

- 튜닝 도구: Keras Tuner의 RandomSearch

<br><br><br>

### 튜닝 및 학습 실행
```py
# RandomSearch로 여러 하이퍼파라미터 조합을 테스트
tuner = RandomSearch(build_model, ...)

# 훈련 데이터를 기반으로 최적 조합 찾기
tuner.search(X_train, y_train, ...)

# 가장 좋은 모델 1개 선택
best_model = tuner.get_best_models(1)[0]

# 최적 모델로 학습 수행
best_model.fit(...)
```

<br><br><br>

### 예측 및 시각화
```py
train_predict = best_model.predict(X_train)  # 학습 데이터에 대한 예측
test_predict = best_model.predict(X_test)    # 테스트 데이터에 대한 예측

# 정규화 복원 (0~1 → 원래 값으로)
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# 미래 10일 예측
future_predictions = test_data.copy()   # 테스트 데이터를 복사 (미래 예측에 활용)

for i in range(100):  # 100일치 예측 반복
    # 최근 100일 데이터를 입력으로 사용
    next_input = future_predictions[341+i:].reshape(1,100,1)

    new_prediction = model.predict(next_input) # 다음 1일 예측

    # 예측 결과를 뒤에 추가하여 다음 입력으로 사용
    future_predictions = np.append(future_predictions, new_prediction)
```
- 출력: 향후 10일간의 주가 예측

- 시각화: 전체 예측 곡선 그리기

---

1번 코드(시계열 분류 모델) 기반으로 매수/매도 이진 분류 코드 작성

<br><br><br>

### 이진 분류 LSTM 모델
```py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.model_selection import train_test_split

# 데이터 불러오기 (여기에 본인 csv 경로 입력)
df = pd.read_csv("your_stock_data.csv")  # 예: 삼성전자 일별 주가 데이터

# 종가(Close) 열 가져오기 (쉼표 제거 + float 변환)
df['Close'] = df['Close'].apply(lambda x: str(x).replace(',', '')).astype(float)
close_prices = df['Close'].values.reshape(-1, 1)

# 0~1 사이 값으로 정규화 (LSTM에 필요)
scaler = MinMaxScaler(feature_range=(0, 1))
close_scaled = scaler.fit_transform(close_prices)
```

<br><br><br>

### 데이터셋 구성 함수 (입력 시퀀스와 레이블 만들기) 
```py
def create_binary_dataset(dataset, time_step=30):
    x_data, y_data = [], []
    for i in range(len(dataset) - time_step - 1):
        x_seq = dataset[i:i + time_step, 0]            # 과거 30일
        next_day = dataset[i + time_step, 0]           # 다음 날 종가
        today = dataset[i + time_step - 1, 0]          # 오늘 종가
        label = 1 if next_day > today else 0           # 상승하면 1, 하락/보합이면 0
        x_data.append(x_seq)
        y_data.append(label)
    return np.array(x_data), np.array(y_data)
```
- x_data: 과거 30일간 종가

- y_data: 다음날 종가가 오늘보다 오르면 1(매수), 아니면 0(매도)

<br><br><br>

### 모델 입력 데이터 만들기
```py
time_step = 30
X, y = create_binary_dataset(close_scaled, time_step)

# LSTM에 맞게 3D 형태로 reshape
X = X.reshape(X.shape[0], X.shape[1], 1)  # (샘플수, 타임스텝, 1)

# 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
```

<br><br><br>

### LSTM 분류 모델 만들기
```py
model = Sequential()
model.add(LSTM(64, return_sequences=False, input_shape=(time_step, 1)))
model.add(Dropout(0.2))  # 과적합 방지
model.add(Dense(1, activation='sigmoid'))  # 이진 분류 (확률 출력)

# 컴파일: 이진 분류니까 binary_crossentropy 사용
model.compile(loss=BinaryCrossentropy(), optimizer=Adam(0.001), metrics=['accuracy'])
```

<br><br><br>

### 모델 학습
```py
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), verbose=1)
```
- epochs: 학습 횟수 (30번 반복)

- batch_size: 32개씩 묶어서 학습

- validation_data: 테스트 정확도도 같이 확인

<br><br><br>

### 모델 평가 및 예측 결과 확인
```py
# 예측 확률 출력 (0~1 사이)
y_pred_prob = model.predict(X_test)

# 0.5 이상이면 매수(1), 아니면 매도(0)로 처리
y_pred = (y_pred_prob > 0.5).astype(int)

# 정확도 출력
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

```

<br><br><br>

### 예측 결과 시각화
```py
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.plot(y_test[:100], label="Real (매도:0 / 매수:1)")
plt.plot(y_pred[:100], label="Predicted", linestyle='--')
plt.title("실제 vs 예측 (100개 샘플)")
plt.legend()
plt.show()

```

<br><br>

|항목|설명|
|:---:|:---:|
|문제 유형|시계열 이진 분류 (다음날 종가 상승 여부 예측)|
|입력 X|과거 30일 종가 (정규화)|
|출력 y|다음날 종가가 오르면 1(매수), 아니면 0(매도)|
|모델 구조|LSTM(64) → Dropout(0.2) → Dense(1, sigmoid)|
|손실 함수|Binary Crossentropy|
|평가 지표|Accuracy (정확도)|
|예측 결과|확률(0~1)을 0.5 기준으로 분류|


<br><br>


---

Buy, Sell, Hold로 3분류 모델

- 1 (Buy) : 종가가 다음날 0.5% 이상 상승

- 0 (Hold) : 변동 거의 없음 (-0.5% ~ +0.5%)

- 2 (Sell) : 종가가 0.5% 이상 하락

<br><br><br>

### 데이터셋 구성 함수 (3분류로 변경)
```py
def create_triple_class_dataset(dataset, time_step=30, threshold=0.005):
    x_data, y_data = [], []
    for i in range(len(dataset) - time_step - 1):
        x_seq = dataset[i:i + time_step, 0]
        today = dataset[i + time_step - 1, 0]
        next_day = dataset[i + time_step, 0]
        rate = (next_day - today) / today

        if rate > threshold:
            label = 1  # Buy
        elif rate < -threshold:
            label = 2  # Sell
        else:
            label = 0  # Hold

        x_data.append(x_seq)
        y_data.append(label)

    return np.array(x_data), np.array(y_data)

```

<br><br><br>

### 데이터 준비
```py
X, y = create_triple_class_dataset(close_scaled, time_step=30)

# 3D 형태로 변환 (LSTM 입력용)
X = X.reshape(X.shape[0], X.shape[1], 1)

# train/test 분리
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

```

<br><br><br>

### LSTM 3분류 모델 만들기
```py
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# one-hot 인코딩 (3개 클래스)
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))  # 3개 클래스 분류

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train_cat, epochs=30, batch_size=32, validation_data=(X_test, y_test_cat))

```

<br><br><br>

### 예측 및 평가 (정밀도, 재현율, F1 score)
```py
from sklearn.metrics import classification_report

# 예측 확률
y_pred_prob = model.predict(X_test)

# 가장 확률 높은 클래스로 예측
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test_cat, axis=1)

# 평가 리포트 출력
print(classification_report(y_true, y_pred, target_names=["Hold", "Buy", "Sell"]))
```
- precision (정밀도): "매수라고 예측한 것 중 진짜 매수일 확률"

- recall (재현율): "실제 매수 중에 얼마나 잘 맞췄나"

- f1-score: precision과 recall의 조화 평균

- support: 각 클래스 개수
