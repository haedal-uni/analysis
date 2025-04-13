# 시계열 머신러닝·딥러닝 모델링 정리

## 1. 시계열 데이터란?

- 시간 순서대로 정렬된 데이터 (예: 주식 가격, 날씨, 환율 등)
  
- 예: 2023-01-01 → 2023-01-02 → 2023-01-03 ...
  
- 일반적인 데이터 분석에서는 독립변수(independent variable)를 이용해 종속변수(dependent variable)를 예측하지만
  
  시계열 데이터에서는 시간을 독립변수처럼 활용

<br><br><br>

### 시계열 데이터 특징
- **Abrupt Change** : 갑작스러운 변화가 있는지 확인
- **Outliers** : 이상치가 존재하는지 확인
- **Trend** : 데이터가 점점 증가 또는 감소하는 경향이 있는지 확인
- **Seasonality** : 일정 시간 주기마다 반복되는 패턴이 있는지 확인
- **Constant Variance** : 분산이 일정한지 확인
- **Long-run Cycle** : 장기간 주기의 순환이 있는지 확인

<br><br><br>

---

## 2. 모델링 기본 흐름

1. 데이터 준비: 시계열 데이터 로드
2. 정상성 확인: 패턴이 일정한지 확인
3. 전처리: 로그 변환, 차분 등
4. 데이터 분할: 훈련/검증/테스트
5. 모델 학습: ARIMA, MLP, RNN, LSTM 등
6. 평가: MAE, MSE, RMSE 등
7. 미래 예측: 향후 데이터 예측

<br><br><br>

---

## 3. 기본 개념 정리

### 정상성 (Stationarity)
- 평균과 분산이 시간에 따라 일정한 데이터
- 전통적인 시계열 모델에 꼭 필요함

<br><br><br>

### 정상성 검정 방법
- **ADF Test (Augmented Dickey-Fuller Test)**
  - 귀무가설: 비정상 시계열 (Unit root 존재)
  - p-value < 0.05 → 정상성 있음 (귀무가설 기각)
```py
from statsmodels.tsa.stattools import adfuller
result = adfuller(data)
print(f'p-value: {result[1]}')
```

<br><br><br>

### 로그 수익률 (Log Return)
- 정의: 로그 수익률은 주가 등의 변화율을 로그로 변환한 것
- 계산 공식: `log_return = log(today_price / yesterday_price)`
- 이유: 변동성이 큰 데이터를 안정화시켜 모델 학습에 유리하게 만듦

<br><br><br>

### 차분 (Differencing)
- 정의: 시간 차이 데이터를 통해 추세 제거 → 정상성 확보
- 계산 공식: `diff = data[t] - data[t-1]`

<br><br><br>

---

## 4. 주요 모델

| 모델 | 분류 | 특징 |
|------|------|------|
| ARIMA | 통계 | 자기회귀(p), 차분(d), 이동평균(q) |
| SARIMA | 통계 | ARIMA + 계절성 요소 포함 |
| SVR | 머신러닝 | 과거 데이터를 기반으로 회귀 |
| Linear Regression | 머신러닝 | 직선 기반 예측 |
| MLP | 딥러닝 | 기본 신경망 구조 |
| RNN | 딥러닝 | 시계열 순서 기억 가능 |
| LSTM | 딥러닝 | 장기 의존성 기억 가능 |
| GRU | 딥러닝 | LSTM보다 단순하지만 유사한 성능 |
| Transformer | 딥러닝 | 순서와 상관관계 학습 가능 (attention 메커니즘 기반) |

<br><br><br>

---

## 5. 모델 성능 평가 지표

| 지표 | 계산법 | 설명 |
|------|--------|------|
| MAE | Mean Absolute Error | 평균 절댓값 오차 |
| MSE | Mean Squared Error | 평균 제곱 오차 (큰 오차에 민감함) |
| RMSE | Root Mean Squared Error | 제곱 오차의 루트 (단위 복원) |
| MAPE | Mean Absolute Percentage Error | 비율 기반 오차, 실제 값이 0에 가까우면 불안정 |
| R^2 | 결정계수 | 1에 가까울수록 좋은 예측 |

<br><br><br>

---

## 6. 자주 쓰는 용어 정리

| 용어 | 설명 |
|------|------|
| Batch | 여러 샘플을 한 번에 학습시키는 단위 |
| Epoch | 전체 데이터를 몇 번 반복해서 학습하는지 횟수 |
| Loss | 모델의 예측이 얼마나 틀렸는지를 수치로 표현한 것 |
| Optimizer | 모델을 더 잘 예측하도록 가중치를 조정하는 함수 (예: Adam, SGD) |
| fit_transform | 데이터를 전처리하고 학습(피팅)까지 한 번에 수행하는 메서드 |
| val_loss | 검증용 데이터(validation set)에서의 손실값. 과적합 여부 판단에 사용 |
| Overfitting | 훈련 데이터에만 지나치게 잘 맞추고 일반화가 안 되는 현상 |
| Underfitting | 훈련 데이터조차 잘 예측하지 못하는 상태 |

<br><br><br>

---

## 7. ACF / PACF 개념

| 지표 | 설명 | 의미 |
|------|------|------|
| ACF | Autocorrelation Function | 전체 시점 간 자기 상관 관계 |
| PACF | Partial ACF | 직접적인 상관관계만 고려함 |

<br><br><br>

### ACF / PACF 패턴 해석

| ACF 패턴 | PACF 패턴 | 추천 모델 |
|-----------|-------------|-------------|
| 천천히 감소 | 뚝 끊김 | AR 모델 |
| 뚝 끊김 | 천천히 감소 | MA 모델 |
| 둘 다 천천히 감소 | - | 비정상 데이터 가능성 |

<br><br><br>

---

## 8. 예측 결과 복원 예시 (로그 수익률 기반)

> 로그 수익률로 예측된 값을 원래의 가격 단위로 복원할 때 사용함

```py
predicted_price = [start_price]  # 시작 가격 (예: 마지막 실제 종가)
for log_return in predicted_log_returns:
    next_price = predicted_price[-1] * np.exp(log_return)
    predicted_price.append(next_price)
```

- `np.exp(log_return)`은 로그 수익률을 다시 가격 변화율로 바꾸는 함수임
- 누적해서 다음 가격을 계산함

<br><br><br>

---

## 9. PyTorch란?

- **PyTorch**는 Python 기반의 딥러닝 프레임워크로, Facebook에서 개발함
- 텐서 연산, 자동 미분, GPU 연산을 지원하며 딥러닝 모델 구현에 많이 사용됨

<br><br><br>

---

## 10. PyTorch 모델 구조 예시

### MLP (다층 퍼셉트론)
```py
self.model = nn.Sequential(
    nn.Linear(input_size, hidden_size),  # 입력층 → 은닉층
    nn.ReLU(),                           # 활성화 함수
    nn.Linear(hidden_size, output_size)  # 은닉층 → 출력층
)
```
- 여러 개의 fully connected layer로 구성된 기본 신경망 구조

<br><br><br>

### RNN (순환 신경망)
```py
self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
self.fc = nn.Linear(hidden_size, output_size)
```
- 이전 시점의 출력(hidden state)을 현재 시점 입력과 함께 사용

<br><br><br>

### LSTM (장기기억 순환 신경망)
```py
self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
self.fc = nn.Linear(hidden_size, output_size)
```
- RNN 구조를 개선하여 장기 의존성(long-term dependency)을 잘 처리함
- cell state와 gate 구조가 핵심

<br><br><br>

---

## 11. 학습 코드 예시

> 모델 학습 과정을 반복(epoch)하며 데이터셋을 이용해 손실을 줄이는 구조

```py
for epoch in range(epochs):                          # epoch 수 만큼 반복
    model.train()                                    # 모델을 학습 모드로 전환
    for X_batch, y_batch in train_loader:            # 배치 단위로 데이터 불러오기
        output = model(X_batch)                      # 예측 수행
        loss = criterion(output, y_batch)            # 손실 계산
        loss.backward()                              # 역전파로 gradient 계산
        optimizer.step()                             # optimizer로 가중치 갱신
        optimizer.zero_grad()                        # gradient 초기화
```

<br><br><br>

---

## 12. 추천 학습 순서

1. 시계열 기본 개념 이해 (정상성, 로그 수익률, 차분)
2. 통계 기반 모델 실습 (ARIMA, ADF 테스트)
3. 머신러닝 회귀 모델 실습 (SVR, 선형 회귀 등)
4. 딥러닝 모델 순차 학습 (MLP → RNN → LSTM → Transformer)
5. PyTorch 기초 학습 (Tensor, 모델 정의, 학습 코드)
6. 입력 구조와 출력 복원 방법 숙지
7. 모델 성능 비교 및 시각화 연습

