
### Prophet 시계열 예측 모델 설명 및 예제 코드 해설

#### Prophet이란?

**Prophet**은 Facebook(현 Meta)에서 개발한 **시계열 데이터 예측**을 위한 도구다. 

- 시간 흐름에 따라 변하는 데이터를 예측할 때 사용한다.

- 특히 **계절성**, **추세 변화**, **휴일 효과** 등을 잘 처리한다.
  
- 사용이 매우 간단하며, 기본 설정만으로도 좋은 예측 성능을 낼 수 있다.

<br><br><br>

---

#### Prophet 모델의 기본 개념

Prophet 모델은 시계열 데이터를 다음과 같은 수학적 구조로 예측한다: 

```
y(t) = g(t) + s(t) + h(t) + ε_t
```

| 구성 요소 | 의미 |
|-----------|------|
| `g(t)`    | 트렌드 (전반적인 상승/하락 흐름) |
| `s(t)`    | 계절성 (주기적인 변화) |
| `h(t)`    | 휴일 등 특별 이벤트 |
| `ε_t`     | 오차 (노이즈) |

- **additive model** (가법 모델): 요소들을 단순히 더함 → 일반적인 시계열 데이터
  
- **multiplicative model** (승법 모델): 요소들을 곱함 → 변화량이 점점 커지는 데이터에 적합

<br><br><br>

---

#### 데이터 준비

```python
df = pd.read_csv('미국 철강 코일 선물 과거 데이터.csv', thousands=",")
df = df.sort_index()
df['날짜'] = pd.to_datetime(df['날짜'])
```

- 날짜와 종가를 포함한 데이터다.
  
- Prophet은 반드시 다음 형식으로 컬럼명을 지정해야 한다:

| 컬럼명 | 설명 |
|--------|------|
| `ds`   | 날짜 (datetime 형식) |
| `y`    | 예측할 값 (여기서는 종가) |

```python
df_train = df[df['날짜'] < '2025-01-01'][["날짜", "종가"]]
df_train = df_train.rename(columns={"날짜": "ds", "종가": "y"})
```

<br><br><br>

---

#### Prophet 모델 설정 및 학습

```python
from prophet import Prophet

m = Prophet(
    changepoint_prior_scale=0.05,  # 트렌드 변화 민감도
    weekly_seasonality=10,         # 주간 패턴 반영
    yearly_seasonality=10,         # 연간 패턴 반영
    daily_seasonality=10,          # 일일 패턴 반영
    seasonality_mode='additive'    # 계절성 + 트렌드의 합 모델
)
m.fit(df_train)
```
<br><br><br>

##### 주요 파라미터 설명:

| 파라미터 | 설명 |
|----------|------|
| `changepoint_prior_scale` | 트렌드 변화 감지 민감도 (높을수록 더 민감) |
| `weekly_seasonality` | 주간 주기성 설정 |
| `yearly_seasonality` | 연간 주기성 설정 |
| `daily_seasonality` | 일일 주기성 설정 |
| `seasonality_mode` | 계절성과 트렌드 결합 방식 (additive / multiplicative) |

<br><br><br>

---

#### 미래 예측 수행

```python
future = m.make_future_dataframe(periods=365)  # 365일 예측
forecast = m.predict(future)
```

- `make_future_dataframe`은 미래 날짜를 자동으로 생성해준다.
  
- `predict`는 예측값(yhat), 예측 하한(yhat_lower), 예측 상한(yhat_upper)을 포함한 결과를 준다. 

<br><br><br>

---  

#### 예측값과 실제값 시각화

```python
# 실제값 가져오기
df_test = df[df['날짜'] >= '2025-01-01'][["날짜", "종가"]]
df_test = df_test.rename(columns={"날짜": "ds", "종가": "Actual"})

# 예측 결과와 병합
forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
merged = pd.merge(forecast, df_test, on='ds', how='left')

# 시각화
plt.figure(figsize=(14, 7))
plt.plot(merged['ds'], merged['yhat'], label='Predicted')         # 예측값
plt.plot(merged['ds'], merged['Actual'], label='Actual', color='r')# 실제값
plt.scatter(df_train['ds'], df_train['y'], color='black', s=10, label='Historical Data')  # 학습 데이터
plt.fill_between(merged['ds'], merged['yhat_lower'], merged['yhat_upper'], alpha=0.2)    # 예측 구간
plt.legend()
plt.show()
```

<br><br><br>

---

#### 그래프 해설

| 요소 | 의미 |
|------|------|
| **파란 선** (`Predicted`) | Prophet이 예측한 미래 종가 (`yhat`) |
| **빨간 선** (`Actual`) | 실제로 관측된 종가 (2025년 이후) |
| **검은 점** (`Historical`) | Prophet이 학습한 과거 데이터 |
| **음영 구간** | 예측의 신뢰 구간 (`yhat_lower ~ yhat_upper`) |

<br><br><br>

---

#### 요약 정리

| 단계 | 설명 |
|------|------|
| 1. 데이터 준비 | `ds`, `y` 형식으로 날짜와 예측값 구성 |
| 2. 모델 생성 | `Prophet()`으로 설정값과 함께 객체 생성 |
| 3. 모델 학습 | `m.fit(df_train)` |
| 4. 미래 생성 | `make_future_dataframe(periods=...)` |
| 5. 예측 수행 | `predict(future)` |
| 6. 결과 시각화 | `matplotlib`으로 시각화하여 성능 확인 |

<br><br><br>

---

#### 참고 사항

- Prophet은 시계열 초보자에게도 매우 친숙한 도구다.
  
- 계절성, 추세, 불확실성까지 자동으로 처리해주므로 빠르게 프로토타입을 만들 수 있다.
  
- 그러나 **외생 변수**(exogenous variables)는 기본 Prophet으로는 다루기 어렵다.
  
     이런 경우에는 확장된 Prophet + Regressor 모델을 사용해야 한다.
