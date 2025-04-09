# LSTM 시계열 예측 모델 정리
## 시계열 데이터란?

**시간의 흐름에 따라 변하는 데이터**를 말한다.

예를 들어,

- 매일의 주가  
- 매시간의 기온  
- 매초 측정되는 센서 값

이런 데이터들은 모두 시간이 지나면서 값이 바뀌므로 **시계열(Time Series)** 데이터다.

예시 : 100 → 101 → 103 → 105 → 108 → ...


시간이 지남에 따라 하나의 숫자씩 차례대로 이어지는 구조

---

## LSTM이란?

**LSTM(Long Short-Term Memory)** 은 딥러닝 모델 중 하나로, 순차적인 데이터를 아주 잘 처리한다.

- 예전 데이터를 잘 기억하면서 미래 값을 예측할 수 있도록 만들어졌다.

- LSTM은 시간 흐름을 기억해서 미래를 예측해주는 모델

ex) 예를 들어 1월1일부터 1월20일까지 가격을 보면, 1월21일 가격을 어느 정도 맞출 수 있지 않을까?

그걸 자동으로 배우는 게 LSTM


### LSTM은 RNN의 발전형!
- 일반적인 RNN은 오래된 정보는 금방 잊어버리는 단점이 있음.
  
- 하지만 **LSTM은 과거 데이터를 오래 기억**해서 예측에 활용할 수 있다.
  
-  **장기 의존성 문제**도 해결


> 모델을 "이전 며칠간의 주가 흐름을 잘 기억하고, 내일의 주가를 맞춰보자!" 라고 말할 수 있다. 

---

## 시퀀스(Sequence)란?

**순서가 있는 데이터 묶음**을 의미한다.

ex)  

- `[100, 101, 103, 105, 106]` → 주가 시퀀스
  
- `['안', '녕', '하', '세', '요']` → 글자 시퀀스  

➡ 시계열에서는 이러한 시퀀스를 가지고 예측 모델을 학습시킨다.

---

## 코드 흐름 요약

시계열 예측 모델은 다음과 같은 순서로 만들어진다.      

1.  CSV 파일 읽기 (날짜를 인덱스로 지정)
2. 결측치 처리 및 문자열을 숫자로 변환 (예: 거래량, 변동%)
3. 종가 그래프 시각화
4. 정규화 (MinMaxScaler 사용)
5. Train/Test 데이터 분할
6. 시퀀스 데이터로 학습용 입력 구성
7. 모델 정의 (Conv1D → LSTM → Dense)
8. 모델 훈련 (EarlyStopping & Checkpoint 사용)
9. 예측 결과 시각화 및 해석

---

## 시퀀스 구성 방식 (Windowing)

예를 들어 `WINDOW_SIZE = 20` 이라면?

> 20일치 종가를 입력으로 넣고 그 다음 날의 종가를 예측하는 구조다.   

ex)              
X: [100, 101, ..., 119]          
Y: 120          
            
➡ 이런 식으로 데이터 윈도우를 조금씩 이동하면서 학습 데이터를 만든다.   

---

## 핵심 개념 요약

| 용어 | 설명 |
|------|------|
| **MinMaxScaler** | 데이터를 0~1 범위로 정규화 |
| **train_test_split** | 학습/테스트용 데이터 분할 |
| **windowed_dataset()** | 시계열을 (입력 n개 → 출력 1개) 형태로 변환 |
| **Conv1D** | 시계열 흐름의 특징 추출 (1차원 필터) |
| **LSTM** | 시계열 패턴 학습 |
| **Dense** | 출력층 |
| **ModelCheckpoint** | 모델 저장 |
| **EarlyStopping** | 과적합 방지 (조기 종료) |

---

## Hyperparameter 정리

| 하이퍼파라미터 | 설명 |
|----------------|------|
| `WINDOW_SIZE`   | 며칠치 데이터를 한 묶음으로 볼지 (예: 20일) |
| `BATCH_SIZE`    | 한 번에 학습할 묶음 개수 (예: 32) |
| `LSTM 노드 수`  | LSTM의 뉴런 수 (예: 16개) |
| `learning rate` | 학습 속도 (예: 0.0005) |
| `epochs`        | 학습 반복 횟수 |
| `optimizer`     | 최적화 방법 (예: Adam) |

WINDOW_SIZE = 20, BATCH_SIZE = 32 → 20개씩 묶은 샘플이 총 32개로 배치 단위로 모델에 전달된다

---

## 왜 Conv1D?

- 주가 데이터는 시간에 따라 변하는 **1차원 시계열**이다.
  
- 그래서 **1차원 CNN (Conv1D)** 을 사용하여 흐름의 패턴을 뽑아낸다.

Conv1D는 아래와 같은 데이터를 처리하기에 좋다: [100, 102, 101, 103, …]

➡ 주가, 센서, 음성 데이터 등 모두 Conv1D로 처리 가능

---

## Huber Loss란?

손실 함수(Loss Function)는 모델이 얼마나 틀렸는지를 측정한다.   

### 종류 비교

| 종류 | 설명 |
|------|------|
| **MSE** (평균제곱오차) | 너무 큰 오차(이상치)에 민감 |
| **MAE** (평균절대오차) | 이상치에 덜 민감하지만, 기울기 불연속 |
| **Huber** | MSE + MAE 장점을 합친 손실 함수 → 더 안정적인 학습 가능!

---

## 코드

### 1. 데이터 로딩 및 전처리

```py
import pandas as pd

df = pd.read_csv('미국 철강 코일 선물 과거 데이터.csv', parse_dates=['날짜'], index_col='날짜', thousands=",")
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


#### 문자열을 숫자로 변환
```py
df['거래량'] = df['거래량'].apply(lambda x: float(x.replace('K', '')) * 1000 if isinstance(x, str) else x)
df['변동 %'] = df['변동 %'].apply(lambda x: float(x.replace('%', '')) / 100 if isinstance(x, str) else x)
```

### 2. 데이터 시각화
```py
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(16, 9))
sns.lineplot(x=df.index, y=df['종가'])
plt.xlabel('Time')
plt.ylabel('Price')
```

시계열 구간별 시각화
```py
time_steps = [['2015', '2018'], ['2018', '2021'], ['2021', '2024'], ['2024', '2025']]
fig, axes = plt.subplots(2, 2, figsize=(16, 9))

for i in range(4):
    ax = axes[i//2, i%2]
    data = df.loc[(df.index > time_steps[i][0]) & (df.index < time_steps[i][1])]
    sns.lineplot(y=data['종가'], x=data.index, ax=ax)
    ax.set_title(f'{time_steps[i][0]}~{time_steps[i][1]}')

plt.tight_layout()
plt.show()
```



### 3. 데이터 정규화 (MinMaxScaler)  
```py
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scale_cols = ['종가', '시가', '고가', '저가', '거래량']
scaled = scaler.fit_transform(df[scale_cols])
```


### 4. Train/Test 분리
```py
from sklearn.model_selection import train_test_split

df_scaled = pd.DataFrame(scaled, columns=scale_cols)
x_train, x_test, y_train, y_test = train_test_split(
    df_scaled.drop('종가', axis=1),
    df_scaled['종가'],
    test_size=0.2,
    shuffle=False
)
```
y는 (n,) 형태로 되어 있으므로 → LSTM에 넣기 위해 (n, 1) 형태로 reshape 필요하다.


### 5. 시퀀스 데이터셋 구성 
데이터를 일정한 간격(윈도우)으로 잘라서 입력(X), 출력(Y) 만들기

```yaml
X: [1~20일 주가], y: 21일 주가  
X: [2~21일 주가], y: 22일 주가
```
이런 방식으로 슬라이딩 윈도우(Window)를 만들어 모델 학습에 활용  

-----

<br><br><br><br>

# 철강 코일 선물 가격 예측 (LSTM 활용)
## 분석 목적

**주가(철강 코일 선물 가격)** 가 앞으로 어떻게 변할지 예측하기 위해 LSTM 기반 시계열 예측 모델.


## 왜 LSTM을 쓰는가?

- 주가는 **시간에 따라 변하는 시계열 데이터**다.
  
- LSTM(Long Short-Term Memory)은 **과거 정보를 기억하며** 미래 값을 예측할 수 있는 딥러닝 모델이다.
  
- 우리가 하고자 하는 것은 **“과거 20일의 가격을 보고, 다음 날 가격을 예측”** 하는 구조다.

---

## 전체 흐름 요약 
| 단계 | 설명 |
| --- | --- |
| 1. 데이터 정리 | 날짜별 주가를 숫자로 바꿔서 보기 쉽게 정리 |
| 2. 정규화 | 값들의 크기를 맞춰 딥러닝 학습에 적합하게 만듦 |
| 3. 시퀀스 만들기 | 과거 20일 → 다음 날 예측 쌍을 만듦 |
| 4. 모델 구성 | Conv1D+LSTM으로 주가 흐름을 학습 |
| 5. 학습 | 데이터를 반복 학습해 미래 예측 능력을 높임 |

---

## 코드 주요 단계별 설명

### (1) 데이터 불러오기
```py
df = pd.read_csv(...)

# 거래량과 변동 수치로 변경
df['거래량'] = df['거래량'].apply(lambda x: float(x.replace('K', '')) * 1000 if isinstance(x, str) and 'K' in x else float(x))
df['변동 %'] = df['변동 %'].apply(lambda x: float(x.replace('%', '')) / 100 if isinstance(x, str) else x)
```


### (2) 주가 시각화
```py
sns.lineplot(...)
```
전체 주가 흐름과 연도별 변화를 선 그래프로 확인

데이터가 어떻게 움직였는지 시각적으로 먼저 확인

### (3) 딥러닝을 위한 데이터 정규화
```py
scaler = MinMaxScaler()
scaled = scaler.fit_transform(...)
```
딥러닝은 숫자의 크기에 민감하므로 0~1 사이로 변환

ex: 605 → 0.15, 932 → 0.35 (상대적인 값으로 변환)


### (4) 학습용 / 테스트용 데이터 분리
```py
train_test_split(...)
```

데이터의 80%는 학습용, 20%는 테스트용으로 사용

과거 데이터로 학습하고, 최근 데이터로 성능 검증

### (5) LSTM이 알아들을 수 있는 형태로 바꾸기
```py
windowed_dataset(...)
```
“과거 20일 데이터 → 다음 날 가격” 쌍을 생성

[1월1일~1월20일] → 1월21일 가격  
[1월2일~1월21일] → 1월22일 가격  

수천 개의 학습 데이터를 자동 생성


### (6) LSTM 모델 구성
```py
model = Sequential([
    Conv1D(...),  # 국소적인 패턴 추출
    LSTM(...),    # 시간 흐름 기억
    Dense(1)      # 가격 1개 예측
])
```
- Conv1D: 짧은 기간 내 패턴 감지

- LSTM: 시간 흐름 기억

- Dense: 결과값(가격) 출력


### (7) 모델 학습
```py
model.fit(...)
```
주어진 데이터를 가지고 반복 학습 진행

EarlyStopping: 성능 향상 없으면 자동 중단

ModelCheckpoint: 가장 좋은 성능일 때 모델 저장
