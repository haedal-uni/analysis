
# 로보어드바이저 가이드 

## 전략 비교표

| 전략명         | 설명                              | 장점                        | 단점                      |
|----------------|-----------------------------------|-----------------------------|---------------------------|
| **전일 변화율** | 오늘과 어제의 가격 차이를 본다       | 빠르게 반응 (단기 감지)       | **노이즈**에 민감 (헛신호 많음) |
| **이동 평균**   | 일정 기간 평균 가격을 본다 (예: 5일) | 추세에 따라 매매 가능 (안정적) | 신호가 **늦게 나올 수 있음**     |

### 용어 설명  

- **노이즈**: 큰 의미 없는 가격의 흔들림
- **추세 기반**: 전체 흐름을 보고 매매하는 방식 (단기보단 중장기 느낌)

---

## 관점별 전략 차이 (비교 정리)

| 관점   | 전일 변화율                     | 이동 평균                           |
|--------|-------------------------------|------------------------------------|
| **전략 목적** | 단기 가격 변화 감지                | 큰 흐름(추세) 판단                  |
| **장점**    | 간단하고 빠르다                   | 안정적으로 매매 판단 가능             |
| **약점**    | 헛신호 가능성 있음 (민감함)         | 신호가 느릴 수 있음 (타이밍 어려움)     |
| **사용 예시** | 급등/급락 포착 (스캘핑 등)         | 골든크로스, 추세 매매 (중기 전략)       |

---

## 골든 크로스란?

> **단기 이동 평균선이 장기 이동 평균선을 위로 돌파할 때** → 상승 신호!

- 예: 5일 이동평균이 20일 이동평균을 돌파하면 "**매수**"
- 반대: **데드 크로스** = 하락 신호

```text
   가격
     ▲
     |     o o o o   ← 5일 평균 (단기)
     |   o        o
     |  o          o
     | o             o ← 20일 평균 (장기)
     +----------------------▶ 날짜
```

---

## 로보어드바이저란?

> **사람 대신 자동으로 투자 판단해주는 시스템**

### 📦 종류

| 유형           | 설명                                           |
|----------------|------------------------------------------------|
| **규칙 기반**     | 조건(룰) 정해서 자동 투자 (예: PER<10, ROE>15)   |
| **통계 기반**     | 과거 통계 기반으로 종목 필터링 또는 매매           |
| **AI 기반**       | 머신러닝·딥러닝으로 종목 추천 (복잡함)            |

### **규칙 기반 로직**

```py
if PER < 10 and ROE > 15:
    buy()
```

---

## `position`과 `diff()` 의미

```py
signal = [0, 0, 1, 1, 0, -1, -1, 0]
position = signal.diff()
```

| 일자 | signal | diff() = position |
|------|--------|-------------------|
| 1    | 0      | NaN               |
| 2    | 0      | 0                 |
| 3    | 1      | 1  ← 매수 신호       |
| 4    | 1      | 0                 |
| 5    | 0      | -1 ← 매도 신호       |
| 6    | -1     | -1 (보유 → 매도)    |
| 7    | -1     | 0                 |
| 8    | 0      | 1  (매도 → 보유)    |

- `1`: 매수 시작
- `-1`: 매도 시작

### 요약

- `signal`: 현재 상태 (0: 없음, 1: 매수 중, -1: 매도 중)
- `diff()`: **변화량**을 계산해서 신호 발생 시점을 잡음

---

## 로보어드바이저 예제 

### 예제 설명
- 5일, 20일 이동평균 계산
- 골든크로스 발생 시 매수, 데드크로스 발생 시 매도

```py
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# 1. 데이터 불러오기
symbol = 'AAPL'  # 애플 주식
stock = yf.download(symbol, start='2023-01-01', end='2024-01-01')

# 2. 이동 평균 계산
stock['MA5'] = stock['Close'].rolling(window=5).mean()
stock['MA20'] = stock['Close'].rolling(window=20).mean()

# 3. 매매 신호 생성
stock['signal'] = 0
stock.loc[stock['MA5'] > stock['MA20'], 'signal'] = 1
stock.loc[stock['MA5'] < stock['MA20'], 'signal'] = -1
stock['position'] = stock['signal'].diff()

# 4. 시각화
plt.figure(figsize=(14, 6))
plt.plot(stock['Close'], label='Close Price')
plt.plot(stock['MA5'], label='MA5')
plt.plot(stock['MA20'], label='MA20')
plt.plot(stock[stock['position'] == 1].index, 
         stock['MA5'][stock['position'] == 1], '^', color='g', label='Buy Signal')
plt.plot(stock[stock['position'] == -1].index, 
         stock['MA5'][stock['position'] == -1], 'v', color='r', label='Sell Signal')
plt.legend()
plt.title(f"{symbol} 골든/데드 크로스 전략")
plt.show()
```

---

## 정리 요약

- **전일 변화율**: 빠른 변화 감지, 헛신호 있음
- **이동 평균**: 추세 추종, 골든크로스/데드크로스 판단에 유용
- **로보어드바이저**: 규칙 기반부터 딥러닝까지 다양, 초보자는 룰 기반 추천
- **코드 실습**: `MA5 > MA20`이면 매수, 아니면 매도

---

2015-01-01 ~ 2025-04-08 까지의 철강 데이터를 활용

```py
# gpt _ 간단한 로드어드바이저 코드 작성  
import warnings
import matplotlib
import sys
import numpy as np
import os
import pandas as pd
warnings.filterwarnings(action='ignore')
if 'google.colab' in sys.modules:
    !echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
    # 나눔 폰트를 설치
    !sudo apt-get -qq -y install fonts-nanum
    import matplotlib.font_manager as fm
    font_files = fm.findSystemFonts(fontpaths=['/usr/share/fonts/truetype/nanum'])
    for fpath in font_files:
        fm.fontManager.addfont(fpath)
matplotlib.rcParams['font.family'] = 'NanumGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('미국 철강 코일 선물 과거 데이터.csv', parse_dates=['날짜'], index_col="날짜", thousands=",")
df['거래량'] = df['거래량'].apply(lambda x: float(x.replace('K', '')) * 1000 if isinstance(x, str) and 'K' in x else float(x))
df['변동 %'] = df['변동 %'].apply(lambda x: float(x.replace('%', '')) / 100 if isinstance(x, str) else x)
df = df.sort_index()
df.dropna(inplace=True)

# 전일 종가와 비교하여 상승/하락 여부 판단
def trading_signal(df, threshold=1.0):
    """
    종가 기준으로 전일 대비 상승(매수) 또는 하락(매도)을 판단하는 함수
    :param df: 데이터프레임
    :param threshold: 매수/매도 판단을 위한 종가 변화율 임계값 (기본 1%)
    :return: 매수/매도 신호

     pct_change : (다음행 - 현재행)÷현재행
     ex. 20일의 종가가 100이고 21일의 종가가 101이라면 pct_change()의 결과는 (101 - 100) / 100 = 0.01로 1% 상승
     변화율이 1.0일 경우 매수, 변화율이 -1.0일 경우 매도, 그 외에는 보유라는 신호를 띄운다. 
    """
    df['변화율'] = df['종가'].pct_change() * 100  # 종가 변화율 계산
    df['신호'] = df['변화율'].apply(lambda x: '매수' if x >= threshold else ('매도' if x <= -threshold else '보유'))
    return df[['종가', '변화율', '신호']]

# 로드 어드바이저 실행
signal_df = trading_signal(df)

# 결과 출력
print(signal_df)
```

```
               종가        변화율  신호
날짜                              
2015-01-07  597.0        NaN  보유
2015-01-09  594.0  -0.502513  보유
2015-01-30  550.0  -7.407407  매도
2015-02-02  550.0   0.000000  보유
2015-03-30  475.0 -13.636364  매도
...           ...        ...  ..
2025-04-01  895.0   0.561798  보유
2025-04-02  915.0   2.234637  매수
2025-04-03  913.0  -0.218579  보유
2025-04-04  932.0   2.081051  매수
2025-04-07  930.0  -0.214592  보유

[1679 rows x 3 columns]
```
```py
df['신호'].hist()
```
![image](https://github.com/user-attachments/assets/bb17084e-7d14-4bf7-99b6-e115515ae1b3)



## 1. 전일 대비 종가 변화를 활용한 간단한 로보 아드버이저

### 로보 아드버이저
- **전일 대비 종가의 변화율**을 기준으로 매수/매도/보유 신호를 생성한다.
- 특정 임계값(`threshold`)을 초과하거나 미만일 경우 **신호를 자동 생성**한다.
- 기초적인 규칙 기반(rule-based) 로보 아드버이저

### 코드
```python
# 종가 변화율 계산하고 신호 생성

df['변화율'] = df['종가'].pct_change() * 100 # 변화율(%)을 계산해서 상승률/하락률을 얻기  
df['신호'] = df['변화율'].apply(
    lambda x: '매수' if x >= threshold else ('매도' if x <= -threshold else '보유')
)
```
변화율이 +threshold 이상이면 매수, 변화율이 -threshold 이하이면 매도, 그 외는 보유

### 추가 설명
- `pct_change()` : (현재 - 이전) / 이전 종가 차이그림
- `apply()` 을 통해 변화율이 `+threshold`이면 `메수`, `-threshold`이면 `메도`, 그 외에는 `보유`로 초과

### 결과 예시
| 종가 | 변화율 | 신호 |
|--------|------------|-------|
| 100    | NaN        | 보유  |
| 102    | 2.00       | 매수  |
| 101    | -0.98      | 보유  |
| 99     | -1.98      | 매도  |



<br><br><br><br>

---

```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 데이터 로딩
df = pd.read_csv('미국 철강 코일 선물 과거 데이터.csv', parse_dates=['날짜'], index_col="날짜", thousands=",")
df['거래량'] = df['거래량'].apply(lambda x: float(x.replace('K', '')) * 1000 if isinstance(x, str) and 'K' in x else float(x))
df['변동 %'] = df['변동 %'].apply(lambda x: float(x.replace('%', '')) / 100 if isinstance(x, str) else x)
df.dropna(inplace=True)

# 이동평균 설정
short_window = 50
long_window = 200

# 이동평균 계산
df['short_mavg'] = df['종가'].rolling(window=short_window, min_periods=1).mean()
df['long_mavg'] = df['종가'].rolling(window=long_window, min_periods=1).mean()

# 매수/매도 신호 생성
df['signal'] = 0
df['signal'][short_window:] = np.where(df['short_mavg'][short_window:] > df['long_mavg'][short_window:], 1, 0)
df['position'] = df['signal'].diff()

# 결과 출력
plt.figure(figsize=(10,5))
plt.plot(df['종가'], label='Close Price')
plt.plot(df['short_mavg'], label=f'{short_window}-Day Moving Average')
plt.plot(df['long_mavg'], label=f'{long_window}-Day Moving Average')

# 매수/매도 지점 표시
plt.plot(df[df['position'] == 1].index, df['short_mavg'][df['position'] == 1], '^', markersize=10, color='g', lw=0, label='Buy Signal')
plt.plot(df[df['position'] == -1].index, df['short_mavg'][df['position'] == -1], 'v', markersize=10, color='r', lw=0, label='Sell Signal')

plt.title('Buy and Sell Signals Based on Moving Average')
plt.legend(loc='best')
plt.show()
```
![image](https://github.com/user-attachments/assets/b949768e-a354-4ad6-90b1-9cbb8acedb3e)


## 2. 이동 평균 기반 로보 아드버이저 (골든 크로스 전력)

### 로보 아드버이저
- 단기/장기 이동 평균선의 교차를 기준으로 매매 신호를 생성  
- `50일` 이동 평균이 `200일` 이동 평균을 돌파하면 **매수**, 다지면 **매도**

### 표현 코드
```python
# 이동평균 계산
short_window = 50
long_window = 200

df['short_mavg'] = df['종가'].rolling(window=short_window, min_periods=1).mean()
df['long_mavg'] = df['종가'].rolling(window=long_window, min_periods=1).mean()

# 신호 생성하고 경계

df['signal'] = 0
df['signal'][short_window:] = np.where(
    df['short_mavg'][short_window:] > df['long_mavg'][short_window:], 1, 0
)
df['position'] = df['signal'].diff()
```
signal: 현재 단기선이 장기선보다 높으면 1, 낮으면 0

position: 변화가 생긴 부분 → 1은 매수, -1은 매도

### 결과 그래프
- 파란선: 종가
- 주황선: 50일 이동평균
- 천지선: 200일 이동평균
- 초록 삼각형 ↑: 매수 시점
- 빨간 삼각형 ↓: 매도 시점

---

## 평가
| 관점 | 전일 변화율 | 이동 평균 |
|--------|----------------|----------------|
| 전력 | 단기 변화 감지 | 차편 관련 최적화 |
| 재점 | 간단, 지금 매무에 최적 | 교차 발생 시 지적 |
| 평가 | 복잡한 날짜에 복가 시 오류 | 신호 복가 지적 가능 |


<br><br>

- 이동평균 설정 :

  - `short_window = 50` : 50일 이동평균을 계산한다.

  - `long_window = 200` : 200일 이동평균을 계산한다.

  이동평균은 특정 기간 동안의 주가 평균을 계산하여 그 추세를 확인하려는 기법이다. 

  50일 이동평균은 비교적 단기적인 추세를 나타내며 200일 이동평균은 더 긴 기간을 기준으로 한 추세를 나타낸다.

- rolling 함수 :

  - `rolling(window=short_window, min_periods=1)`은 주어진 window 기간(여기서는 50일)에 대해 이동평균을 계산한다. 

  - `min_periods=1`은 계산할 수 있는 최소 기간을 의미한다. 

    - ex. 첫 50일 동안은 50일의 이동평균을 계산하지만  

      그보다 적은 기간이 주어지면 최소 1일만 사용하여 이동평균을 계산한다.

- np.where 함수 :

  - `np.where(df['short_mavg'] > df['long_mavg'], 1, 0)`는 조건문을 사용하여 

    단기 이동평균이 장기 이동평균보다 크면 1(매수 신호)을, 그렇지 않으면 0(매도 신호)을 할당한다.

  - 즉 단기 이동평균선이 장기 이동평균선 위에 있을 때 매수 신호, 아래에 있을 때 매도 신호가 된다.

- diff 함수 :

  - `df['position'] = df['signal'].diff()`는 차분을 계산한다.

  - 차분(diff())은 이전 값과 현재 값의 차이를 구한다. 

    그래서 신호가 바뀐 지점(매수→매도 또는 매도→매수)에서 1 또는 -1 값을 갖게 된다.

  - 예를 들어 signal 값이 0에서 1로 바뀌면 position 값은 1, 1에서 0으로 바뀌면 position 값은 -1이 된다. 

- 그래프 해석 :

  - 녹색 삼각형(▲): 매수 신호를 나타낸다. 이는 단기 이동평균이 장기 이동평균을 넘어설 때 발생한다.

  - 빨간색 삼각형(▼): 매도 신호를 나타낸다. 이는 단기 이동평균이 장기 이동평균을 하회할 때 발생한다.

- 그래프에서의 각 범례 해석 :

  - Close Price : 주가의 실제 종가

  - 50-Day Moving Average : 단기 이동평균 (50일간의 평균)

  - 200-Day Moving Average : 장기 이동평균 (200일간의 평균)

  - Buy Signal : 단기 이동평균이 장기 이동평균을 상향 돌파할 때 발생한 매수 신호

  - Sell Signal : 단기 이동평균이 장기 이동평균을 하향 돌파할 때 발생한 매도 신호

<br><br>

이동평균은 주식이나 자산의 평균 가격을 시간에 따라 계산하는 방법으로 주로 가격의 추세를 파악하기 위해 사용된다.

단기 이동평균은 최근 데이터(예: 50일)만 반영한 평균이므로 빠르게 반응한다. 

장기 이동평균은 더 긴 기간의 데이터를 반영하기 때문에 가격 변화에 느리게 반응한다.  

단기 상승세가 지속될 가능성 : 단기적으로 상승세가 뚜렷하고 

그 상승세가 장기 추세까지 뚫고 올라왔다는 것은 강한 상승 신호다.



