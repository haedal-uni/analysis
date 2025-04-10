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



