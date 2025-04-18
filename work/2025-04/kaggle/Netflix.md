```py
x_train = train[['Open', 'High', 'Low', 'Volume']].values
x_test = test[['Open', 'High', 'Low', 'Volume']].values
```
입력값(Features)은 주가 관련 변수들 중에서

**Open(시가), High(고가), Low(저가), Volume(거래량)** 을 선택

.values는 DataFrame을 Numpy 배열로 변환한다 (모델에 넣기 위해)

<br><br><br>

```py
y_train = train['Close'].values
y_test = test['Close'].values
```
출력값(Target)은 **Close(종가)** 다.

모델이 Open, High, Low, Volume을 보고 Close(종가)를 예측하는 구조

<br><br><br>

### 선형 회귀 모델 학습
```py
model_lnr = LinearRegression()
model_lnr.fit(x_train, y_train)
```
`LinearRegression()`은 선형 회귀 모델이다.  

`.fit()`으로 학습을 시킨다.

→ 주어진 X 데이터에 대해 y를 잘 맞추는 직선(또는 평면)을 찾음.

<br><br><br>

### 테스트 데이터 예측 및 실제 예시 예측
```py
y_pred = model_lnr.predict(x_test)
```
테스트 입력값에 대해 예측값 출력

<br><br><br>

```py
result = model_lnr.predict([[262.000000, 267.899994, 250.029999, 11896100]])
print(result)
```
임의의 새로운 데이터 (실제 값들을 수동으로 넣음)에 대해 종가 예측.

출력: [257.54904974] → 예측된 종가

<br><br><br>

### 성능 지표 출력
```py
print("MSE", round(mean_squared_error(y_test, y_pred), 3))
print("RMSE", round(np.sqrt(mean_squared_error(y_test, y_pred)), 3))
print("MAE", round(mean_absolute_error(y_test, y_pred), 3))
print("MAPE", round(mean_absolute_percentage_error(y_test, y_pred), 3))
print("R2 Score : ", round(r2_score(y_test, y_pred), 3))
```
지표	설명
MSE	평균 제곱 오차 (오차를 제곱해서 평균)
RMSE	MSE의 제곱근. 실제 오차의 크기와 유사한 단위
MAE	평균 절대 오차
MAPE	평균 절대 백분율 오차
R2 Score	결정 계수. 1에 가까울수록 모델이 잘 맞춤 (1은 완벽 예측)

<br><br><br>

```py
def style():
    plt.figure(facecolor='black', figsize=(15,10))
    ax = plt.axes()

    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')

    ax.set_facecolor("black")
```
시각화 스타일을 검정 배경 + 흰색 선으로 통일.

plt.figure()로 크기 및 배경색 설정.

plt.axes()로 축 설정 → 글씨, 테두리 등 색깔 조절

<br><br><br>

### 날짜 처리 및 데이터 구성 
```py
viz['Date'] = pd.to_datetime(viz['Date'], format='%Y-%m-%d')
```
Date 컬럼을 문자열에서 datetime 타입으로 변환 (시계열 분석을 위함).

<br><br>

```py
data = pd.DataFrame(viz[['Date','Close']])
data = data.reset_index()
data = data.drop('index', axis=1)
```
Date와 Close만 추출해서 새로운 DataFrame data 생성.

`reset_index()`로 인덱스를 새로 만들고, 기존 인덱스는 제거.

<br><br>

```py
data.set_index('Date', inplace=True)
```
Date를 인덱스로 설정 → 시계열 분석이나 플롯용.

<br><br>

```py
data = data.asfreq('D')
```
빈 날짜도 포함해 일 단위 빈도로 시계열 재정렬.

<br><br>

```py
test_pred[['Close', 'Close_Prediction']].describe().T
```
예측 결과 DataFrame에서 Close와 Close_Prediction 컬럼의 기초 통계 요약




