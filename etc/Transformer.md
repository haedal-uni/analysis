# Transformer 
Transformer 모델은 **시간이 흐르는 데이터(= 시계열)** 를 잘 예측해준다.

- 주식 가격 예측 📈

- 날씨 예보 ⛅

- 번역하기 (한국어 → 영어) 🗣️

ex. “내일 주식이 오를까?” 라는 질문을 한다면

Transformer는 과거 가격 중 중요한 부분에 **주의(attention)** 를 집중해서 답을 해준다.

<br><br><br>

## Transformer의 핵심
| 구성 요소 | 설명 📝 |
| --- | --- |
| `Embedding` | 숫자로 된 데이터를 Transformer가 이해하기 쉽게 바꿔줌 |
| `Multi-Head Attention` | 여러 방향에서 정보를 동시에 바라봄 (중요한 정보에 집중!) |
| `Feed Forward` | 뇌처럼 계산을 함 |
| `Positional Encoding` | 시계열 데이터라서 **순서 정보**가 중요! 순서를 기억하게 해줌 |
| `Decoder` | 미래 값을 예측하는 뇌 |

<br><br><br>

## Transformer는 언제 사용?
| 사용 상황 | 예시 |
| --- | --- |
| 시계열 예측 | 주가, 온도, 판매량 예측 등 |
| 자연어 처리 | 번역기, 챗봇 |
| 음악 생성 | 시간 흐름을 고려해서 음표 예측 |

<br><br><br>

## 코드 
### 요약
```
1. 데이터를 불러온다           👉 yfinance 또는 CSV
2. 정규화한다                👉 숫자를 0~1로 바꿈
3. 데이터를 묶는다            👉 [과거 50일 → 미래 10일] 예측용
4. Transformer 모델 정의     👉 뇌 만들기
5. 학습시킨다                👉 데이터를 보여주며 예측 연습
6. 미래를 예측한다            👉 테스트용 데이터로 실전 예측
7. 시각화한다                👉 그래프 그려서 성능 보기
```

### 1. 데이터 불러오기 & 전처리
```py
df = yf.download(ticker, start=start_date, end=end_date)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
```

### 2. 시계열 데이터를 묶는 Dataset 클래스
```
src = 과거 50일
tgt = 미래 10일
```

### 3. Transformer 모델 구조
```py
self.embedding = nn.Linear(input_dim, d_model)  # 5 → 64
self.transformer = nn.Transformer(...)
self.fc_out = nn.Linear(d_model, input_dim)     # 64 → 5
```
5개의 정보를 64차원으로 바꿔서 계산하고 다시 5개로 되돌려 예측한다.

### 4. 학습 방식 (Teacher Forcing)
```py
teacher_forcing_ratio = 0.2
```
80%는 정답 없이 스스로 예측, 20% 확률로 정답을 힌트로 줘서 예측 훈련

### 4. 예측 결과 보기
```py
smoothed_predictions = np.convolve(..., mode='valid')
```
예측 결과가 울퉁불퉁할 수 있어서 부드럽게 다듬어요 (이걸 smoothing이라 한다).

<br><br><br>

## 용어 정리 
| 용어 | 설명 |
| --- | --- |
| 시계열 | 시간이 흐름에 따라 생긴 데이터 (예: 주가, 날씨) |
| Transformer | 주의 집중해서 미래를 예측하는 뇌 |
| Embedding | 숫자를 더 이해하기 쉽게 바꿔줌 |
| Decoder | 미래를 만들어내는 뇌 |
| Teacher Forcing | 예측할 때 정답을 살짝 알려주는 방식 |
| Smoothing | 예측 결과를 부드럽게 다듬기 |




