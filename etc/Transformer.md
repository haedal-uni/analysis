# Transformer 
Transformer 모델은 **시간이 흐르는 데이터(= 시계열)** 를 잘 예측해준다.

- 주식 가격 예측 

- 날씨 예보 

- 번역하기 (한국어 → 영어) 

ex. “내일 주식이 오를까?” 라는 질문을 한다면

Transformer는 과거 가격 중 중요한 부분에 **주의(attention)** 를 집중해서 답을 해준다.


Transformer 모델은 주로 **자연어 처리(NLP)** 에서 활용되는 딥러닝 모델이지만 

시계열 예측을 비롯한 다양한 분야에도 강력한 성능을 보여주고 있다. 

이 모델은 순차적인 데이터에서 중요한 정보를 추출하고 이를 바탕으로 다음 값을 예측하거나 출력을 생성하는 데 특화되어 있다.


**Transformer의 주요 특징**     

Transformer는 단어 또는 시계열 데이터 같은 연속된 데이터를 다룰 때 강력한 성능을 발휘한다. 

예를 들어, 주식 가격 예측, 날씨 예보, 번역과 같은 작업에 잘 활용된다. 

이 모델의 핵심 아이디어는 주목(attention) 메커니즘을 통해 데이터 내에서 중요한 부분에 집중하는 것이다.


<br><br><br>

## Transformer의 핵심
| 구성 요소 | 설명 |
| --- | --- |
| `Embedding` | 숫자 데이터나 텍스트 데이터를 Transformer가 이해할 수 있는 형태로 변환. <br> 예를 들어, 텍스트의 각 단어를 고차원 벡터로 표현 |
| `Multi-Head Attention` | 여러 개의 attention을 동시에 적용하여 입력 데이터의 중요한 부분에 집중한다. <br> 이를 통해 더 넓은 관점에서 정보를 파악할 수 있다. |
| `Feed Forward` | 각 단계에서 계산을 처리하는 네트워크로, 뇌에서 뉴런들이 처리하는 방식처럼 데이터를 처리한다. |
| `Positional Encoding` | 시계열 데이터에서는 순서 정보가 중요하기 때문에 입력 데이터의 순서를 모델에 인식시켜 주는 기능이다. |
| `Decoder` | 주어진 입력을 바탕으로 미래 값을 예측한다. <br> 주식 가격 예측처럼 과거 데이터를 기반으로 미래 값을 추정한다. |

<br><br><br>

## Transformer는 언제 사용?
| 사용 상황 | 예시 |
| --- | --- |
| 시계열 예측 | 주가, 온도, 판매량 예측 등 |
| 자연어 처리 | 번역기, 챗봇, 문서 요약 등 |
| 음악 생성 | 음악의 시간 흐름을 고려한 음표 예측 |

<br>

시계열 예측을 할 때, Transformer 모델은 과거 데이터를 바탕으로 미래 값을 예측하는 데 유용합니다. 

예를 들어 주식 가격 예측을 한다면 

Transformer는 과거 주식 가격 중 중요한 부분에 **주목(attention)** 을 집중하여 미래의 가격을 예측한다. 

Transformer는 기존의 ARIMA나 LSTM과 같은 모델과 비교해도 더 길고 복잡한 패턴을 잘 학습할 수 있는 장점이 있다.

<br><br><br>

## 코드 
### 요약
```
1. 데이터를 불러온다 → yfinance 또는 CSV
2. 정규화한다 → 숫자를 0~1로 바꿈
3. 데이터를 묶는다 → [과거 50일 → 미래 10일] 예측용
4. Transformer 모델 정의 → 뇌 만들기
5. 학습시킨다 → 데이터를 보여주며 예측 연습
6. 미래를 예측한다 → 테스트용 데이터로 실전 예측
7. 시각화한다 → 그래프 그려서 성능 보기
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

### 5. 예측 결과 보기 
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


---

## 코드 실행
Transformer 모델은 과거 데이터를 바탕으로 미래 값을 예측하는 데 유용해서 종가 가격 예측으로 사용했다


### 1. 데이터 로딩 및 전처리
```py
# 데이터 로드 및 전처리 함수 수정
def load_and_preprocess_data(file_path, features=['종가', '거래량', '시가', '고가', '저가'], split_ratio=0.8):
    # CSV 파일 로드 (이미 존재하는 데이터)
    df = pd.read_csv(file_path, parse_dates=['날짜'], index_col="날짜", thousands=",")
    
    # 거래량 컬럼 처리 (단위가 'K'로 표시된 경우 1000을 곱해줌)
    df['거래량'] = df['거래량'].apply(lambda x: float(x.replace('K', '')) * 1000 if isinstance(x, str) and 'K' in x else float(x))
    
    # 필요한 피처들만 선택
    data = df[features].values  # [num_samples, num_features]
    
    # 데이터 전처리
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    # 훈련/테스트 데이터 분리
    split_idx = int(len(data_scaled) * split_ratio)
    train_data = data_scaled[:split_idx]
    test_data = data_scaled[split_idx:]
    
    print(f"Data Sample: {data[:5]}")
    print(f"Data Shape: {data.shape}")
    print(f"Train Data Shape: {train_data.shape}")
    print(f"Test Data Shape: {test_data.shape}")
    
    return train_data, test_data, scaler
```
- `load_and_preprocess_data` : 주식 데이터를 다운로드하고 데이터를 정규화한 후 훈련 데이터와 테스트 데이터를 분할하는 함수

  - ticker: 주식 종목 코드 (예: "AAPL"은 애플 주식)

  - start_date, end_date: 주식 데이터를 가져올 시작 날짜와 종료 날짜

  - features: 사용할 주식 지표 (종가, 거래량 등)

  - split_ratio: 훈련 데이터와 테스트 데이터로 나눌 비율 (기본값은 80%)
  
### 2. Dataset 및 DataLoader 구성 (멀티스텝 예측 버전)
이 클래스는 주식 시계열 데이터를 배치로 나누어 훈련에 사용할 수 있게 만드는 역할을 한다.

```py
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, pred_length):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_length = seq_length
        self.pred_length = pred_length
'''
data: 주식 데이터를 Tensor 형태로 변환한다. (Tensor는 PyTorch에서 데이터 구조다.)

seq_length: 시퀀스 길이: 모델이 과거 몇 개의 데이터를 사용할지를 결정한다.  

pred_length: 예측 길이: 모델이 예측하려는 미래 시점 수다.
'''

    def __len__(self): # 데이터셋의 길이를 반환하는데 seq_length와 pred_length를 고려해서 가능한 데이터 샘플의 수를 계산
        return len(self.data) - self.seq_length - self.pred_length + 1

    def __getitem__(self, idx): # __getitem__: 인덱스 idx에서 과거 데이터(src)와 미래 데이터(tgt)를 잘라내어 반환
        src = self.data[idx: idx+self.seq_length]
        tgt = self.data[idx+self.seq_length: idx+self.seq_length+self.pred_length]
        return src, tgt
'''
src: 예측할 과거 데이터를 의미한다. seq_length만큼 가져온다.

tgt: 예측할 미래 데이터를 의미한다. pred_length만큼 가져온다.
'''

# 데이터를 배치 단위로 나누어 훈련할 수 있게 도와주는 DataLoader를 생성하는 함수 
def create_dataloader(data, seq_length, pred_length, batch_size, shuffle=True):
    dataset = TimeSeriesDataset(data, seq_length, pred_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
'''
batch_size: 한 번에 처리할 데이터의 수

shuffle: 데이터를 섞을지 말지를 결정
'''
```
- Dataset : PyTorch에서 데이터를 처리할 때 사용하는 클래스

- TimeSeriesDataset : 시계열 데이터를 다루는 사용자 정의 클래스


### 3. Transformer 기반 시계열 예측 모델 정의 (멀티스텝 예측 지원)
```py
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout
        )
        self.fc_out = nn.Linear(d_model, input_dim)

    def forward(self, src, tgt):
        # src: [S, N, input_dim], tgt: [T, N, input_dim]
        src_emb = self.embedding(src)   # [S, N, d_model]
        tgt_emb = self.embedding(tgt)     # [T, N, d_model]
        output = self.transformer(src_emb, tgt_emb)  # [T, N, d_model]
        return self.fc_out(output)      # [T, N, input_dim]
```
- nn.Module은 PyTorch에서 모든 모델의 기본 클래스

- embedding: 입력 데이터를 d_model 차원으로 변환하는 선형 계층이다. (입력 차원 input_dim을 d_model 차원으로 매핑)

<br>

- transformer: Transformer 모델을 정의한다.

  - nhead: Attention 헤드의 수

  - num_layers: 인코더와 디코더의 층 수
 
- fc_out: Transformer의 출력을 다시 원래의 입력 차원으로 변환하는 선형 계층이다.

- forward: 입력 데이터 src와 tgt를 embedding을 통해 변환한다.

- transformer는 과거 데이터(src)와 미래 데이터(tgt)를 처리하여 예측 결과를 출력한다.

  그 후 `fc_out`을 통해 예측 결과를 원래의 차원으로 되돌린다.

  
<br><br>

### 4. 모델 학습 함수 (수정된 Loss Function 및 Teacher Forcing)
```py
def train_model(model, train_loader, device, epochs, learning_rate=0.0005, teacher_forcing_ratio=0.2):
    model.train()
    criterion = nn.HuberLoss()  # Huber Loss로 변경
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in range(epochs):
        epoch_loss = 0
        for src, tgt in train_loader:
            src = src.to(device)   # [batch, seq_length, input_dim]
            tgt = tgt.to(device)   # [batch, pred_length, input_dim]

            src = src.transpose(0, 1)  # [seq_length, batch, input_dim]
            batch_size = src.size(1)
            input_dim = src.size(2)
            pred_length = tgt.size(1)

            # Teacher Forcing 결정
            use_teacher_forcing = True if np.random.rand() < teacher_forcing_ratio else False
            if use_teacher_forcing:
                start_token = torch.zeros(1, batch_size, input_dim).to(device)
                tgt_transposed = tgt.transpose(0, 1)  # [pred_length, batch, input_dim]
                decoder_input = torch.cat([start_token, tgt_transposed[:-1]], dim=0)
            else:
                decoder_input = torch.zeros(pred_length, batch_size, input_dim).to(device)

            optimizer.zero_grad()
            output = model(src, decoder_input)  # [pred_length, batch, input_dim]
            loss = criterion(output, tgt.transpose(0, 1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(train_loader):.6f}, LR: {current_lr:.6f}")
```
- `model.train()` : 모델을 훈련 모드로 설정한다. 모델이 학습을 할 준비가 되었다는 뜻


### 5. 미래 예측 (결과 Smoothing 추가)  
```py
def predict_future(model, test_data, seq_length, pred_length, total_predictions, device, smooth_window=5):
    model.eval()
    test_input = torch.tensor(test_data[:seq_length], dtype=torch.float32).to(device)  # [seq_length, input_dim]
    predictions = []
    with torch.no_grad():
        steps = total_predictions // pred_length
        remainder = total_predictions % pred_length
        for _ in range(steps):
            src = test_input.unsqueeze(1)  # [seq_length, 1, input_dim]
            decoder_input = torch.zeros(pred_length, 1, test_input.size(-1)).to(device)
            out = model(src, decoder_input)    # [pred_length, 1, input_dim]
            out = out.squeeze(1)               # [pred_length, input_dim]
            predictions.append(out.cpu().numpy())
            test_input = torch.cat([test_input[pred_length:], out], dim=0)
        if remainder > 0:
            src = test_input.unsqueeze(1)
            decoder_input = torch.zeros(remainder, 1, test_input.size(-1)).to(device)
            out = model(src, decoder_input)
            out = out.squeeze(1)
            predictions.append(out.cpu().numpy())
        predictions = np.concatenate(predictions, axis=0)

    # Smoothing 예측 결과
    smoothed_predictions = np.convolve(predictions[:, 0], np.ones(smooth_window)/smooth_window, mode='valid')
    return predictions, smoothed_predictions
```

### 6. 시각화 함수 (RMSE 추가)
```py
def plot_predictions(actual, predictions, smoothed_predictions, seq_length):
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(actual[seq_length:seq_length + len(predictions), 0], predictions[:, 0]))
    plt.figure(figsize=(12,6))
    plt.plot(actual[:, 0], label="Actual Close Price")
    plt.plot(range(seq_length, seq_length + len(predictions)), predictions[:, 0],
             label="Predicted Close Price", linestyle="dashed")
    plt.plot(range(seq_length, seq_length + len(smoothed_predictions)), smoothed_predictions,
             label="Smoothed Predictions", linestyle="dotted")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.title(f"Tesla Stock Price Prediction using Transformer (RMSE: {rmse:.2f})")
    plt.legend()
    plt.show()
```

### 7. 메인 실행 함수
```py
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 데이터 로드 및 전처리 함수
def load_and_preprocess_data(file_path, features=['종가', '거래량', '시가', '고가', '저가'], split_ratio=0.8):
    # CSV 파일 로드 (이미 존재하는 데이터)
    df = pd.read_csv(file_path, parse_dates=['날짜'], index_col="날짜", thousands=",")
    
    # 거래량 컬럼 처리 (단위가 'K'로 표시된 경우 1000을 곱해줌)
    df['거래량'] = df['거래량'].apply(lambda x: float(x.replace('K', '')) * 1000 if isinstance(x, str) and 'K' in x else float(x))
    
    # 필요한 피처들만 선택
    data = df[features].values  # [num_samples, num_features]
    
    # 데이터 전처리
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    # 훈련/테스트 데이터 분리
    split_idx = int(len(data_scaled) * split_ratio)
    train_data = data_scaled[:split_idx]
    test_data = data_scaled[split_idx:]
    
    print(f"Data Sample: {data[:5]}")
    print(f"Data Shape: {data.shape}")
    print(f"Train Data Shape: {train_data.shape}")
    print(f"Test Data Shape: {test_data.shape}")
    
    return train_data, test_data, scaler

# DataLoader 생성 함수 (미리 정의되어 있어야 합니다)
def create_dataloader(data, seq_length, pred_length, batch_size):
    # 데이터 로더를 생성하는 로직 (여기서는 예시로 작성)
    pass

# 모델 학습 함수 (미리 정의되어 있어야 합니다)
def train_model(model, train_loader, device, epochs, learning_rate=0.0005, teacher_forcing_ratio=0.2):
    pass

# 예측 함수 (미리 정의되어 있어야 합니다)
def predict_future(model, test_data, seq_length, pred_length, total_predictions, device, smooth_window=5):
    pass

# 결과 시각화 함수 (미리 정의되어 있어야 합니다)
def plot_predictions(actual_test, predictions_inverse, smoothed_predictions, seq_length):
    pass

# 메인 실행 함수
def main():
    # 설정값
    file_path = "미국 철강 코일 선물 과거 데이터.csv"  # CSV 파일 경로
    seq_length = 60  # 시퀀스 길이
    pred_length = 10  # 예측 길이
    batch_size = 32  # 배치 크기
    epochs = 100  # 학습 에포크 수
    total_predictions = 200  # 예측할 총 값
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU가 있으면 GPU 사용

    # 데이터 로딩 및 전처리
    train_data, test_data, scaler = load_and_preprocess_data(
        file_path,
        features=['종가', '거래량', '시가', '고가', '저가'],
        split_ratio=0.8
    )

    # DataLoader 생성
    train_loader = create_dataloader(train_data, seq_length, pred_length, batch_size)

    # 모델 생성
    model = TimeSeriesTransformer(input_dim=5, d_model=256, nhead=4, num_layers=4).to(device)

    # 모델 학습
    train_model(model, train_loader, device, epochs, learning_rate=0.0005, teacher_forcing_ratio=0.2)

    # 미래 예측
    predictions, smoothed_predictions = predict_future(
        model, test_data, seq_length, pred_length, total_predictions, device, smooth_window=5
    )
    predictions_inverse = scaler.inverse_transform(predictions)
    actual_test = scaler.inverse_transform(test_data)

    # 결과 시각화
    plot_predictions(actual_test, predictions_inverse, smoothed_predictions, seq_length)
```

