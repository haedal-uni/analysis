```py

# 데이터셋 준비
class Cu(Dataset):  # 학습용 데이터 준비
    def __init__(self):
        # df 데이터 사용
        self.data = df[['시가', '고가', '저가']].values  # 입력 : 3가지
        self.label = df['종가'].values  # 정답 : 종가

    # 사용 가능한 데이터 개수
    def __len__(self):
        return len(self.data) - 30  # 사용 가능한 배치 개수

    # 데이터와 라벨 반환
    def __getitem__(self, i): # 30일치 주식 데이터를 가지고 그 다음 날의 주식 종가를 예측하려는 목적
        data = self.data[i:i+30]  # 입력 데이터 30일치(30일치 데이터를 입력으로)
        label = self.label[i+30]  # 종가 데이터 30일치(그 다음날의 종가를 정답으로)

        return data, label


# RNN 클래스는 PyTorch의 nn.Module을 상속받아 RNN 모델을 정의
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__() # 부모 클래스인 nn.Module의 초기화 메서드

        # RNN층 정의
        self.rnn = nn.RNN(input_size=3, hidden_size=8, num_layers=5, batch_first=True)
        # input_size=3 : 입력 데이터의 특성 수 (시가, 고가, 저가 3개)
        # hidden_size=8 : 모델의 "기억" 용량
        # num_layers=5 : 5층 구조

        # MLP (선형층)으로 종가 예측
        ## 30일 동안 총 30개의 기억(출력)이 생기고 그게 8칸씩이니까 👉 30 × 8 = 240
        self.fc1 = nn.Linear(in_features=240, out_features=64) # 30일 × hidden_size(8) = 240
        self.fc2 = nn.Linear(in_features=64, out_features=1) # 마지막으로 예측할 값, 종가 한 개를 출력

        self.relu = nn.ReLU()  # 활성화 함수(음수는 버리고 양수만 통과)

    def forward(self, x, h0):
        x, hn = self.rnn(x, h0)  # RNN층의 출력(순서대로 RNN을 통과시킴)

        # MLP층의 입력으로 사용되게 모양 변경
        x = torch.reshape(x, (x.shape[0], -1)) # (batch_size, 240)로 바꿈

        # MLP층을 이용해 종가 예측
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        # 예측한 종가를 1차원 벡터로 표현
        x = torch.flatten(x)

        return x


# 학습 준비
device = "cuda" if torch.cuda.is_available() else "cpu" # 컴퓨터에 그래픽카드(GPU)가 있으면 GPU 쓰고 없으면 CPU 쓰기
model = RNN().to(device)  # 모델 정의
dataset = Cu()       # 데이터셋 정의

# 배치 크기 설정
batch_size = 32
loader = DataLoader(dataset, batch_size=batch_size)  # 배치 크기 32로 설정

# 최적화 정의
optim = Adam(params=model.parameters(), lr=0.0001)  # 최적화 설정

# 학습 루프
for epoch in range(200):
    iterator = tqdm.tqdm(loader)
    for data, label in iterator:
        optim.zero_grad()

        # 초기 은닉 상태(초기 기억)
        h0 = torch.zeros(5, data.shape[0], 8).to(device)

        # 모델 예측값
        pred = model(data.type(torch.FloatTensor).to(device), h0)

        # 손실 계산
        loss = nn.MSELoss()(pred, label.type(torch.FloatTensor).to(device))
        loss.backward()  # 오차 역전파
        optim.step()     # 최적화 진행(파라미터 업데이트)

        iterator.set_description(f"epoch{epoch} loss:{loss.item()}")

# 모델 저장
torch.save(model.state_dict(), "./rnn.pth")

# 모델 성능 평가
loader = DataLoader(dataset, batch_size=1)  # 예측값을 위한 데이터 로더(예측할 땐 하나씩 천천히 정확하게 확인하고 싶기 때문에 batch_size = 1)

preds = []  # 예측값 리스트
total_loss = 0

with torch.no_grad():
    # 모델 가중치 불러오기
    model.load_state_dict(torch.load("./rnn.pth", map_location=device))
    # 모델을 학습시킨 후 가장 잘 배운 상태(가중치)를 저장해놓음.
    # 나중에 예측을 할 때는 그 잘 배운 상태를 다시 불러와야 똑똑한 모델이 종가를 정확하게 예측할 수 있다.

    for data, label in loader:
        h0 = torch.zeros(5, data.shape[0], 8).to(device)  # 초기 은닉 상태 정의(layer(5), 배치크기(32), 기억크기(8))

        # 예측값 출력
        pred = model(data.type(torch.FloatTensor).to(device), h0)
        preds.append(pred.item())  # 예측값 추가

        # 손실 계산
        loss = nn.MSELoss()(pred, label.type(torch.FloatTensor).to(device))
        total_loss += loss / len(loader)

# 평균 손실 출력
print(f"Total Loss: {total_loss.item()}")

# 예측값 그래프 출력
plt.plot(preds, label="Prediction")
plt.plot(dataset.label[30:], label="Actual")
plt.legend()
plt.show()
```
