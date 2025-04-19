## 1. 머신러닝(Machine Learning)과 딥러닝(Deep Learning)의 차이

| 구분 | 머신러닝 | 딥러닝 |
|------|-----------|------------|
| 정의 | 데이터에서 패턴을 찾아 예측하는 알고리즘 | 신경망 구조를 이용한 머신러닝의 한 분야 |
| 입력 데이터 | 수작업 특징 추출 필요 | 원시 데이터를 바로 사용 가능 |
| 모델 복잡도 | 낮음~중간 | 매우 높음 (수많은 파라미터) |
| 학습 속도 | 빠름 | 느림 (많은 연산량 필요) |
| 대표 알고리즘 | SVM, KNN, 의사결정나무 등 | CNN, RNN, LSTM, Transformer 등 |

target은 prediction이 맞춰야 할 정답이고 epoch은 학습의 횟수를 가리킨다.

<br><br><br>

---

## 2. 머신러닝 주요 알고리즘 분류

### 지도학습(Supervised Learning)
- 레이블(정답)이 있는 데이터를 학습
- 예시: 분류(Classification), 회귀(Regression)
  - 스팸 이메일 분류, 집값 예측

  
**주요 알고리즘:**
- 선형 회귀(Linear Regression)
- 의사결정나무(Decision Tree)
- 랜덤 포레스트(Random Forest)
- SVM(Support Vector Machine)

<br><br>

### 비지도학습(Unsupervised Learning)
- 레이블이 없는 데이터를 군집화 또는 특징 추출(정답이 없는 데이터에서 패턴을 찾기)   
- ex.고객 군집화, 데이터 차원 축소

**주요 알고리즘:**
- K-Means
- PCA (주성분 분석)
- 계층적 군집(Hierarchical Clustering)

<br><br>

### 강화학습(Reinforcement Learning)
- 환경과 상호작용하며 보상을 최대화하도록 학습하는 방식
- **특징**: 행동 → 보상 → 정책 학습 방식

**주요 개념:**
- Agent (행위자)
- Environment (환경)
- Action (행동)
- Reward (보상)

<br><br><br>

---

## 3. 딥러닝 기본 용어

| 용어 | 설명 |
|------|------|
| Epoch | 전체 학습 데이터를 몇 번 반복 학습할 것인지 |
| Batch Size | 한 번에 학습하는 데이터 샘플 수 |
| Iteration | 한 Epoch 동안 반복 횟수 (데이터 수 / 배치 크기) |
| Loss | 예측값과 실제값의 차이 (오차) |
| Optimizer | 모델 파라미터를 갱신해 손실을 최소화하는 도구 (Adam, SGD 등) |
| Learning Rate | 가중치를 얼마나 크게 조정할지 결정하는 값 |
| Overfitting | 학습 데이터에 과도하게 적합되어 일반화되지 않는 문제 |
| Underfitting | 학습 데이터조차 잘 예측하지 못하는 상태 |

<br><br><br>

---

## 4. 손실 함수(Loss Function)

| 종류 | 설명 |
|------|------|
| MSE (Mean Squared Error) | 평균 제곱 오차, 회귀에서 주로 사용 |
| MAE (Mean Absolute Error) | 평균 절대 오차, 이상치에 덜 민감 |
| Cross Entropy | 분류 문제에서 많이 사용되는 손실 함수 |

<br><br><br>

---

## 5. 활성화 함수(Activation Function)

| 함수 | 설명 |
|------|------|
| ReLU | 0보다 작으면 0, 크면 그대로 출력 (가장 많이 사용됨) |
| Sigmoid | 출력이 0~1 사이 (이진 분류에 사용) |
| Tanh | 출력이 -1~1 사이 |
| Softmax | 출력값을 확률로 변환 (다중 클래스 분류) |

```py
model = Sequential([
    # 1차원 feature map 생성
    Conv1D(filters=32, kernel_size=5,
           padding="causal",
           activation="relu",
           input_shape=[WINDOW_SIZE, 1]),
    # LSTM
    LSTM(16, activation='tanh'),
    Dense(16, activation="relu"),
    Dense(1),
])
```

[참고 코드](https://github.com/haedal-uni/analysis/blob/main/work/2025-04/250408_LSTM%EC%9D%84_%ED%99%9C%EC%9A%A9%ED%95%9C_%EC%A3%BC%EA%B0%80_%EC%98%88%EC%B8%A1_%EB%AA%A8%EB%8D%B8.ipynb)

<br><br><br>

---

## 6. 신경망 기본 구조

### 1) 퍼셉트론(Perceptron)
- 인공 뉴런 한 개를 모델링한 것
- 입력 * 가중치 → 총합 → 활성화 함수 → 출력

<br><br>

### 2) MLP (다층 퍼셉트론)
```py
self.model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size)
)
```

<br><br><br>

---

## 7. 모델 훈련 기본 코드 흐름
```py
for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

<br><br><br>

---

## 8. 자주 등장하는 용어 정리

| 용어 | 설명 |
|------|------|
| Feature | 입력 변수 (설명 변수) |
| Label | 출력 변수 (목표 값) |
| Weight | 각 입력에 곱해지는 가중치 |
| Bias | 출력에 더해지는 값 (편향) |
| Backpropagation | 오차를 기준으로 가중치를 조정하는 방법 |
| Gradient Descent | 손실 최소화를 위한 최적화 알고리즘 |

<br><br><br>

---

## 9. 데이터 전처리 

| 개념 | 설명 |
|------|------|
| 정규화(Normalization) | 데이터의 범위를 [0,1]로 조정 |
| 표준화(Standardization) | 평균 0, 표준편차 1로 조정 |
| One-hot encoding | 범주형 데이터를 이진 벡터로 변환 |
| 결측값 처리 | 평균/중앙값 대체, 제거, 예측 등 |

<br><br><br>

---

## 10. 추천 학습 순서

1. 머신러닝의 개념과 지도/비지도 학습 이해
2. 대표적인 모델(SVM, KNN, Decision Tree) 실습
3. 오차와 평가 지표 이해 (MSE, MAE, Accuracy 등)
4. 딥러닝 구조 (Perceptron → MLP → CNN → RNN) 익히기
5. PyTorch 또는 TensorFlow 기본 문법 연습
6. 학습 흐름, 손실 함수, 옵티마이저 역할 파악
7. 데이터 전처리 및 모델 성능 비교 실습

<br><br><br>

---

## 11. 데이터 분할 전략 및 예제 코드

### 11.1 `train_test_split()` 사용
```py
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
*20%를 테스트 데이터셋으로 사용

또는 Numpy로 직접 분할하거나 StratifiedKFold/GroupKFold 사용 가능

<br><br>

### 11.2 Numpy로 직접 나누기 (학습/검증/테스트)
```py
import numpy as np

# 데이터 섞기
indices = np.arange(len(X))
np.random.seed(42)
np.random.shuffle(indices)

# 비율 설정 (60% 학습, 20% 검증, 20% 테스트)
train_end = int(0.6 * len(X))
val_end = int(0.8 * len(X))

X_train, y_train = X[indices[:train_end]], y[indices[:train_end]]
X_val, y_val = X[indices[train_end:val_end]], y[indices[train_end:val_end]]
X_test, y_test = X[indices[val_end:]], y[indices[val_end:]]
```

<br><br>

### 11.3 `StratifiedKFold` (계층적 분할)
```py
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```

<br><br>

### 11.4 `GroupKFold` (그룹 기반 분할)
```py
from sklearn.model_selection import GroupKFold

groups = ...  # 예: 사용자 ID 등

gkf = GroupKFold(n_splits=5)
for train_idx, test_idx in gkf.split(X, y, groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

<br><br><br>

---

## 12. 대표적인 이미지 데이터셋

### 12.1 MNIST (숫자 손글씨)
```py
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```
`mnist.load_data()` : MNIST 데이터셋은 60000개의 훈련 데이터와 10000개의 테스트 데이터로 고정되어 제공

<br><br>

### 12.2 Fashion-MNIST (의류 이미지)
```py
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
```

<br><br>

### 12.3 CIFAR-10 (컬러 이미지)
```py
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

<br><br>

### 12.4 PyTorch용 TensorDataset & DataLoader 예시
```py
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
```

<br><br><br>

---


## 13. 하이퍼파라미터 튜닝 (Hyperparameter Tuning)

**하이퍼파라미터란?**
- 학습 전에 사용자가 지정하는 값 (예: learning rate, batch size, epoch 수)

**튜닝 방법:**
- **Grid Search**: 미리 정의한 값들을 조합으로 탐색
- **Random Search**: 무작위 조합으로 탐색
- **Optuna / Hyperopt**: 베이지안 최적화 기반 자동화 도구

```py
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

params = {'n_estimators': [100, 200], 'max_depth': [5, 10]}
grid = GridSearchCV(RandomForestClassifier(), param_grid=params, cv=3)
grid.fit(X_train, y_train)
print(grid.best_params_)
```

---

## 14. 파인 튜닝 (Fine-Tuning)

**정의:**
- 이미 학습된 모델(pretrained model)의 일부 또는 전체를 미세 조정하여 새로운 데이터셋에 맞추는 방법

**예시:**
- ImageNet으로 학습된 VGG16 모델의 가중치를 가져와, 마지막 layer만 교체해 재학습

```py
from tensorflow.keras.applications import VGG16

base_model = VGG16(weights='imagenet', include_top=False)
for layer in base_model.layers:
    layer.trainable = False  # 기존 가중치 고정

# 새롭게 분류 레이어 추가
model = Sequential([
    base_model,
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
```

---

## 15. 윈도우 슬라이싱 (Windowing)과 롤링 (Rolling)

### 15.1 윈도우(Windowing)
- 시계열 데이터를 일정한 크기의 구간(window)으로 잘라 입력으로 사용
- 예: 과거 7일 데이터를 이용해 오늘 값을 예측

```py
WINDOW_SIZE = 7
X, y = [], []
for i in range(len(data) - WINDOW_SIZE):
    X.append(data[i:i+WINDOW_SIZE])
    y.append(data[i+WINDOW_SIZE])
```

### 15.2 롤링(Rolling)
- 시간축을 따라 이동하며 통계값(평균, 표준편차 등) 계산

```py
import pandas as pd

df['rolling_mean'] = df['price'].rolling(window=7).mean()
df['rolling_std'] = df['price'].rolling(window=7).std()
```

---

## 16. 딥러닝 모델 예제 (Conv1D + LSTM)

```py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense

model = Sequential([
    Conv1D(filters=32, kernel_size=5, padding="causal", activation="relu", input_shape=[WINDOW_SIZE, 1]),
    LSTM(16, activation='tanh'),
    Dense(16, activation="relu"),
    Dense(1),
])
```

### 코드 해석:
- **Conv1D**: 시계열의 로컬 패턴(변화량)을 먼저 추출
- **LSTM**: 시간의 흐름을 반영해 기억하고 예측
- **Dense**: 최종 출력값 계산 (회귀 문제에서 사용)

---

## 17. PyTorch MLP 코드

```py
import torch.nn as nn

self.model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size)
)
```

- **input_size**: 입력 벡터 크기
- **hidden_size**: 은닉층 노드 수
- **output_size**: 예측값 크기 (1이면 회귀)

---

## 18. 모델 학습 루프

```py
for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

- **loss.backward()**: 오차를 기준으로 기울기 계산
- **optimizer.step()**: 파라미터 업데이트
- **optimizer.zero_grad()**: 이전 기울기 초기화


---

## 기타 정리

- **과적합(Overfitting)** : 모델이 학습 데이터에만 너무 잘 맞추고 새로운 데이터에는 성능이 낮은 상태
- **손실(Loss)** : 정답과 예측값 간의 거리 또는 차이를 수치로 표현한 값 (낮을수록 좋음)
- **지도학습** : 정답이 있는 데이터를 이용해 학습하는 방법. 분류/회귀 문제에서 사용
- **비지도학습** : 정답 없이 데이터의 구조나 패턴을 학습 (군집화, 차원 축소 등)
- **강화학습** : 보상을 기반으로 최적의 행동을 학습하는 방법. 게임, 로봇 등에 많이 사용


---

## 13. 하이퍼파라미터 튜닝 (Hyperparameter Tuning)

**하이퍼파라미터란?**

모델이 학습되기 전에 사람이 직접 설정해주는 값입니다.

예: 학습률(Learning Rate), 배치 크기(Batch Size), 은닉층 크기, 에포크 수 등

| 하이퍼파라미터 | 설명 |
| --- | --- |
| Learning Rate | 가중치를 얼마나 크게 업데이트할지 결정 |
| Epochs | 전체 데이터를 몇 번 반복할 것인지 |
| Batch Size | 한 번에 학습할 데이터 샘플 수 |
| Hidden Units | 은닉층 뉴런 수 (모델의 복잡도 결정) |
| Dropout Rate | 과적합 방지를 위한 무작위 노드 비활성화 비율 |

### 13.1 그리드 서치 (GridSearchCV)
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
grid_search.fit(X_train, y_train)

print("최적 파라미터:", grid_search.best_params_)

```

- 여러 하이퍼파라미터 조합을 시도해보고 가장 좋은 성능을 찾음

<br>

### 13.2 랜덤 서치 (RandomizedSearchCV)

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [None, 10, 20, 30]
}

random_search = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_dist, n_iter=10, cv=3)
random_search.fit(X_train, y_train)

print("최적 파라미터:", random_search.best_params_)
```

- GridSearch보다 빠르게 튜닝 가능 (일부 조합만 시도)

<br><br><br>

## 14. 모델 평가 지표

| 평가 지표 | 설명 | 사용 예시 |
| --- | --- | --- |
| Accuracy | 전체 예측 중 맞춘 비율 | 분류(Classification) |
| Precision | 예측 중 진짜 정답인 비율 | 스팸 필터에서 중요 |
| Recall | 실제 정답 중 얼마나 맞췄는지 | 암 진단 등 민감한 경우 |
| F1 Score | 정밀도와 재현율의 조화 평균 | 불균형 데이터 |
| MSE | 예측값과 실제값의 제곱 평균 | 회귀(Regression) |
| MAE | 절대 오차 평균 | 회귀, 이상치에 덜 민감 |

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"정확도: {accuracy}, 정밀도: {precision}, 재현율: {recall}, F1: {f1}")

```

<br><br><br>

## 15. PyTorch 기본 학습 코드 템플릿

### 15.1 모델 정의

```python
import torch
import torch.nn as nn

# 다층 퍼셉트론 (MLP)
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

```

### 15.2 학습/검증 코드 흐름

```python
model = MLPModel(input_dim=10, hidden_dim=32, output_dim=1)
criterion = nn.MSELoss()  # 손실 함수
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):  # 총 50번 반복
    model.train()  # 학습 모드 설정
    for X_batch, y_batch in train_loader:
        y_pred = model(X_batch)  # 예측
        loss = criterion(y_pred, y_batch)  # 손실 계산
        optimizer.zero_grad()  # 기존 기울기 초기화
        loss.backward()  # 역전파
        optimizer.step()  # 파라미터 업데이트

    print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")

```

> 💡 train_loader는 학습 데이터를 배치 단위로 묶어주는 역할

<br><br><br>

---

## 16. 딥러닝 주요 구조 요약

| 구조 | 설명 | 사용 예시 |
| --- | --- | --- |
| MLP | 입력 → 은닉층 → 출력층 | 기초 회귀, 분류 문제 |
| CNN | 이미지 처리, 필터 활용 | 이미지 분류 (MNIST 등) |
| RNN | 시계열 데이터 순서 유지 | 주가 예측, 자연어 처리 |
| LSTM | 장기 기억 유지가 가능한 RNN | 뉴스 감성 분석 |
| GRU | LSTM보다 구조 단순한 RNN | 시계열 예측 |
| Transformer | 병렬처리 강점, 어텐션 기반 | 번역, 챗봇, BERT |

<br><br><br>

---

## 17. 시계열 예측 딥러닝 모델 예시 (LSTM)

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 시계열 예측을 위한 LSTM 모델 구성
model = Sequential([
    LSTM(50, activation='tanh', input_shape=(10, 1)),  # 10일치 데이터를 입력
    Dense(1)  # 다음 날 값을 예측
])

model.compile(loss='mse', optimizer='adam')
model.summary()
```

`input_shape=(10, 1)`은 10일치 데이터를 시계열 순서대로 1차원으로 입력한다는 뜻입니다.

## 18. 용어 시각 예시 (한눈에 이해)

```
[Feature]   [Weight]    [Sum]   [Activation]   [Output]
   x1     →    w1     →    +     →   ReLU     →    y1
   x2     →    w2     →    +     →   ReLU     →    y2
```

- 각 입력 x에 가중치 w를 곱하고, 합을 구해 활성화 함수로 전달 → 출력 생성

## 19. 마무리 요약

- **머신러닝**은 규칙을 스스로 학습하는 알고리즘, **딥러닝**은 신경망 기반의 복잡한 모델
- 지도/비지도/강화학습으로 구분
- 모델 구조, 손실 함수, 옵티마이저, 평가 지표 등은 실무의 핵심
- PyTorch와 TensorFlow는 딥러닝을 구현할 수 있는 대표 프레임워크
- 하이퍼파라미터 튜닝을 통해 성능 개선 가능
- 전처리 → 학습 → 평가 → 튜닝 → 배포 순으로 모델링 진행
