## 1. 머신러닝(Machine Learning)과 딥러닝(Deep Learning)의 차이

| 구분 | 머신러닝 | 딥러닝 |
|------|-----------|------------|
| 정의 | 데이터에서 패턴을 찾아 예측하는 알고리즘 | 신경망 구조를 이용한 머신러닝의 한 분야 |
| 입력 데이터 | 수작업 특징 추출 필요 | 원시 데이터를 바로 사용 가능 |
| 모델 복잡도 | 낮음~중간 | 매우 높음 (수많은 파라미터) |
| 학습 속도 | 빠름 | 느림 (많은 연산량 필요) |
| 대표 알고리즘 | SVM, KNN, 의사결정나무 등 | CNN, RNN, LSTM, Transformer 등 |

<br><br><br>

---

## 2. 머신러닝 주요 알고리즘 분류

### 지도학습(Supervised Learning)
- 레이블(정답)이 있는 데이터를 학습
- 예시: 분류(Classification), 회귀(Regression)

**주요 알고리즘:**
- 선형 회귀(Linear Regression)
- 의사결정나무(Decision Tree)
- 랜덤 포레스트(Random Forest)
- SVM(Support Vector Machine)

<br><br>

### 비지도학습(Unsupervised Learning)
- 레이블이 없는 데이터를 군집화 또는 특징 추출

**주요 알고리즘:**
- K-Means
- PCA (주성분 분석)
- 계층적 군집(Hierarchical Clustering)

<br><br>

### 강화학습(Reinforcement Learning)
- 환경과 상호작용하며 보상을 최대화하도록 학습하는 방식

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

## 9. 데이터 전처리 필수 개념

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
20%를 테스트 데이터셋으로 사용

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

## 기타 정리

- **과적합(Overfitting)** : 모델이 학습 데이터에만 너무 잘 맞추고 새로운 데이터에는 성능이 낮은 상태
- **손실(Loss)** : 정답과 예측값 간의 거리 또는 차이를 수치로 표현한 값 (낮을수록 좋음)
- **지도학습** : 정답이 있는 데이터를 이용해 학습하는 방법. 분류/회귀 문제에서 사용
- **비지도학습** : 정답 없이 데이터의 구조나 패턴을 학습 (군집화, 차원 축소 등)
- **강화학습** : 보상을 기반으로 최적의 행동을 학습하는 방법. 게임, 로봇 등에 많이 사용

