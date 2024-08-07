## [혼공머신]2-2 -추가학습(데이터 전처리)

### 데이터 전처리 기술

표준점수 외에도 여러 가지 데이터 전처리 기술이 있습니다. 주요한 몇 가지를 소개하고 예시 코드를 함께 설명하겠습니다.

#### 1. Min-Max 스케일링
Min-Max 스케일링은 데이터를 최소값과 최대값을 사용하여 0과 1 사이의 값으로 변환하는 방법입니다.

```python
from sklearn.preprocessing import MinMaxScaler

# 데이터 생성
X, y = np.random.rand(100, 2), np.random.randint(0, 2, 100)

# Min-Max 스케일링
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print("Original Data:", X[:5])
print("Scaled Data:", X_scaled[:5])
```

#### 2. 로버스트 스케일링
로버스트 스케일링은 중간값과 사분위수를 사용하여 스케일링합니다. 이는 이상치에 덜 민감하게 작용합니다.

```python
from sklearn.preprocessing import RobustScaler

# 로버스트 스케일링
scaler = RobustScaler()
X_robust_scaled = scaler.fit_transform(X)

print("Original Data:", X[:5])
print("Robust Scaled Data:", X_robust_scaled[:5])
```

#### 3. 정규화 (Normalization)
정규화는 각 샘플의 길이를 1로 만들어줍니다. 이는 주로 텍스트 데이터나 유클리드 거리를 사용하는 알고리즘에서 사용됩니다.

```python
from sklearn.preprocessing import Normalizer

# 정규화
normalizer = Normalizer()
X_normalized = normalizer.fit_transform(X)

print("Original Data:", X[:5])
print("Normalized Data:", X_normalized[:5])
```

#### 4. 원-핫 인코딩 (One-Hot Encoding)
카테고리형 데이터를 이진 벡터로 변환하는 방법입니다.

```python
from sklearn.preprocessing import OneHotEncoder

# 카테고리 데이터 생성
categories = np.array(['apple', 'banana', 'cherry', 'date', 'apple', 'banana'])
categories = categories.reshape(-1, 1)

# 원-핫 인코딩
encoder = OneHotEncoder()
categories_encoded = encoder.fit_transform(categories).toarray()

print("Original Categories:", categories.ravel())
print("One-Hot Encoded Data:", categories_encoded)
```

### 예시 코드: 여러 전처리 기술 적용
다양한 전처리 기술을 적용한 종합 예시입니다.

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer, OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 데이터 생성
X, y = np.random.rand(100, 2), np.random.randint(0, 2, 100)
categories = np.random.choice(['cat', 'dog', 'mouse'], size=100).reshape(-1, 1)

# 훈련 세트와 테스트 세트로 분리
X_train, X_test, y_train, y_test, categories_train, categories_test = train_test_split(X, y, categories, test_size=0.25, stratify=y)

# Min-Max 스케일링
min_max_scaler = MinMaxScaler()
X_train_min_max_scaled = min_max_scaler.fit_transform(X_train)
X_test_min_max_scaled = min_max_scaler.transform(X_test)

# 표준점수 스케일링
standard_scaler = StandardScaler()
X_train_standard_scaled = standard_scaler.fit_transform(X_train)
X_test_standard_scaled = standard_scaler.transform(X_test)

# 로버스트 스케일링
robust_scaler = RobustScaler()
X_train_robust_scaled = robust_scaler.fit_transform(X_train)
X_test_robust_scaled = robust_scaler.transform(X_test)

# 정규화
normalizer = Normalizer()
X_train_normalized = normalizer.fit_transform(X_train)
X_test_normalized = normalizer.transform(X_test)

# 원-핫 인코딩
one_hot_encoder = OneHotEncoder()
categories_train_encoded = one_hot_encoder.fit_transform(categories_train).toarray()
categories_test_encoded = one_hot_encoder.transform(categories_test).toarray()

# 시각화
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], label='Original Data')
plt.title('Original Data')

plt.subplot(2, 3, 2)
plt.scatter(X_train_min_max_scaled[:, 0], X_train_min_max_scaled[:, 1], label='Min-Max Scaled', color='g')
plt.title('Min-Max Scaled Data')

plt.subplot(2, 3, 3)
plt.scatter(X_train_standard_scaled[:, 0], X_train_standard_scaled[:, 1], label='Standard Scaled', color='r')
plt.title('Standard Scaled Data')

plt.subplot(2, 3, 4)
plt.scatter(X_train_robust_scaled[:, 0], X_train_robust_scaled[:, 1], label='Robust Scaled', color='m')
plt.title('Robust Scaled Data')

plt.subplot(2, 3, 5)
plt.scatter(X_train_normalized[:, 0], X_train_normalized[:, 1], label='Normalized', color='c')
plt.title('Normalized Data')

plt.tight_layout()
plt.show()

print("Original Categories:", categories_train.ravel()[:5])
print("One-Hot Encoded Categories:\n", categories_train_encoded[:5])
```

### 요약
- **Min-Max 스케일링**: 데이터를 0과 1 사이로 변환.
- **로버스트 스케일링**: 중간값과 사분위수를 사용하여 스케일링.
- **정규화**: 각 샘플의 길이를 1로 만들어줌.
- **원-핫 인코딩**: 카테고리형 데이터를 이진 벡터로 변환.

이 예제 코드를 통해 다양한 전처리 기술을 학습하고 적용하는 방법을 익힐 수 있습니다. 이를 통해 머신러닝 모델의 성능을 더욱 향상시킬 수 있습니다.