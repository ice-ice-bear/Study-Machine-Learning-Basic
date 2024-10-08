### 정규화 기법 (Regularization Techniques)

정규화는 머신러닝 모델의 과적합(overfitting)을 방지하고 일반화 성능을 향상시키기 위해 사용되는 기법입니다. 특히 로지스틱 회귀와 같은 선형 모델에서 중요한 역할을 합니다. 정규화는 모델의 복잡도를 줄여, 학습 데이터에 너무 치우치지 않도록 하는 데 도움을 줍니다. 주로 사용되는 정규화 기법에는 L1 정규화와 L2 정규화가 있습니다.

#### 1. L1 정규화 (Lasso Regularization)
L1 정규화는 가중치 벡터의 절대값 합을 패널티로 추가하는 방식입니다.

**수식:**
$$

\text{Loss function} = \text{original loss} + \lambda \sum_{j=1}^{p} |w_j|
$$
여기서 $ \lambda $는 정규화 강도를 제어하는 하이퍼파라미터이고,$ w_j $는 모델의 가중치입니다.

**특징:**

- L1 정규화는 일부 가중치를 정확히 0으로 만듭니다. 이는 변수 선택(feature selection) 효과를 가지며, 모델을 더 해석 가능하게 만듭니다.
- 희소 모델(sparse model)을 생성하는 데 유리합니다.

#### 2. L2 정규화 (Ridge Regularization)
L2 정규화는 가중치 벡터의 제곱 합을 패널티로 추가하는 방식입니다.

**수식:**
$$
\text{Loss function} = \text{original loss} + \lambda \sum_{j=1}^{p} w_j^2
$$
**특징:**
- 모든 가중치를 작게 만들어, 모델이 극단적인 가중치 값을 가지지 않도록 합니다.
- L2 정규화는 가중치를 0에 가깝게 만들지만 정확히 0으로 만들지는 않습니다.

#### 3. Elastic Net Regularization
Elastic Net은 L1 정규화와 L2 정규화를 결합한 방식입니다. 두 패널티를 모두 사용하는 모델입니다.

**수식:**
$$
\text{Loss function} = \text{original loss} + \lambda_1 \sum_{j=1}^{p} |w_j| + \lambda_2 \sum_{j=1}^{p} w_j^2
$$
**특징:**
- L1과 L2 정규화의 장점을 모두 취합니다.
- 변수 선택과 함께 모든 변수의 가중치를 적절히 줄여줍니다.

### 실습 예제

다음은 L1, L2, 그리고 Elastic Net 정규화를 적용한 로지스틱 회귀 모델의 예제입니다.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# 데이터셋 로드 및 분할
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# L1 정규화 적용 (라쏘 방식)
lr_l1 = LogisticRegression(penalty='l1', solver='saga', C=1.0, max_iter=10000)
lr_l1.fit(X_train_scaled, y_train)
print("L1 정규화 모델의 정확도:", lr_l1.score(X_test_scaled, y_test))

# L2 정규화 적용 (릿지 방식)
lr_l2 = LogisticRegression(penalty='l2', solver='lbfgs', C=1.0, max_iter=10000)
lr_l2.fit(X_train_scaled, y_train)
print("L2 정규화 모델의 정확도:", lr_l2.score(X_test_scaled, y_test))

# Elastic Net 정규화 적용
lr_en = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=1.0, max_iter=10000)
lr_en.fit(X_train_scaled, y_train)
print("Elastic Net 정규화 모델의 정확도:", lr_en.score(X_test_scaled, y_test))
```

이 코드는 Iris 데이터셋을 사용하여 L1, L2, Elastic Net 정규화를 각각 적용한 로지스틱 회귀 모델을 훈련시키고, 테스트 세트에 대한 모델의 정확도를 출력합니다.