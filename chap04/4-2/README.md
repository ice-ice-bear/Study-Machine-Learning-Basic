### 확률적 경사 하강법(SGD, Stochastic Gradient Descent)

확률적 경사 하강법(SGD)은 기계 학습에서 널리 사용되는 최적화 알고리즘입니다. 이 알고리즘은 대규모 데이터 셋에서도 빠르고 효율적으로 동작합니다. 아래는 확률적 경사 하강법에 대한 핵심 포인트와 상세한 설명입니다.

#### 핵심 포인트

- **확률적 경사 하강법**: 훈련 세트에서 샘플 하나씩 꺼내 손실 함수의 경사를 따라 최적의 모델을 찾는 알고리즘.
- **미니배치 경사 하강법**: 여러 개의 샘플을 동시에 사용하는 방법.
- **배치 경사 하강법**: 전체 샘플을 한 번에 사용하는 방법.
- **손실 함수**: SGD가 최적화할 대상. 이진 분류에는 로지스틱 회귀, 다중 분류에는 크로스엔트로피 손실 함수, 회귀 문제에는 평균 제곱 오차 손실 함수를 사용.
- **에포크**: 전체 샘플을 모두 사용하는 한 번의 반복. 수십에서 수백 번의 에포크를 반복.

#### 주요 패키지와 함수

- scikit-learn

  - ```
    SGDClassifier
    ```

    : 확률적 경사 하강법을 사용한 분류 모델.

    - `loss` 매개변수: 최적화할 손실 함수를 지정 (기본값은 'hinge' 손실 함수).
    - `penalty` 매개변수: 규제의 종류를 지정 (기본값은 L2 규제를 위한 'l2').
    - `alpha` 매개변수: 규제 강도 (기본값은 0.0001).
    - `max_iter` 매개변수: 에포크 횟수 지정 (기본값은 1000).
    - `tol` 매개변수: 반복을 멈출 조건 (기본값은 0.001).

  - ```
    SGDRegressor
    ```

    : 확률적 경사 하강법을 사용한 회귀 모델.

    - `loss` 매개변수: 손실 함수를 지정 (기본값은 'squared_loss').

### 1. 라이브러리 임포트 및 데이터 준비

먼저, 필요한 라이브러리를 임포트하고 데이터를 준비합니다.

```python
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 예시 데이터 준비
X, y = load_some_data()  # 적절한 데이터 로드 함수 사용
```

- `SGDClassifier`, `SGDRegressor`: 확률적 경사 하강법을 사용한 분류 및 회귀 모델을 생성하는 클래스.
- `StandardScaler`: 데이터 표준화를 위한 클래스.
- `train_test_split`: 데이터를 훈련 세트와 테스트 세트로 분할하기 위한 함수.
- `load_some_data()`: 실제로는 사용자가 로드하는 데이터셋을 가져오는 함수입니다.

### 2. 데이터 분할

데이터를 훈련 세트와 테스트 세트로 분할합니다.

```python
train_data, test_data, train_target, test_target = train_test_split(X, y, test_size=0.2, random_state=42)
```

- `train_test_split`: 데이터를 무작위로 섞어서 훈련 세트와 테스트 세트로 나눕니다.
  - `test_size=0.2`: 데이터의 20%를 테스트 세트로 사용합니다.
  - `random_state=42`: 결과 재현성을 위해 난수 시드를 설정합니다.

### 3. 데이터 스케일링

훈련 세트와 테스트 세트를 표준화합니다.

```python
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)
```

- `StandardScaler()`: 각 특징의 평균을 0, 분산을 1로 맞추어 표준화합니다.
- `fit_transform(train_data)`: 훈련 데이터의 평균과 분산을 계산하여 변환합니다.
- `transform(test_data)`: 훈련 데이터의 평균과 분산을 사용하여 테스트 데이터를 변환합니다.

### 4. SGDClassifier 사용 예제

로지스틱 회귀를 사용한 확률적 경사 하강법 분류 모델을 학습시키고 평가합니다.

```python
clf = SGDClassifier(loss='log', max_iter=1000, tol=0.001, random_state=42)
clf.fit(train_scaled, train_target)
print("훈련 세트 점수:", clf.score(train_scaled, train_target))
print("테스트 세트 점수:", clf.score(test_scaled, test_target))
```

- `SGDClassifier(loss='log', max_iter=1000, tol=0.001, random_state=42)`: SGDClassifier 객체를 생성합니다.
  - `loss='log'`: 로지스틱 회귀 손실 함수를 사용합니다.
  - `max_iter=1000`: 최대 1000번의 에포크 동안 학습합니다.
  - `tol=0.001`: 손실 함수가 이 값 이하로 감소하면 학습을 중지합니다.
  - `random_state=42`: 결과 재현성을 위해 난수 시드를 설정합니다.
- `fit(train_scaled, train_target)`: 훈련 데이터를 사용하여 모델을 학습시킵니다.
- `score(train_scaled, train_target)`: 훈련 세트에 대한 모델의 성능을 평가합니다.
- `score(test_scaled, test_target)`: 테스트 세트에 대한 모델의 성능을 평가합니다.

### 5. SGDRegressor 사용 예제

제곱 오차를 사용한 확률적 경사 하강법 회귀 모델을 학습시키고 평가합니다.

```python
reg = SGDRegressor(loss='squared_loss', max_iter=1000, tol=0.001, random_state=42)
reg.fit(train_scaled, train_target)
print("훈련 세트 점수:", reg.score(train_scaled, train_target))
print("테스트 세트 점수:", reg.score(test_scaled, test_target))
```

- `SGDRegressor(loss='squared_loss', max_iter=1000, tol=0.001, random_state=42)`: SGDRegressor 객체를 생성합니다.
  - `loss='squared_loss'`: 평균 제곱 오차 손실 함수를 사용합니다.
  - `max_iter=1000`: 최대 1000번의 에포크 동안 학습합니다.
  - `tol=0.001`: 손실 함수가 이 값 이하로 감소하면 학습을 중지합니다.
  - `random_state=42`: 결과 재현성을 위해 난수 시드를 설정합니다.
- `fit(train_scaled, train_target)`: 훈련 데이터를 사용하여 모델을 학습시킵니다.
- `score(train_scaled, train_target)`: 훈련 세트에 대한 모델의 성능을 평가합니다.
- `score(test_scaled, test_target)`: 테스트 세트에 대한 모델의 성능을 평가합니다.
