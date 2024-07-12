### 머신러닝 실습 교육자료

#### 1. 데이터 준비
먼저, 사용할 데이터를 준비합니다. 여기서는 물고기의 길이와 무게를 포함한 데이터를 사용합니다.

```python
import numpy as np

perch_length = np.array([
    8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0,
    21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5,
    22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5,
    27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0,
    36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0,
    40.0, 42.0, 43.0, 43.0, 43.5, 44.0
])

perch_weight = np.array([
    5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0,
    110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0,
    130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0,
    197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0,
    514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0,
    820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0,
    1000.0, 1000.0
])
```

#### 2. 데이터 분할
훈련 데이터와 테스트 데이터를 나눕니다. 이를 통해 모델의 성능을 평가할 수 있습니다.

```python
from sklearn.model_selection import train_test_split

# 훈련 세트와 테스트 세트로 나눕니다
train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42)

# 훈련 세트와 테스트 세트를 2차원 배열로 바꿉니다
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
```

#### 3. 선형 회귀 모델
사이킷런의 `LinearRegression` 클래스를 사용하여 선형 회귀 모델을 학습합니다.

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
# 선형 회귀 모델 훈련
lr.fit(train_input, train_target)

# 50cm 농어에 대한 예측
print(lr.predict([[50]]))

# 회귀 계수와 절편 출력
print(lr.coef_, lr.intercept_)
```

#### 4. 모델 평가
훈련 데이터와 테스트 데이터에 대한 모델의 성능을 평가합니다.

```python
print(lr.score(train_input, train_target))
print(lr.score(test_input, test_target))
```

#### 5. 다항 회귀 모델
다항 회귀 모델을 학습하여 비선형 데이터를 모델링합니다.

```python
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))

lr = LinearRegression()
lr.fit(train_poly, train_target)

# 50cm 농어에 대한 예측
print(lr.predict([[50**2, 50]]))

# 다항 회귀 계수와 절편 출력
print(lr.coef_, lr.intercept_)

# 모델 평가
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))
```

#### 6. 시각화
데이터와 모델을 시각화하여 결과를 확인합니다.

```python
import matplotlib.pyplot as plt

# 훈련 세트의 산점도를 그립니다
plt.scatter(train_input, train_target)

# 15에서 50까지 1차 방정식 그래프를 그립니다
plt.plot([15, 50], [15*lr.coef_[1]+lr.intercept_, 50*lr.coef_[1]+lr.intercept_])

# 50cm 농어 데이터
plt.scatter(50, lr.predict([[50**2, 50]]), marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 구간별 직선을 그리기 위해 15에서 49까지 정수 배열을 만듭니다
point = np.arange(15, 50)

# 훈련 세트의 산점도를 그립니다
plt.scatter(train_input, train_target)

# 15에서 49까지 2차 방정식 그래프를 그립니다
plt.plot(point, lr.coef_[0]*point**2 + lr.coef_[1]*point + lr.intercept_)

# 50cm 농어 데이터
plt.scatter(50, lr.predict([[50**2, 50]]), marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

### 강의 요약
- **선형 회귀**는 특성과 타깃 사이의 관계를 선형 방정식으로 나타냅니다.
- **다항 회귀**는 다항식을 사용하여 비선형 관계를 모델링할 수 있습니다.
- `LinearRegression` 클래스는 선형 회귀 모델을 쉽게 구현할 수 있게 도와줍니다.
- 학습된 모델의 성능은 훈련 데이터와 테스트 데이터를 통해 평가할 수 있습니다.
- 모델의 시각화를 통해 데이터를 더 직관적으로 이해할 수 있습니다.

이 자료를 통해 선형 회귀와 다항 회귀의 기본 개념과 이를 실제 데이터에 적용하는 방법을 익힐 수 있습니다.