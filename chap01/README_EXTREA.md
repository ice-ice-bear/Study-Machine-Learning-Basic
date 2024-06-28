# 머신러닝 심화 학습 자료

## 추가 학습 요소와 모델 설명

### 정확도와 정밀도의 차이
정확도와 정밀도는 머신러닝 모델의 성능을 평가하는 데 중요한 지표입니다.

- **정확도(Accuracy)**: 전체 예측 중에서 정확하게 맞춘 예측의 비율입니다.

  [ \text{정확도} = \frac{\text{정확히 맞춘 개수}}{\text{전체 예측 개수}} \]

- **정밀도(Precision)**: 양성으로 예측한 것들 중에서 실제 양성인 것의 비율입니다.
\[ \text{정밀도} = \frac{\text{정확히 맞춘 양성}}{\text{예측한 양성}} \]

예를 들어, 암 진단 모델에서 100개의 샘플 중 90개를 양성으로 예측했고, 그 중 80개가 실제로 양성이라면:
\[ \text{정밀도} = \frac{80}{90} \approx 0.89 \]

### k-최근접 이웃 알고리즘 외에 활용해 볼 수 있는 분류 알고리즘

#### 1. 로지스틱 회귀 (Logistic Regression)
로지스틱 회귀는 선형 회귀를 확장한 알고리즘으로, 특정 클래스에 속할 확률을 예측합니다. 선형 회귀와 달리, 로지스틱 회귀는 종속 변수가 이진 분류 문제에 적합한 S자 형태의 로지스틱 함수(시그모이드 함수)를 사용합니다.

**장점:**
- 구현이 간단하고 이해하기 쉬움
- 예측 확률을 제공
- 계산 효율성이 높음

**단점:**
- 비선형 문제를 해결할 수 없음
- 다중 공선성 문제에 민감함

```python
from sklearn.linear_model import LogisticRegression

# 모델 생성
lr = LogisticRegression()

# 모델 훈련
lr.fit(fish_data, fish_target)

# 새로운 데이터 예측
new_data = [[30, 600]]
print(lr.predict(new_data))

# 모델 성능 평가
score = lr.score(fish_data, fish_target)
print(f"모델의 정확도: {score}")
```

#### 2. 서포트 벡터 머신 (Support Vector Machine, SVM)
SVM은 데이터를 분류하기 위해 최적의 경계를 찾는 알고리즘입니다. 이 경계는 클래스 간의 여백을 최대화하는 초평면(hyperplane)입니다. SVM은 선형 및 비선형 분류에 모두 사용할 수 있으며, 비선형 분류의 경우 커널 트릭(kernel trick)을 사용합니다.

**장점:**
- 고차원 데이터에서 잘 작동
- 분류 경계가 명확함
- 커널 트릭을 사용하여 비선형 분류 가능

**단점:**
- 대규모 데이터셋에서는 비효율적일 수 있음
- 최적의 커널과 매개변수를 선택해야 함

```python
from sklearn.svm import SVC

# 모델 생성
svm = SVC()

# 모델 훈련
svm.fit(fish_data, fish_target)

# 새로운 데이터 예측
new_data = [[30, 600]]
print(svm.predict(new_data))

# 모델 성능 평가
score = svm.score(fish_data, fish_target)
print(f"모델의 정확도: {score}")
```

#### 3. 결정 트리 (Decision Tree)
결정 트리는 데이터의 특성에 따라 의사결정을 나무 구조로 모델링하는 알고리즘입니다. 각 내부 노드는 특성의 조건을 나타내며, 각 잎 노드는 클래스 레이블을 나타냅니다.

**장점:**
- 데이터 전처리가 거의 필요 없음
- 시각화가 용이하여 해석이 쉬움
- 다중 클래스 분류에 적합

**단점:**
- 과적합(overfitting) 되기 쉬움
- 작은 변화에도 민감하여 불안정할 수 있음

```python
from sklearn.tree import DecisionTreeClassifier

# 모델 생성
dt = DecisionTreeClassifier()

# 모델 훈련
dt.fit(fish_data, fish_target)

# 새로운 데이터 예측
new_data = [[30, 600]]
print(dt.predict(new_data))

# 모델 성능 평가
score = dt.score(fish_data, fish_target)
print(f"모델의 정확도: {score}")
```

#### 4. 랜덤 포레스트 (Random Forest)
랜덤 포레스트는 여러 개의 결정 트리를 앙상블하여 예측 성능을 향상시키는 알고리즘입니다. 각각의 결정 트리는 데이터의 서로 다른 부분집합을 사용하여 학습되며, 최종 예측은 모든 트리의 예측을 결합하여 결정됩니다.

**장점:**
- 과적합을 줄여줌
- 고성능 예측 모델을 생성
- 다양한 데이터 유형에 적합

**단점:**
- 해석이 어려움
- 훈련 시간이 오래 걸릴 수 있음

```python
from sklearn.ensemble import RandomForestClassifier

# 모델 생성
rf = RandomForestClassifier()

# 모델 훈련
rf.fit(fish_data, fish_target)

# 새로운 데이터 예측
new_data = [[30, 600]]
print(rf.predict(new_data))

# 모델 성능 평가
score = rf.score(fish_data, fish_target)
print(f"모델의 정확도: {score}")
```

### 혼동 행렬 (Confusion Matrix)
혼동 행렬은 모델의 성능을 자세히 평가할 수 있는 도구입니다. 각 행은 실제 클래스, 각 열은 예측된 클래스를 나타냅니다.

```python
from sklearn.metrics import confusion_matrix

# 예측값 계산
predictions = kn.predict(fish_data)

# 혼동 행렬 계산
cm = confusion_matrix(fish_target, predictions)
print(cm)
```

### F1-Score
F1-Score는 정밀도와 재현율의 조화 평균입니다. 정밀도와 재현율 간의 균형을 유지하는 데 유용합니다.

```python
from sklearn.metrics import f1_score

# F1-Score 계산
f1 = f1_score(fish_target, predictions)
print(f"F1-Score: {f1}")
```

## 결론
이 자료를 통해 머신러닝 모델의 성능을 다양한 측면에서 평가할 수 있습니다. 추가된 알고리즘 예제를 통해 k-최근접 이웃 알고리즘 외에도 다양한 분류 알고리즘을 적용해 볼 수 있습니다. 정밀도, 재현율, F1-Score 등의 지표를 이해하고 혼동 행렬을 분석하여 모델의 성능을 더욱 정밀하게 평가해 보세요.