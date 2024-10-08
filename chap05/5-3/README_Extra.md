## 기타 앙상블학습

앙상블 학습에는 다양한 방법들이 있으며, 여기서는 그 중 일부를 소개하겠습니다. 대표적인 앙상블 방법으로 배깅(Bagging), 부스팅(Boosting), 스태킹(Stacking), 그리고 배깅의 일종인 랜덤 서브스페이스(Random Subspace)를 설명하고, 예시 코드를 제공합니다.

### 배깅 (Bagging)

배깅은 같은 모델을 여러 개 학습시키고, 데이터의 부분 집합을 사용하여 각각의 모델을 학습시킨 후, 예측 시 이들의 평균 또는 투표 결과를 사용하는 방법입니다. 랜덤 포레스트가 대표적인 배깅 기법입니다.

#### 배깅 예제 코드

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_validate
import pandas as pd

# 데이터 로드 및 전처리
wine = pd.read_csv('https://bit.ly/wine_csv_data')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

# 배깅 모델 학습
bagging = BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, random_state=42)
scores = cross_validate(bagging, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

bagging.fit(train_input, train_target)
print(bagging.score(test_input, test_target))
```

### 부스팅 (Boosting)

부스팅은 약한 학습기(Weak Learner)를 순차적으로 학습시키고, 이전 모델의 오류를 수정해 나가는 방식입니다. 그레이디언트 부스팅, XGBoost, LightGBM 등이 여기에 속합니다.

#### 부스팅 예제 코드

그레이디언트 부스팅은 앞서 다룬 예시를 참조하십시오.

### 스태킹 (Stacking)

스태킹은 여러 모델을 학습시키고, 이들의 예측 결과를 새로운 모델의 입력으로 사용하여 최종 예측을 수행하는 방법입니다. 다양한 모델의 조합을 통해 성능을 향상시킬 수 있습니다.

#### 스태킹 예제 코드

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_validate
import pandas as pd

# 데이터 로드 및 전처리
wine = pd.read_csv('https://bit.ly/wine_csv_data')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

# 기본 모델 정의
estimators = [
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('svc', SVC(random_state=42))
]

# 스태킹 모델 학습
stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
scores = cross_validate(stacking, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

stacking.fit(train_input, train_target)
print(stacking.score(test_input, test_target))
```

### 랜덤 서브스페이스 (Random Subspace)

랜덤 서브스페이스는 배깅의 변형으로, 각 모델을 학습할 때마다 특성 공간의 무작위 부분 집합을 사용합니다. 이를 통해 모델의 다양성을 높입니다.

#### 랜덤 서브스페이스 예제 코드

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_validate
import pandas as pd

# 데이터 로드 및 전처리
wine = pd.read_csv('https://bit.ly/wine_csv_data')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

# 랜덤 서브스페이스 모델 학습
random_subspace = BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, max_features=2, random_state=42)
scores = cross_validate(random_subspace, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

random_subspace.fit(train_input, train_target)
print(random_subspace.score(test_input, test_target))
```

이와 같이 다양한 앙상블 기법을 활용하면 머신러닝 모델의 성능을 크게 향상시킬 수 있습니다. 각 기법의 특징과 사용 사례를 이해하고, 문제에 맞는 방법을 선택하여 적용하는 것이 중요합니다.