## 결정 트리의 정확도를 높이기 위한 방법

### 1. 가지치기 (Pruning)
가지치기는 트리의 불필요한 가지를 제거하여 과대적합을 방지하는 방법입니다.

- **사전 가지치기 (Pre-pruning)**:
  - `max_depth`: 트리의 최대 깊이를 제한합니다.
  - `min_samples_split`: 노드를 분할하기 위한 최소 샘플 수를 지정합니다.
  - `min_samples_leaf`: 리프 노드가 가져야 하는 최소 샘플 수를 지정합니다.
  - `max_leaf_nodes`: 리프 노드의 최대 개수를 제한합니다.

```python
model = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42)
model.fit(X_train, y_train)
```

- **사후 가지치기 (Post-pruning)**:
  - 사후 가지치기는 트리가 생성된 후 가지를 잘라내는 방법입니다. `scikit-learn`에서 직접 제공하지는 않지만, 트리를 생성한 후 수동으로 가지를 잘라낼 수 있습니다.

### 2. 앙상블 방법 (Ensemble Methods)
여러 개의 결정 트리를 사용하여 예측을 개선하는 방법입니다.

- **랜덤 포레스트 (Random Forest)**:
  - 여러 결정 트리를 학습시키고, 그 결과를 앙상블하여 최종 예측을 도출합니다.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

- **배깅 (Bagging)**:
  - 데이터의 서브셋을 생성하여 각 서브셋에 대해 결정 트리를 학습시킵니다. 그 결과를 평균화하거나 다수결로 예측합니다.

```python
from sklearn.ensemble import BaggingClassifier

model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

- **부스팅 (Boosting)**:
  - 이전 트리의 오차를 보정하는 방식으로 순차적으로 트리를 학습시킵니다. 대표적으로 `AdaBoost`와 `Gradient Boosting`이 있습니다.

```python
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### 3. 특성 공학 (Feature Engineering)
데이터의 특성을 개선하여 모델의 성능을 높이는 방법입니다.

- **특성 선택 (Feature Selection)**:
  - 중요한 특성만을 선택하여 모델을 단순화하고 성능을 향상시킵니다.

- **특성 생성 (Feature Generation)**:
  - 기존 특성에서 새로운 특성을 생성하여 모델의 성능을 개선합니다.

### 4. 하이퍼파라미터 튜닝 (Hyperparameter Tuning)
모델의 하이퍼파라미터를 최적화하여 성능을 향상시키는 방법입니다.

- **그리드 서치 (Grid Search)**:
  - 여러 하이퍼파라미터 조합을 시도하여 최적의 조합을 찾습니다.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

- **랜덤 서치 (Random Search)**:
  - 무작위로 하이퍼파라미터 조합을 시도하여 최적의 조합을 찾습니다.

```python
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['auto', 'sqrt', 'log2']
}
random_search = RandomizedSearchCV(DecisionTreeClassifier(), param_distributions, n_iter=100, cv=5, random_state=42)
random_search.fit(X_train, y_train)
```

### 5. 데이터 전처리 (Data Preprocessing)
- **데이터 정규화 (Normalization)**: 데이터의 스케일을 조정하여 모델의 성능을 개선할 수 있습니다.
- **결측치 처리 (Handling Missing Values)**: 결측치를 적절히 처리하여 모델의 정확성을 높일 수 있습니다.

이 방법들을 조합하여 사용하면 결정 트리의 정확도를 높이고 성능을 개선할 수 있습니다.