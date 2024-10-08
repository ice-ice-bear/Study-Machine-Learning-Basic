## 하이퍼파라미터를 튜닝하는 방법

그리드 서치와 랜덤 서치 외에도 하이퍼파라미터를 튜닝하는 방법은 여러 가지가 있습니다. 대표적으로는 **베이지안 최적화**, **하이퍼밴드(Hyperband)**, 그리고 **진화 알고리즘(Evolutionary Algorithms)** 등이 있습니다. 각각의 방법을 예시 코드와 함께 구체적으로 설명하겠습니다.

### 1. 베이지안 최적화 (Bayesian Optimization)

베이지안 최적화는 함수의 최대값이나 최소값을 찾는 데 사용되는 방법으로, 하이퍼파라미터 튜닝에 자주 사용됩니다. 이를 위해 `scikit-optimize` 라이브러리의 `BayesSearchCV`를 사용할 수 있습니다.

```python
from skopt import BayesSearchCV
from sklearn.datasets import load_iris
from sklearn.svm import SVC

# 데이터 준비
iris = load_iris()
X, y = iris.data, iris.target

# SVM 모델 정의
svc = SVC()

# 하이퍼파라미터 범위 설정
param_space = {
    'C': (1e-6, 1e+6, 'log-uniform'),
    'gamma': (1e-6, 1e+1, 'log-uniform'),
    'degree': (1, 8),
    'kernel': ['linear', 'poly', 'rbf']
}

# 베이지안 최적화 수행
opt = BayesSearchCV(estimator=svc, search_spaces=param_space, n_iter=32, cv=3, n_jobs=-1, random_state=42)
opt.fit(X, y)

# 최적의 하이퍼파라미터와 성능 출력
print("Best parameters:", opt.best_params_)
print("Best cross-validation score:", opt.best_score_)
```

### 2. 하이퍼밴드 (Hyperband)

하이퍼밴드는 자원의 효율적인 할당을 통해 하이퍼파라미터 최적화를 수행하는 방법입니다. `scikit-optimize` 라이브러리의 `SuccessiveHalvingSearchCV`를 사용하여 구현할 수 있습니다.

```python
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import RandomForestClassifier

# 데이터 준비
X, y = load_iris(return_X_y=True)

# 랜덤 포레스트 모델 정의
rfc = RandomForestClassifier()

# 하이퍼파라미터 범위 설정
param_grid = {
    'max_depth': [3, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'n_estimators': [10, 50, 100]
}

# 하이퍼밴드 수행
hb_search = HalvingGridSearchCV(estimator=rfc, param_grid=param_grid, factor=2, random_state=42)
hb_search.fit(X, y)

# 최적의 하이퍼파라미터와 성능 출력
print("Best parameters:", hb_search.best_params_)
print("Best cross-validation score:", hb_search.best_score_)
```

### 3. 진화 알고리즘 (Evolutionary Algorithms)

진화 알고리즘은 유전 알고리즘 등을 사용해 하이퍼파라미터를 최적화하는 방법입니다. `DEAP` 라이브러리를 사용하여 구현할 수 있습니다.

```python
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# 데이터 준비
X, y = load_iris(return_X_y=True)

# SVM 모델 정의
def evaluate(individual):
    C, gamma = individual
    clf = SVC(C=C, gamma=gamma)
    return cross_val_score(clf, X, y, cv=3).mean(),

# 하이퍼파라미터 범위 설정
toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, 1e-6, 1e+6)
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_float, toolbox.attr_float), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 진화 알고리즘 설정
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# 진화 알고리즘 수행
population = toolbox.population(n=50)
ngen = 10
cxpb = 0.5
mutpb = 0.2

algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=True)

# 최적의 하이퍼파라미터와 성능 출력
best_individual = tools.selBest(population, k=1)[0]
print("Best individual:", best_individual)
print("Best cross-validation score:", evaluate(best_individual))
```

### 요약

- **베이지안 최적화**: 하이퍼파라미터 공간을 효율적으로 탐색.
- **하이퍼밴드**: 자원 효율성을 높여 탐색.
- **진화 알고리즘**: 유전 알고리즘을 사용해 탐색.

이들 방법은 각기 다른 특성을 가지며, 문제의 특성과 자원 제약에 따라 적절한 방법을 선택해 사용할 수 있습니다.