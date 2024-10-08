## 결정 트리 (Decision Tree)

결정 트리는 예/아니오에 대한 질문을 통해 데이터를 분류하거나 예측하는 데 사용되는 알고리즘입니다. 이해하기 쉽고, 예측 과정도 투명하게 드러나기 때문에 매우 유용한 도구입니다.

### 핵심 포인트

1. **불순도 (Impurity)**:
   - 결정 트리는 데이터를 나눌 때, 각 분할이 얼마나 순수한지를 평가합니다.
   - `사이킷런`은 지니 불순도와 엔트로피 불순도를 제공합니다.
   - 불순도는 노드의 순수도를 나타내는 척도로, 낮을수록 더 순수합니다.
2. **정보 이득 (Information Gain)**:
   - 정보 이득은 부모 노드와 자식 노드의 불순도 차이로 정의됩니다.
   - 결정 트리는 정보 이득이 최대화되도록 학습합니다.
3. **과대적합 (Overfitting)**:
   - 결정 트리는 트리가 너무 깊어지면 훈련 데이터에 과대적합될 수 있습니다.
   - 이를 방지하기 위해 가지치기 (Pruning)을 사용합니다.
   - `사이킷런`은 여러 가지 가지치기 매개변수를 제공합니다.
4. **특성 중요도 (Feature Importance)**:
   - 결정 트리는 각 특성이 분할에 기여한 정도를 계산할 수 있습니다.
   - 이는 모델의 해석성을 높여주는 중요한 장점입니다.

### 결정 트리 알고리즘 이해하기

결정 트리 알고리즘은 데이터 분류와 회귀 문제를 해결하는 데 사용됩니다. 예/아니오와 같은 질문을 반복하여 데이터를 분할하며, 최종적으로 목표 변수의 예측을 돕습니다. 아래는 결정 트리의 시각화와 이를 읽는 방법에 대한 설명입니다.

#### 결정 트리 시각화

이미지

#### 트리 시각화 해독 방법

1. **노드 (Node)**:
   - 각 사각형은 하나의 노드를 나타냅니다.
   - 노드는 데이터를 특정 특성에 따라 분할합니다.

2. **루트 노드 (Root Node)**:
   - 트리의 최상단에 위치한 노드입니다.
   - 전체 데이터를 가장 잘 분할하는 기준을 나타냅니다.

3. **내부 노드 (Internal Node)**:
   - 루트 노드 아래에 위치한 노드들입니다.
   - 데이터를 계속 분할하여 보다 구체적인 질문을 만듭니다.

4. **리프 노드 (Leaf Node)**:
   - 더 이상 분할되지 않는 최종 노드입니다.
   - 각 리프 노드는 최종 예측을 나타냅니다.

5. **노드 구성 요소**:
   - **특성 (Feature)**: 노드가 데이터를 분할하는 기준이 되는 특성입니다.
   - **임계값 (Threshold)**: 분할 기준이 되는 값입니다.
   - **gini**: 지니 불순도로, 0에 가까울수록 순수한 노드를 나타냅니다.
   - **samples**: 노드에 있는 샘플의 수입니다.
   - **value**: 각 클래스에 속한 샘플의 수입니다.
   - **class**: 예측 클래스입니다.

### 결정 트리 시각화 예제 코드 설명

#### 데이터 생성 및 전처리

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 데이터 생성
X, y = make_classification(n_samples=100, n_features=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- `make_classification()`: 분류 문제를 위한 가상 데이터를 생성합니다.
- `train_test_split()`: 데이터를 훈련 세트와 테스트 세트로 나눕니다.

#### 결정 트리 모델 학습

```python
from sklearn.tree import DecisionTreeClassifier

# 모델 학습
model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
model.fit(X_train, y_train)
```
- `DecisionTreeClassifier()`: 결정 트리 분류기를 생성합니다.
  - `criterion='gini'`: 지니 불순도를 사용하여 노드를 분할합니다.
  - `max_depth=3`: 트리의 최대 깊이를 3으로 제한하여 과대적합을 방지합니다.
  - `random_state=42`: 결과 재현성을 위해 랜덤 시드를 설정합니다.
- `fit()`: 모델을 학습 데이터에 맞춰 학습시킵니다.

#### 결정 트리 시각화

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# 트리 시각화
plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=[f'Feature {i}' for i in range(1, 5)])
plt.show()
```
- `plot_tree()`: 결정 트리 모델을 시각화합니다.
  - `filled=True`: 각 노드를 색으로 채워 타깃값을 시각적으로 구분합니다.
  - `feature_names=[f'Feature {i}' for i in range(1, 5)]`: 특성의 이름을 그래프에 표시합니다.
- `plt.show()`: 시각화된 트리를 화면에 출력합니다.

이 시각화는 결정 트리 모델이 어떻게 데이터를 분류하고 있는지를 직관적으로 보여줍니다. 각 노드에서 데이터를 분할하는 기준과 그에 따른 결과를 이해할 수 있습니다.