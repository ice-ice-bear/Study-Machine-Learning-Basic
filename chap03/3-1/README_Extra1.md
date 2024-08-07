## k-최근접 이웃 모델의 원리와 활용

### k-최근접 이웃 모델의 수학적 원리

k-최근접 이웃(K-Nearest Neighbors, KNN) 모델은 비모수적 방법으로, 데이터 포인트가 주어졌을 때 가장 가까운 \( k \)개의 이웃을 기반으로 새로운 데이터를 예측하는 방법입니다. 이 알고리즘은 다음과 같은 단계를 따릅니다:

1. **거리 측정**: 새로운 데이터 포인트와 기존 데이터 포인트 간의 거리를 계산합니다. 일반적으로 유클리드 거리(Euclidean distance)를 사용합니다. 유클리드 거리 \( d \)는 다음과 같이 계산됩니다:
   $$
   d(\mathbf{x}_i, \mathbf{x}_j) = \sqrt{\sum_{l=1}^{n} (x_{il} - x_{jl})^2}
   $$
   여기서 $\mathbf{x}_i $와 $ \mathbf{x}_j $는 각각 \( n \)차원의 벡터를 나타냅니다.

2. **이웃 선택**: 계산된 거리들을 기준으로 가장 가까운 \( k \)개의 데이터 포인트(이웃)를 선택합니다.

3. **예측**:
   - **분류(Classification)**: 이웃들의 클래스 레이블 중 다수결을 통해 새로운 데이터 포인트의 클래스를 결정합니다. 예를 들어, \( k \)개의 이웃 중 가장 많은 클래스가 예측 클래스가 됩니다.
   - **회귀(Regression)**: 이웃들의 타깃값의 평균을 구하여 새로운 데이터 포인트의 값을 예측합니다. 예를 들어, \( k \)개의 이웃의 타깃값의 산술 평균이 예측값이 됩니다.

### 분류와 회귀를 모두 해결할 수 있는 이유

k-최근접 이웃 모델은 본질적으로 **비모수적**(non-parametric)이며, 데이터의 구조를 미리 가정하지 않습니다. 이는 이 모델이 다양한 문제 유형에 유연하게 적용될 수 있는 이유입니다. 

#### 1. 분류 문제
- **다수결 원칙**: 분류 문제에서는 새로운 데이터 포인트의 클래스가 가장 가까운 \( k \)개의 이웃 중에서 가장 빈번한 클래스로 결정됩니다. 이는 다수결 원칙에 기반한 간단한 투표 방식입니다. 이 방식은 데이터 포인트들이 특정 클래스에 속할 확률을 반영합니다.
- **예시**: 고양이와 개를 구분하는 문제에서 새로운 이미지가 들어왔을 때, 가장 가까운 \( k \)개의 이미지가 고양이인지 개인지 투표하여 새로운 이미지의 클래스를 결정합니다.

#### 2. 회귀 문제
- **평균 계산**: 회귀 문제에서는 새로운 데이터 포인트의 값이 가장 가까운 \( k \)개의 이웃의 값의 평균으로 결정됩니다. 이는 근처 데이터 포인트들의 값을 기반으로 새로운 값을 예측하는 것입니다.
- **예시**: 집의 크기와 가격 데이터를 기반으로 새로운 집의 가격을 예측할 때, 가장 가까운 \( k \)개의 집들의 가격의 평균을 구하여 새로운 집의 가격을 예측합니다.

### 수학적 예시

#### 분류 예시
주어진 데이터 포인트 \((3, 3)\)에 대한 클래스를 예측한다고 가정합니다. 이웃의 개수 \( k = 3 \)로 설정합니다. 기존 데이터 포인트와의 거리를 계산하여 가장 가까운 3개의 이웃을 찾습니다.

| 포인트 | 클래스 | 거리 (유클리드) |
| ------ | ------ | --------------- |
| (1, 2) | A      | 2.236           |
| (4, 2) | B      | 1.414           |
| (2, 3) | A      | 1.000           |
| (5, 3) | B      | 2.000           |
| (3, 4) | A      | 1.000           |

가장 가까운 3개의 이웃은 (4, 2), (2, 3), (3, 4)이며, 클래스는 B, A, A입니다. 따라서 예측 클래스는 다수결 원칙에 따라 A가 됩니다.

#### 회귀 예시
주어진 데이터 포인트 \((3, 3)\)에 대한 값을 예측한다고 가정합니다. 이웃의 개수 \( k = 3 \)로 설정합니다. 기존 데이터 포인트와의 거리를 계산하여 가장 가까운 3개의 이웃을 찾습니다.

| 포인트 | 값   | 거리 (유클리드) |
| ------ | ---- | --------------- |
| (1, 2) | 10   | 2.236           |
| (4, 2) | 20   | 1.414           |
| (2, 3) | 15   | 1.000           |
| (5, 3) | 25   | 2.000           |
| (3, 4) | 30   | 1.000           |

가장 가까운 3개의 이웃은 (4, 2), (2, 3), (3, 4)이며, 값은 20, 15, 30입니다. 따라서 예측값은 \((20 + 15 + 30) / 3 = 21.67\)이 됩니다.

### 요약
k-최근접 이웃 모델은 비모수적 접근법을 통해 분류와 회귀 문제를 모두 해결할 수 있습니다. 분류에서는 다수결 원칙을 통해 클래스를 예측하고, 회귀에서는 평균 계산을 통해 값을 예측합니다. 이는 이 모델이 데이터의 분포나 가정에 의존하지 않고, 단순히 거리 기반의 이웃 관계를 통해 예측을 수행하기 때문에 가능합니다.