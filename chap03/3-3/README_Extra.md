### L1 규제와 L2 규제

L1 규제와 L2 규제는 머신러닝 모델, 특히 선형 회귀 모델에서 과대적합을 방지하고 모델의 일반화 성능을 향상시키기 위해 사용하는 두 가지 주요 규제 방법입니다.

#### L1 규제 (L1 Regularization, Lasso Regression)

- **정의**: L1 규제는 비용 함수에 계수의 절대값 합을 추가합니다.
  $$
  
  \text{L1 규제 비용 함수} = \text{기본 비용 함수} + \lambda \sum_{i=1}^{n} |\beta_i|
  $$
  여기서 $ \lambda $는 규제 강도를 조절하는 하이퍼파라미터이고, $ \beta_i  $는 모델의 계수입니다.
  
- **특징**:
  
  - **계수 희소화**: L1 규제는 일부 계수를 0으로 만들 수 있습니다. 즉, 특정 특성의 영향을 완전히 제거합니다.
  - **특성 선택**: L1 규제를 통해 자동으로 불필요한 특성을 선택하지 않게 됩니다. 이는 모델을 더 간결하고 해석 가능하게 만듭니다.
  - **모델 복잡도 감소**: 계수의 절대값 합을 최소화함으로써 모델의 복잡도를 줄이고, 과대적합을 방지합니다.
  
- **적용 예**: Lasso 회귀는 L1 규제를 적용한 회귀 모델입니다.

#### L2 규제 (L2 Regularization, Ridge Regression)

- **정의**: L2 규제는 비용 함수에 계수의 제곱합을 추가합니다.
  $$
  
  \text{L2 규제 비용 함수} = \text{기본 비용 함수} + \lambda \sum_{i=1}^{n} \beta_i^2
  $$
  여기서 $ \lambda $는 규제 강도를 조절하는 하이퍼파라미터이고, $ \beta_i $는 모델의 계수입니다.
  
- **특징**:
  
  - **계수 감소**: L2 규제는 계수를 작게 만들지만, 완전히 0으로 만들지는 않습니다. 모든 특성의 영향을 유지합니다.
  - **과대적합 방지**: 계수의 제곱합을 최소화함으로써 모델의 복잡도를 줄이고, 과대적합을 방지합니다.
  - **모델 안정화**: 모든 특성을 유지하면서 과대적합을 방지하기 때문에 모델이 좀 더 안정적입니다.
  
- **적용 예**: Ridge 회귀는 L2 규제를 적용한 회귀 모델입니다.

### 시각적 비교

다음 그림은 L1 규제와 L2 규제의 차이를 시각적으로 설명합니다.

![regularization](regularization.png)

1. **L1 규제 (Lasso)**
    - 절대값 합을 최소화하므로, 계수가 0이 되는 경우가 발생합니다.
    - 결과적으로, 일부 특성은 모델에서 완전히 제거됩니다.
    
2. **L2 규제 (Ridge)**
    - 제곱합을 최소화하므로, 계수가 작아지지만 0이 되지는 않습니다.
    - 모든 특성이 모델에 포함되어 기여하게 됩니다.

### 시각적 비교 예시

이 그림은 L1 규제와 L2 규제가 계수에 어떻게 영향을 미치는지를 보여줍니다. L1 규제에서는 다이아몬드 모양의 등고선을 형성하여 계수가 0이 될 가능성이 높음을 나타내고, L2 규제에서는 원형 등고선을 형성하여 모든 계수가 작아지지만 0이 되지 않는 특성을 나타냅니다.