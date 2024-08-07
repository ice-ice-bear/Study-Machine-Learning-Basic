## [혼공머신]2-1 -추가학습(데이터 분할 방법)

### 데이터 분할의 종류와 기준

데이터 분할은 모델의 성능을 평가하고, 과적합을 방지하기 위해 필수적인 단계입니다. 데이터 분할의 주요 종류와 각 방법의 기준은 다음과 같습니다.

#### 1. 홀드아웃 방법 (Holdout Method)
- **정의**: 데이터를 훈련 세트와 테스트 세트로 한 번만 나누는 방법입니다.
- **분할 기준**:
  - **훈련 세트**: 전체 데이터의 70-80%를 차지합니다.
  - **테스트 세트**: 전체 데이터의 20-30%를 차지합니다.
- **장점**: 간단하고 빠릅니다.
- **단점**: 데이터의 특정 부분이 훈련 세트나 테스트 세트에 치우칠 수 있습니다.
- **코드 예시**:
  ```python
  from sklearn.model_selection import train_test_split
  train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)
  ```

#### 2. 교차 검증 (Cross-Validation)
- **정의**: 데이터를 여러 번 분할하여 각각 다른 훈련 세트와 테스트 세트를 사용하는 방법입니다.
- **분할 기준**:
  - **K-폴드 교차 검증**: 데이터를 K개의 폴드로 나누고, 각 폴드를 한 번씩 테스트 세트로 사용합니다.
  - **스트래티파이드 K-폴드 교차 검증**: 분류 문제에서 각 클래스 비율이 동일하게 유지되도록 데이터를 나눕니다.
- **장점**: 데이터의 모든 부분이 훈련 및 테스트에 사용되어 더 안정적인 평가를 제공합니다.
- **단점**: 계산 비용이 많이 들 수 있습니다.
- **코드 예시**:
  ```python
  from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
  
  # K-폴드 교차 검증
  kf = KFold(n_splits=5, shuffle=True, random_state=42)
  scores = cross_val_score(model, data, target, cv=kf)
  
  # 스트래티파이드 K-폴드 교차 검증
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  stratified_scores = cross_val_score(model, data, target, cv=skf)
  ```

#### 3. 시간 기반 분할 (Time-Based Split)
- **정의**: 시간 순서가 중요한 시계열 데이터에서 사용되는 방법입니다. 과거 데이터를 훈련 세트로, 미래 데이터를 테스트 세트로 사용합니다.
- **분할 기준**:
  - **훈련 세트**: 과거 데이터
  - **테스트 세트**: 미래 데이터
- **장점**: 시계열 데이터의 특성을 반영하여 적합한 모델 평가가 가능합니다.
- **단점**: 데이터가 충분하지 않으면 어려울 수 있습니다.
- **코드 예시**:
  ```python
  # 예시로 시계열 데이터를 80:20으로 분할
  train_size = int(len(data) * 0.8)
  train, test = data[:train_size], data[train_size:]
  ```

#### 4. 부트스트래핑 (Bootstrapping)
- **정의**: 데이터에서 중복을 허용하여 여러 번 샘플링을 수행하는 방법입니다. 훈련 세트와 테스트 세트를 여러 번 재샘플링하여 모델을 평가합니다.
- **분할 기준**:
  - **훈련 세트**: 원본 데이터에서 샘플링된 데이터
  - **테스트 세트**: 샘플링에서 제외된 데이터
- **장점**: 데이터의 여러 샘플에 대해 모델의 성능을 평가할 수 있습니다.
- **단점**: 원본 데이터가 충분히 크지 않으면 샘플링 편향이 발생할 수 있습니다.
- **코드 예시**:
  ```python
  from sklearn.utils import resample
  
  # 부트스트래핑 예시
  n_iterations = 1000
  n_size = int(len(data) * 0.8)
  for i in range(n_iterations):
      train = resample(data, n_samples=n_size)
      test = [x for x in data if x not in train]
      model.fit(train)
      score = model.score(test)
  ```

### 데이터 분할 시 고려 사항
1. **데이터 균형**: 분류 문제에서 각 클래스의 비율이 훈련 세트와 테스트 세트에서 동일하게 유지되도록 합니다.
2. **무작위성**: 데이터 분할 시 랜덤성을 도입하여 특정 패턴이 훈련 세트나 테스트 세트에 치우치지 않도록 합니다.
3. **데이터 크기**: 데이터가 충분히 클수록 더 작은 테스트 세트로도 모델의 일반화 성능을 평가할 수 있습니다.
4. **시간 순서**: 시계열 데이터의 경우, 시간 순서를 고려하여 데이터를 분할해야 합니다.

이러한 다양한 분할 방법과 기준을 이해하고 상황에 맞게 적절히 사용하는 것이 중요합니다. 이를 통해 모델의 성능을 더 정확하게 평가하고, 과적합을 방지할 수 있습니다.