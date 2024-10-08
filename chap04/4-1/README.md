### 로지스틱 회귀

로지스틱 회귀는 선형 회귀와 달리 분류 문제에 사용되는 지도 학습 알고리즘입니다. 주요한 특징은 선형 방정식을 사용하여 데이터 포인트를 분류하는데, 이 과정에서 시그모이드 함수나 소프트맥스 함수를 사용해 클래스 확률을 출력합니다.

#### 핵심 개념
1. **로지스틱 회귀 (Logistic Regression)**
   - 선형 회귀와 달리, 로지스틱 회귀는 이진 분류 문제에 사용됩니다.
   - 출력값은 시그모이드 함수에 의해 0과 1 사이의 확률 값으로 변환됩니다.

2. **시그모이드 함수 (Sigmoid Function)**
   - $  \sigma(x) = \frac{1}{1 + e^{-x}} $
   - 선형 방정식의 출력을 0과 1 사이의 값으로 압축합니다.
   - 주로 이진 분류에서 사용됩니다.

3. **소프트맥스 함수 (Softmax Function)**
   - 다중 분류 문제에서 사용됩니다.
   - 여러 선형 방정식의 출력을 정규화하여 확률 분포를 만듭니다.
   - 각 클래스의 확률 합이 1이 되도록 조정합니다.

#### 주요 패키지 및 함수 (scikit-learn)
- **LogisticRegression**
  - 로지스틱 회귀를 위한 클래스입니다.
  - `solver` 매개변수로 최적화 알고리즘을 선택할 수 있습니다. 예: `'lbfgs'`, `'sag'`, `'saga'`.
  - `penalty` 매개변수로 규제 방법을 선택합니다. 예: `'l2'` (릿지), `'l1'` (라쏘).
  - `C` 매개변수로 규제 강도를 제어합니다. 기본값은 1.0입니다.

- **predict_proba()**
  - 예측 확률을 반환합니다.
  - 이진 분류의 경우 음성 클래스와 양성 클래스의 확률을 반환합니다.
  - 다중 분류의 경우 모든 클래스에 대한 확률을 반환합니다.

- **decision_function()**
  - 모델이 학습한 선형 방정식의 출력을 반환합니다.
  - 이진 분류의 경우 양성 클래스의 점수를 반환합니다.
  - 다중 분류의 경우 각 클래스에 대한 선형 방정식의 결과를 반환합니다.

#### 예제 코드 설명
노트북에 포함된 주요 코드 블록들을 설명하면 다음과 같습니다.

1. **데이터 전처리 및 모델 훈련**
   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler
   from sklearn.linear_model import LogisticRegression
   import numpy as np
   
   # 데이터셋을 훈련 세트와 테스트 세트로 분할
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
   # 데이터 스케일링
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   
   # 로지스틱 회귀 모델 훈련
   lr = LogisticRegression()
   lr.fit(X_train_scaled, y_train)
   ```

2. **예측 및 확률 계산**
   ```python
   # 테스트 데이터에 대한 예측 확률 계산
   proba = lr.predict_proba(X_test_scaled[:5])
   print(np.round(proba, decimals=3))
   
   # 결정 함수 값 계산
   decision = lr.decision_function(X_test_scaled[:5])
   print(np.round(decision, decimals=2))
   ```

3. **소프트맥스 함수 적용**
   ```python
   from scipy.special import softmax
   
   # 결정 함수 값을 소프트맥스 함수로 확률 변환
   proba = softmax(decision, axis=1)
   print(np.round(proba, decimals=3))
   ```

### 추가 정보
- **오버피팅과 언더피팅**: 모델의 일반화 성능을 평가하고 조절하는 방법.
- **정규화 기법**: L1, L2 정규화를 사용하여 모델의 복잡도를 제어하는 방법.
- **최적화 알고리즘**: Gradient Descent, Stochastic Gradient Descent 등 다양한 최적화 방법의 원리와 적용법.
