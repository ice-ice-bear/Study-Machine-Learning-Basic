## [혼공머신]2-1 -추가학습(지도학습 및 비지도학습)

### 지도 학습 (Supervised Learning) 알고리즘

#### 1 **선형 회귀 (Linear Regression)**
- **특징**:
  
  - 입력 변수와 출력 변수 간의 선형 관계를 모델링
  - 회귀 계수를 사용하여 예측
  - 간단하고 빠르며 해석이 용이함
  
- **용도**:
  
  - 연속형 변수 예측 (예: 집값 예측, 매출 예측)
  
- **예시코드**:

  ```python
  import numpy as np
  from sklearnlinear_model import LinearRegression
  
  # 예제 데이터
  X = nparray([[1], [2], [3], [4], [5]])
  y = nparray([1, 4, 9, 16, 25])
  
  # 모델 훈련
  model = LinearRegression()
  modelfit(X, y)
  
  # 예측
  predictions = modelpredict(nparray([[6], [7]]))
  print(predictions)
  ```

  

#### 2 **로지스틱 회귀 (Logistic Regression)**
- **특징**:
  - 이진 분류 문제에 사용
  - 시그모이드 함수로 출력값을 0과 1 사이로 변환
  - 결과를 확률로 해석할 수 있음
  
- **용도**:
  - 분류 문제 (예: 스팸 메일 분류, 질병 진단)
  
- **예시코드**:

  ```python
  import numpy as np
  from sklearnlinear_model import LogisticRegression
  
  # 예제 데이터
  X = nparray([[0], [1], [2], [3], [4], [5]])
  y = nparray([0, 0, 0, 1, 1, 1])
  
  # 모델 훈련
  model = LogisticRegression()
  modelfit(X, y)
  
  # 예측
  predictions = modelpredict(nparray([[15], [35]]))
  print(predictions)
  ```

  

#### 3 **k-최근접 이웃 (k-Nearest Neighbors, k-NN)**
- **특징**:
  - 데이터 포인트 간의 거리를 기반으로 분류 및 회귀 수행
  - 단순하지만 계산 비용이 높음
  - 비선형 문제에 효과적
  
- **용도**:
  
  - 분류 및 회귀 (예: 이미지 분류, 추천 시스템)
  
- **예시코드**:

  ```python
  import numpy as np
  from sklearnneighbors import KNeighborsClassifier
  
  # 예제 데이터
  X = nparray([[0], [1], [2], [3], [4], [5]])
  y = nparray([0, 0, 0, 1, 1, 1])
  
  # 모델 훈련
  model = KNeighborsClassifier(n_neighbors=3)
  modelfit(X, y)
  
  # 예측
  predictions = modelpredict(nparray([[15], [35]]))
  print(predictions)
  ```

  

#### 4 **서포트 벡터 머신 (Support Vector Machine, SVM)**
- **특징**:
  - 최대 마진 분리 초평면을 사용하여 분류
  - 커널 트릭을 통해 비선형 문제 해결
  - 고차원 데이터에 효과적
  
- **용도**:
  
  - 분류 문제 (예: 텍스트 분류, 이미지 인식)
  
- **예시코드**:

  ```python
  import numpy as np
  from sklearn.svm import SVC
  
  # 예제 데이터
  X = np.array([[0], [1], [2], [3], [4], [5]])
  y = np.array([0, 0, 0, 1, 1, 1])
  
  # 모델 훈련
  model = SVC(kernel='linear')
  model.fit(X, y)
  
  # 예측
  predictions = model.predict(np.array([[1.5], [3.5]]))
  print(predictions
  ```



#### 5 **결정 트리 (Decision Tree)**

- **특징**:
  
  - 트리 구조를 사용하여 결정 규칙을 모델링
  - 해석이 용이하고 시각화가 가능
  - 과적합 가능성이 높음
  
- **용도**:
  - 분류 및 회귀 (예: 고객 세분화, 리스크 분석)
  
- **예시코드**:

  ```python
  import numpy as np
  from sklearn.tree import DecisionTreeClassifier
  
  # 예제 데이터
  X = np.array([[0], [1], [2], [3], [4], [5]])
  y = np.array([0, 0, 0, 1, 1, 1])
  
  # 모델 훈련
  model = DecisionTreeClassifier()
  model.fit(X, y)
  
  # 예측
  predictions = model.predict(np.array([[1.5], [3.5]]))
  print(predictions)
  ```

  

### 비지도 학습 (Unsupervised Learning) 알고리즘

#### 1 **k-평균 클러스터링 (k-Means Clustering)**
- **특징**:
  - 데이터 포인트를 k개의 클러스터로 분할
  - 각 클러스터의 중심을 기준으로 군집화
  - 간단하고 빠르지만 초기값에 민감
  
- **용도**:
  - 데이터 군집화 (예: 고객 분류, 이미지 세그먼트)
  
- **예시코드**:

  ```python
  import numpy as np
  from sklearn.cluster import KMeans
  
  # 예제 데이터
  X = np.array([[1, 2], [1, 4], [1, 0],
                [10, 2], [10, 4], [10, 0]])
  
  # 모델 훈련
  model = KMeans(n_clusters=2, random_state=0)
  model.fit(X)
  
  # 클러스터 할당
  labels = model.predict(X)
  print(labels)
  ```

  

#### 2 **주성분 분석 (Principal Component Analysis, PCA)**
- **특징**:
  - 고차원 데이터를 저차원으로 변환
  - 데이터의 분산을 최대화하는 새로운 축을 찾음
  - 차원 축소 및 데이터 시각화에 유용
  
- **용도**:
  - 차원 축소 (예: 이미지 압축, 노이즈 제거)
  
- **예시코드**:

  ```python
  import numpy as np
  from sklearn.decomposition import PCA
  
  # 예제 데이터
  X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
  
  # 모델 훈련
  pca = PCA(n_components=2)
  X_reduced = pca.fit_transform(X)
  
  print(X_reduced)
  ```

  

#### 3 **아이소맵 (Isomap)**
- **특징**:
  - 비선형 차원 축소 기법
  - 지오데식 거리(비선형 거리)를 사용하여 차원 축소
  - 고차원 데이터의 비선형 구조를 보존
  
- **용도**:
  - 차원 축소 (예: 얼굴 인식, 문서 시각화)
  
- **예시코드**:

  ```python
  import numpy as np
  from sklearn.manifold import Isomap
  
  # 예제 데이터
  X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
  
  # 모델 훈련
  isomap = Isomap(n_neighbors=2, n_components=2)
  X_reduced = isomap.fit_transform(X)
  
  print(X_reduced)
  ```



#### 4 **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**

- **특징**:
  
  - 밀도 기반 클러스터링 알고리즘
  - 임의의 모양을 가진 클러스터를 탐지
  - 노이즈 데이터 포인트를 잘 처리
  
- **용도**:
  - 데이터 군집화 (예: 지리 공간 데이터 분석, 이상 감지)
  
- **예시코드**:

  ```python
  import numpy as np
  from sklearn.cluster import DBSCAN
  
  # 예제 데이터
  X = np.array([[1, 2], [2, 2], [2, 3],
                [8, 7], [8, 8], [25, 80]])
  
  # 모델 훈련
  dbscan = DBSCAN(eps=3, min_samples=2)
  labels = dbscan.fit_predict(X)
  
  print(labels)
  ```

  

### 비교 요약
- **지도 학습**:
  - 입력과 타깃 데이터 사용
  - 예측 모델을 학습하여 새로운 데이터에 대한 예측 수행
  - 대표적인 알고리즘: 선형 회귀, 로지스틱 회귀, k-NN, SVM, 결정 트리

- **비지도 학습**:
  - 타깃 데이터 없이 입력 데이터만 사용
  - 데이터의 패턴이나 구조를 탐지
  - 대표적인 알고리즘: k-평균 클러스터링, PCA, 아이소맵, DBSCAN