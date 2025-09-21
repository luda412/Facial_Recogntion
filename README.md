# 🍹 표정 인식 술 추천 서비스

혼술을 소비하는 사람들 중 현재 감정에 맞는 술을 추천 받고 싶어하는 잠재적 사용자들을 위한 표정 인식 술 추천 서비스

## 📆 개발 기간

2022.09 ~ 2022.12

## 🛠️ 기술 스택

<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/javascript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black">

## 📝 프로젝트 소개

- 🍷 레드 와인

  - 레드 와인을 마신 사람들 중 60% -> 피곤한 감정을 느낌
  - 레드 와인을 마신 사람들 중 53% -> <strong>긴장이 풀린</strong> 것을 느낌

- 🥂 화이트 와인

  - 화이트 와인을 마신 사람들 중 33% -> <strong>긴장이 풀린</strong> 것을 느낌
  - 화이트 와인을 마신 사람들 중 28% -> 자신감이 넘치는 감정을 느낌
    - 레드 와인에 비해 적은 감정 변화를 보임

- 🥃 위스키, 보드카, 럼, 진 등(이하 양주)

  - 양주를 마신 사람들 중 59% -> <strong>자신감이 넘치는 감정</strong>을 느낌
  - 양주를 마신 사람들 중 58% -> 힘이 솟는 것 같은 감정을 느낌
  - 양주를 마신 사람들 중 30% -> 공격적인 성향을 보임

- 🍺 맥주
  - 맥주를 마신 사람들 중 50% -> 긴장이 풀린 것을 느낌
  - 맥주를 마신 사람들 중 45% -> <strong>자신감이 넘치는 감정</strong>을 느낌
  - 맥주를 마신 사람들 중 39% -> 피곤한 감정을 느낌

> 기사 인용: https://www.fnnews.com/news/201711231532426767

## 🏹 개념 모델

<img width="1249" height="533" alt="Image" src="https://github.com/user-attachments/assets/2cdc4518-ecbd-4adb-ae72-d52072e5464a" />

## 🖼️ Wire Frame

<img width="1247" height="703" alt="Image" src="https://github.com/user-attachments/assets/64e81567-199d-401b-a022-d3c323b297a2" />

## ⭐️ 주요 기능

### 표정 인식을 통한 사용자의 현재 감정 추출

#### 1. 얼굴 랜드마크 검출

- `Mediapipe`와 `OpenCV` `DNN`을 이용해 얼굴을 감지하고, 특정 랜드마크 지점의 좌표를 추출합니다. 이는 `FacialLandmarkDetector` 클래스에서 처리됩니다.

#### 2. 특징점 정규화

- 추출된 랜드마크 좌표는 <strong>정규화 과정</strong>을 거칩니다. 이는 `min_max_normalization` 함수에서 구현되며, 좌표 값을 0과 1 사이로 변환합니다. 이 과정을 통해 다양한 얼굴 크기와 위치에 대한 변동성을 줄여줍니다.

#### 3. 데이터셋 준비

- emotions 리스트에 정의된 감정<span style = 'background-color:#fff5b1'>(분노, 행복, 슬픔, 놀람)</span>에 따라 이미지 데이터를 로드합니다. 각 이미지에 대해 랜드마크 좌표를 추출하고, 정규화된 좌표를 특징점으로 사용합니다. 이 과정은 `extract_landmark_features` 함수에서 처리되며, 각 이미지의 감정에 해당하는 레이블도 함께 저장됩니다.

#### 4. 표정 인식 모델 학습

- `FacialExpressionRecognizer` 클래스를 사용하여 SVM 모델을 학습시킵니다. SVM 모델은 선형 커널과 정규화 파라미터 C=1을 사용합니다. 학습된 모델은 train 메서드를 통해 학습되고, save 메서드를 통해 저장됩니다.

#### 5. 성능 평가

- 학습된 모델을 사용하여 예측을 수행하고, 실제 레이블과 비교하여 <strong>혼동 행렬(confusion matrix)</strong>을 생성합니다. 이를 통해 모델의 성능을 평가할 수 있습니다. 혼동 행렬은 모델의 예측 정확도를 시각적으로 표현합니다.

## 🍬 기대효과

#### 맞춤형 추천

- 표정 인식 기술을 통해 <span style = 'background-color:#fff5b1'>사용자의 현재 감정을 실시간으로 분석</span>하고, 그에 맞는 술을 추천함으로써 개인화된 음주 경험을 제공합니다. 이를 통해 사용자는 자신의 감정 상태에 가장 잘 맞는 술을 선택하여 만족스러운 음주 경험을 가질 수 있습니다.

#### 정확한 감정 분석

- `Mediapipe`와 `OpenCV` `DNN`을 활용한 얼굴 랜드마크 검출 및 정규화 과정을 통해 정확하고 신뢰할 수 있는 감정을 분석하여 추천의 정확성을 높이고 사용자로 하여금 신뢰도를 향상시킵니다.
