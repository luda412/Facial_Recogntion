from libs.FacialExpression import FacialExpressionRecognizer
from libs.Face import FacialLandmarkDetector
from sklearn.metrics import confusion_matrix


import libs.Face as Face
import cv2
import numpy as np
import glob


# 학습을 위해 필요한 것
# 이미지
# 이미지 --> 랜드마크 추출
# 이미지 --> 어떤 감성 인지 레이블링


FEATURE_LANDMARK_INDEX = [70, 107, 336, 300, 159, 130, 133, 362, 386, 359, 8, 11, 16, 61, 291, 50, 280,127, 93, 58,136,140,148,377,378,365,288,323,356,
                           296, 293,300 ,107, 66,70,63,117, 50, 105, 425, 280, 346,72,73, 74, 184, 77, 90, 180, 85, 315, 404, 320, 307, 408, 304, 303, 302,
                           29, 30, 247, 25, 110, 24, 23, 22, 26, 112, 190, 56, 28,257, 258, 286, 414, 463, 341, 256, 252, 253, 254, 339, 255, 359, 467, 259
                          ,168, 6, 197,200, 208, 428, 175,260,334,205,27]

DATASET_PATH = "./datasets"

facial_landmark_detector = FacialLandmarkDetector(
        model=Face.FACIAL_LANDMARK_DETECTION_MODEL_MEDIAPIPE,
        face_detector=Face.FACE_DETECTION_MODEL_OPENCV_DNN)

def extract_landmark_features(img_paths):
    features = []

    for img_path in img_paths:
        feature = []
        img = cv2.imread(img_path)
        # img = array (n, n, 3)

        facial_landmark_detector.feed(img)

        if facial_landmark_detector.getFacialLandmark():
            landmarks = facial_landmark_detector.getFacialLandmark()

            x = landmarks.getX()
            y = landmarks.getY()

            x = min_max_normalization(x)
            y = min_max_normalization(y)

            for i in FEATURE_LANDMARK_INDEX:
                feature.append(x[i])
                feature.append(y[i])

            features.append(feature)

    return features

def min_max_normalization(value):
    value = list(value)

    _max = max(value)
    _min = min(value)

    result = []

    for val in value:
        _val = (val - _min) / (_max - _min)
        result.append(_val)

    return np.array(result)

def main():
    emotions = ['anger', 'happiness', 'sadness', 'surprise']
    labels = []
    features = []

    for emotion in emotions:
        # 감성 별 이미지 저장 폴더 접근
        img_paths = glob.glob(DATASET_PATH + "/" + emotion + "/*")
        # 폴더 안에 이미지들로 부터 랜드마크 특징 추출
        _feature = extract_landmark_features(img_paths)
        features = features + _feature
        # 감성 레이블 저장
        for i in range(len(_feature)):
            labels.append(emotions.index(emotion))

    features = np.array(features)
    labels = np.array(labels)

    print(features.shape)
    print(labels.shape)

    ### SVM_Train ###
    facial_expression_recognizer = FacialExpressionRecognizer(kernel='linear', C=1)

    facial_expression_recognizer.train(features, labels)

    # test_img, test_label = features[0], labels[0]

    facial_expression_recognizer.feed(features)

    # facial_expression_recognizer.test(features, labels)

    result = facial_expression_recognizer.getPrediction()

    # print(result)

    cm = confusion_matrix(labels, result)

    print(cm)

    facial_expression_recognizer.save("2022_12_08_face_model")























if __name__ == '__main__':
    main()
