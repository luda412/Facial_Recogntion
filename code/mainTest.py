from libs.FacialExpression import DataAnalyzer
from libs.Face import FacialLandmarkDetector
from libs.FacialExpression import FacialExpressionRecognizer
import cv2
import numpy as np
import libs.Face as Face
import Scripts5
import Scripts6
import Scripts7
import Scripts8

import webbrowser
import sys

facial_landmark_detector = FacialLandmarkDetector(
        model=Face.FACIAL_LANDMARK_DETECTION_MODEL_MEDIAPIPE,
        face_detector=Face.FACE_DETECTION_MODEL_OPENCV_DNN)


FEATURE_LANDMARK_INDEX = [70, 107, 336, 300, 159, 130, 133, 362, 386, 359, 8, 11, 16, 61, 291, 50, 280,127, 93, 58,136,140,148,377,378,365,288,323,356,
                           296, 293,300 ,107, 66,70,63,117, 50, 105, 425, 280, 346,72,73, 74, 184, 77, 90, 180, 85, 315, 404, 320, 307, 408, 304, 303, 302,
                           29, 30, 247, 25, 110, 24, 23, 22, 26, 112, 190, 56, 28,257, 258, 286, 414, 463, 341, 256, 252, 253, 254, 339, 255, 359, 467, 259
                          ,168, 6, 197,200, 208, 428, 175,260,334,205,27]


def min_max_normalization(value):
    value = list(value)

    _max = max(value)
    _min = min(value)

    result = []

    for val in value:
        _val = (val - _min) / (_max - _min)
        result.append(_val)

    return np.array(result)


# 를 만들어
def extract_landmark_features(frame):
    features = []
    feature = []
    facial_landmark_detector.feed(frame)

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


def main():
    filepath = "hello.html"

    facial_landmark_detector = FacialLandmarkDetector(
        model=Face.FACIAL_LANDMARK_DETECTION_MODEL_MEDIAPIPE,
        face_detector=Face.FACE_DETECTION_MODEL_OPENCV_DNN)

    face_model = FacialExpressionRecognizer()
    face_model.load('2022_12_08_face_model.pkl')

    cap = cv2.VideoCapture(0)

    labelss = []
    featuress = []

    while True:

        _, frame = cap.read()

        frame = cv2.flip(frame, 1)  # 좌우반전

        frame_feature = []

        frame_feature = extract_landmark_features(frame)

        frame_feature =np.array(frame_feature)
        frame_feature.shape

        face_model.feed(frame_feature)

        result = face_model.getPrediction()

        if result == 0:
            text = 'anger'
        elif result == 1:
            text = 'happiness'
        elif result == 2:
            text = 'sadness'
        elif result == 3:
            text = 'surprise'

        # cv2.putText(frame, text, (), )
        # cv2.putText(frame, "Text" , (10,10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        facial_landmark_detector.feed(frame)

        if facial_landmark_detector.getIsDetect():

            landmark = facial_landmark_detector.getFacialLandmark()

            x = landmark.getX()
            y = landmark.getY()

            for i in FEATURE_LANDMARK_INDEX:
                cv2.circle(frame, (int(x[i]), int(y[i])), 1, (255, 0, 0), 1)

        cv2.putText(frame, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # facial_landmark_detector.feed(frame) #feed로 이미지 넣기
        #
        # if facial_landmark_detector.getIsDetect():
        #
        #     landmark = facial_landmark_detector.getFacialLandmark()
        #
        #     x=landmark.getX()
        #     y=landmark.getY()
        #
        #     for i in FEATURE_LANDMARK_INDEX:
        #         cv2.circle(frame, (int(x[i]),int(y[i])),1,(255,0,0),1)
        #
        #         print("point {}: {}, {}".format(i,x[i],y[i]))

        cv2.imshow("frame", frame)

        key = cv2.waitKey(1)

        if key == 27:
            if result == 0:
                Scripts5.main()
            elif result == 1:
                Scripts6.main()
            elif result == 2:
                Scripts7.main()
            elif result == 3:
                Scripts8.main()
            break


    cap.release()


if __name__ == '__main__':
    main()

