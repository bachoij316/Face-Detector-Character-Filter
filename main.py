import cv2
import dlib
import sys
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture("TwoGirls.mp4")


while True:
    ret, img = cap.read()
    if not ret:
        break
    width = int(img.shape[1]*50/100)
    height = int(img.shape[0]*50/100)
    dim = (width, height)
    img = cv2.resize(
        img, dim)
    faces = detector(img)
    try:
        face = faces[0]

        dlib_shape = predictor(img, face)
        shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

        top_left = np.min(shape_2d, axis=0)
        bottom_right = np.max(shape_2d, axis=0)

        face_size = int(max(bottom_right - top_left) * 1.4)

        center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)

        img = cv2.rectangle(img, pt1=(face.left(), face.top()), pt2=(face.right(
        ), face.bottom()), color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        for s in shape_2d:
            cv2.circle(img, center=tuple(s), radius=1, color=(
                255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    except IndexError:
        pass

    try:
        face2 = faces[1]

        dlib_shape = predictor(img, face2)
        shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

        top_left = np.min(shape_2d, axis=0)
        bottom_right = np.max(shape_2d, axis=0)

        face_size = int(max(bottom_right - top_left) * 1.4)

        center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)

        img = cv2.rectangle(img, pt1=(face2.left(), face2.top()), pt2=(face2.right(
        ), face2.bottom()), color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        for s in shape_2d:
            cv2.circle(img, center=tuple(s), radius=1, color=(
                255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    except IndexError:
        pass

    cv2.imshow("Two Girls", img)
    cv2.waitKey(1)
