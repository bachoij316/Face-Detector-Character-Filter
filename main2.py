import cv2
import dlib
import sys
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture("Girl.mp4")

overlay = cv2.imread("RightFace.png", cv2.IMREAD_UNCHANGED)


def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    try:
        bg_img = background_img.copy()
        # convert 3 channels to 4 channels
        if bg_img.shape[2] == 3:
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

        if overlay_size is not None:
            img_to_overlay_t = cv2.resize(
                img_to_overlay_t.copy(), overlay_size)

        b, g, r, a = cv2.split(img_to_overlay_t)

        mask = cv2.medianBlur(a, 5)

        h, w, _ = img_to_overlay_t.shape
        roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

        img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(),
                                  mask=cv2.bitwise_not(mask))
        img2_fg = cv2.bitwise_and(
            img_to_overlay_t, img_to_overlay_t, mask=mask)

        bg_img[int(y-h/2):int(y+h/2), int(x-w/2)               :int(x+w/2)] = cv2.add(img1_bg, img2_fg)

        # convert 4 channels to 4 channels
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

        return bg_img
    except Exception:
        return background_img


while True:
    ret, img = cap.read()
    if not ret:
        break
    width = int(img.shape[1]*32/100)
    height = int(img.shape[0]*32/100)
    dim = (width, height)
    img = cv2.resize(
        img, dim)
    origin = img.copy()
    faces = detector(img)

    try:
        face = faces[0]

        dlib_shape = predictor(img, face)
        shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

        top_left = np.min(shape_2d, axis=0)
        bottom_right = np.max(shape_2d, axis=0)

        face_size = int(max(bottom_right - top_left) * 1.4)

        center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)

        result = overlay_transparent(
            origin, overlay, center_x + 20, center_y - 30, overlay_size=(face_size, face_size))

        img = cv2.rectangle(img, pt1=(face.left(), face.top()), pt2=(face.right(
        ), face.bottom()), color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        for s in shape_2d:
            cv2.circle(img, center=tuple(s), radius=1, color=(
                255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        cv2.circle(img, center=tuple(top_left), radius=1, color=(
            255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(img, center=tuple(bottom_right), radius=1,
                   color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(img, center=tuple((center_x, center_y)), radius=1,
                   color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    except IndexError:
        pass

    cv2.imshow("Girl", result)
    cv2.waitKey(1)
