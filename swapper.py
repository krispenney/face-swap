import cv2
import numpy as np

# Load Haar classifer for profile faces
profile_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect faces
    faces = profile_cascade.detectMultiScale(gray, 1.3, 5)

    if(len(faces) >= 2):
        (x1, y1, w1, h1) = faces[0]
        (x2, y2, w2, h2) = faces[1]

        # Get ROIs
        roi_face1 = img[y1:y1+h1, x1:x1+w1]
        roi_face2 = img[y2:y2+h2, x2:x2+w2]

        #resize and swap faces
        roi_face1 = cv2.resize(roi_face1, (h2, w2), interpolation = cv2.INTER_AREA)
        roi_face2 = cv2.resize(roi_face2, (h1, w1), interpolation = cv2.INTER_AREA)
        img[y2:y2+h2, x2:x2+w2] = roi_face1
        img[y1:y1+h1, x1:x1+w1] = roi_face2

    # output new frame
    cv2.imshow('img', img)

    # Exit loop
    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
        break

# Dealloc
cap.release()
cv2.destroyAllWindows()
