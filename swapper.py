import cv2
import numpy as np

smile_cascade = cv2.CascadeClassifier('parojos.xml')
profile_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # smiles = smile_cascade.detectMultiScale(gray, 1.3, 5)
    smiles = profile_cascade.detectMultiScale(gray, 1.3, 5)
    faces = []
    for (x, y, w, h) in smiles:
    # if(len(smiles) > 0):
        # (x, y, w, h) = smiles[0]
        faces.append((x, y, w, h))
        # sub_face = img[y:y+h, x:x+w]
        # sub_face = cv2.GaussianBlur(sub_face, (23, 23), 5)
        # img[y:y+h, x:x+w] = sub_face
        # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # roi_gray = gray[y:y+h, x:x+w]
        # roi_color = img[y:y+h, x:x+w]

    if(len(faces) >= 2):
        (x1, y1, w1, h1) = faces[0]
        (x2, y2, w2, h2) = faces[1]
        sub_face1 = img[y1:y1+h1, x1:x1+w1]
        sub_face2 = img[y2:y2+h2, x2:x2+w2]

        #resize and swap faces
        sub_face1 = cv2.resize(sub_face1, (h2, w2), interpolation = cv2.INTER_AREA)
        sub_face2 = cv2.resize(sub_face2, (h1, w1), interpolation = cv2.INTER_AREA)
        img[y2:y2+h2, x2:x2+w2] = sub_face1
        img[y1:y1+h1, x1:x1+w1] = sub_face2


    cv2.imshow('img', img)
    cv2.waitKey(0)
    # k = cv2.waitKey(30) & 0xff
    # if k == 27:
    #     break

cap.release()
cv2.destroyAllWindows()
