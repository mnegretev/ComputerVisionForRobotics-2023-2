import numpy as np
import cv2

cap  = cv2.VideoCapture(0) #Default resolution 1920x1080
cap.set(3, 640) #Change the camera resolution to 640x480
cap.set(4, 480)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame = cv2.resize(frame,(640,480))
    print(frame.shape)
    cv2.imshow('My Video', frame)
    if cv2.waitKey(10) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
