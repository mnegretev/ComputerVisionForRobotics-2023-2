#
# COMPUTER VISION FOR ROBOTICS - FI-UNAM - 2023-2
# PRACTICE 02 - BORDER DETECTION
#

import cv2
import numpy as np
import math

def get_sobel_x_gradient(A):
    Gx = np.asarray([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])/8.0
    return cv2.convertScaleAbs(cv2.filter2D(A, cv2.CV_16S, Gx))

def get_sobel_y_gradient(A):
    Gy = np.asarray([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])/8.0
    return cv2.convertScaleAbs(cv2.filter2D(A, cv2.CV_16S, Gy))

def filter_gaussian(k,sigma):
    siz=k//2
    H = np.zeros((2*siz+1, 2*siz+1))
    for i in range(2*siz+1):
        for j in range(2*siz+1):
            arg=-((i-siz)**2+(j-siz)**2)/(2*sigma**2)
            H[i,j]=math.exp(arg)/(2*math.pi*sigma**2)
    H=H/np.sum(H)
    return H

def main():
    kernel=filter_gaussian(5,1)
    cap  = cv2.VideoCapture(0) #Default resolution 1920x1080
    cap.set(3, 640) #Change the camera resolution to 640x480
    cap.set(4, 480)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame_copy = frame.copy()
        frame_grey = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
        frame_filter=cv2.filter2D(frame_grey, cv2.CV_16S, kernel)
        frame_filter=cv2.convertScaleAbs(frame_filter) #Convert Scale image
        Gx = get_sobel_x_gradient(frame_filter)
        Gy = get_sobel_y_gradient(frame_filter)
        
        cv2.imshow('Original', frame_copy)
        cv2.imshow('Filter', frame_filter)
        cv2.imshow('Sobel x', Gx)
        cv2.imshow('Sobel y', Gy)
        #valores del 1 al 60 aprox en Gx
        if cv2.waitKey(10) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

