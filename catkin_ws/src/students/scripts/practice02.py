#
# COMPUTER VISION FOR ROBOTICS - FI-UNAM - 2023-2
# PRACTICE 02 - BORDER DETECTION
#

import cv2
import numpy as np
import math

def trackbar_callback_3(val):
    global k_size
    k_size=val
    return

def trackbar_callback_2(val):
    global t2
    t2 = val
    return

def trackbar_callback_1(val):
    global t1
    t1 = val
    return

def threshold(G, t_1, t_2):
    T = np.zeros(G.shape)
    r,c = G.shape
    for i in range(1,r-1):
        for j in range(1,c-1):
            if G[i,j]>t_1 and G[i,j]<=t_2:
                T[i,j]=60
            elif G[i,j] > t_2 :
                T[i,j]=255
    return T.astype(np.uint8)

def final_supress(T):
    F = np.zeros(T.shape)
    r,c = T.shape
    for i in range(1,r-1):
        for j in range(1,c-1):
            if T[i,j]==60 and (T[i+1,j]==255 or T[i+1,j+1]==255 or T[i+1,j-1]==255 or T[i-1,j]==255 or T[i-1,j+1]==255 or T[i-1,j-1]==255 or T[i,j+1]==255 or T[i,j-1]==255 ):
                F[i,j]=255
            elif T[i,j]==255:
                F[i,j]=255
    return F.astype(np.uint8)


def supress_non_maximum(Gm, Ga):
    G = np.zeros(Gm.shape)
    r,c = Gm.shape
    for i in range(1,r-1):
        for j in range(1,c-1):
            if Ga[i,j] <= 22 or Ga[i,j] > 157:
                di, dj = 0, 1
            elif Ga[i,j] > 22 and Ga[i,j] <= 67:
                di, dj = 1, 1
            elif Ga[i,j] > 67 and Ga[i,j] <= 112:
                di, dj = 1, 0
            else:
                di, dj = 1, -1
            if Gm[i,j] >= Gm[i+di, j+dj] and Gm[i,j] > Gm[i-di, j-dj]:
                G[i,j] = Gm[i,j]
            else:
                G[i,j] = 0
    return G.astype(np.uint8)


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

def mag_angle(A):
    Gx = get_sobel_x_gradient(A)
    Gy = get_sobel_y_gradient(A)
    r,c=Gx.shape
    Gm=np.zeros(Gx.shape)
    Ga=np.zeros(Gx.shape)
    for i in range(r):
        for j in range(c):
            Gm[i,j]=math.sqrt(Gx[i,j]**2+Gy[i,j]**2)
            Ga[i,j]=math.atan2(Gy[i,j],Gx[i,j])
            if Ga[i,j]<0:
                Ga[i,j]+=math.pi
            Ga[i,j] = int(Ga[i,j]/math.pi*180)
    return Gm.astype(np.uint8), Ga.astype(np.uint8)

def main():
    global k_size, t1, t2
    k_size=7
    t1=1
    t2=4
    cap  = cv2.VideoCapture(0) #Default resolution 1920x1080
    cap.set(3, 640) #Change the camera resolution to 640x480
    cap.set(4, 480)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cv2.createTrackbar('Umbral 1','Original', t1, 10, trackbar_callback_1)
        cv2.createTrackbar('Umbral 2','Original', t2, 20, trackbar_callback_2)
        cv2.createTrackbar('Kernel size','Original', k_size, 20, trackbar_callback_3)
        kernel=filter_gaussian(k_size,1.2)
        frame_copy = frame.copy()
        frame_grey = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
        frame_grey = cv2.resize(frame_grey, None, fx = 0.35, fy = 0.35, interpolation = cv2.INTER_CUBIC)
        frame_filter=cv2.filter2D(frame_grey, cv2.CV_16S, kernel)
        frame_filter=cv2.convertScaleAbs(frame_filter) #Convert Scale image
        Gm, Ga=mag_angle(frame_filter)
        G=supress_non_maximum(Gm, Ga)
        T=threshold(G, t1, t2)
        F=final_supress(T)
        cv2.imshow("threshold",T)
        cv2.imshow("Final cany",F)
        cv2.imshow('Original', frame)
        #valores del 1 al 60 aprox en Gx
        if cv2.waitKey(30) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

