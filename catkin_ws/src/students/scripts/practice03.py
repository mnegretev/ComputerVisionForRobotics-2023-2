#
# COMPUTER VISION FOR ROBOTICS - FI-UNAM - 2023-2
# PRACTICE 02 - LINE DETECTION BY HOUGH TRANSFORM
#

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def convolve2d(A, kernel):
    return cv2.convertScaleAbs(cv2.filter2D(A, cv2.CV_16S, kernel))

def get_gaussian_kernel(k,sigma):
    k = k//2
    H = np.zeros((2*k+1, 2*k+1))
    r,c = H.shape
    for i in range(r):
        for j in range(c):
            H[i,j] = 1.0/(2*math.pi*sigma*sigma)*math.exp(-((i-k)**2 + (j-k)**2)/(2*sigma*sigma))
    H /= np.sum(H)
    return H

def get_sobel_x_gradient(A):
    Gx = np.asarray([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])/8.0
    return convolve2d(A, Gx)

def get_sobel_y_gradient(A):
    Gy = np.asarray([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])/8.0
    return convolve2d(A, Gy)

def get_sobel_mag_angle(A):
    Gx = get_sobel_x_gradient(A)
    Gy = get_sobel_y_gradient(A)
    Gm = np.zeros(Gx.shape)
    Ga = np.zeros(Gx.shape)
    r,c = Gx.shape
    for i in range(r):
        for j in range(c):
            Gm[i,j] = math.sqrt(Gx[i,j]**2 + Gy[i,j]**2)
            Ga[i,j] = math.atan2(Gy[i,j],Gx[i,j])
            if Ga[i,j] < 0:
                Ga[i,j] += math.pi
            Ga[i,j] = int(Ga[i,j]/math.pi*180)
    return Gm.astype(np.uint8), Ga.astype(np.uint8)

def supress_non_maximum(Gm, Ga):
    G = np.zeros(Gm.shape)
    r,c = Gm.shape
    for i in range(1,r-1):
        for j in range(1,c-1):
            if Ga[i,j] <= 22 or Ga[i,j] > 157:
                if Gm[i,j] >= Gm[i, j+1] and Gm[i,j] > Gm[i, j-1]:
                    G[i,j] = Gm[i,j]
                else:
                    G[i,j]=0

            elif Ga[i,j] > 22 and Ga[i,j] <= 67:
                if Gm[i,j] >= Gm[i-1, j+1] and Gm[i,j] > Gm[i+1, j-1]:
                    G[i,j] = Gm[i,j]
                else:
                    G[i,j]=0

            elif Ga[i,j] > 67 and Ga[i,j] <= 112:
                if Gm[i,j] >= Gm[i-1, j] and Gm[i,j] > Gm[i+1,j]:
                    G[i,j] = Gm[i,j]
                else:
                    G[i,j]=0

            else:       
                if Gm[i,j] >= Gm[i-1, j-1] and Gm[i,j] > Gm[i+1, j+1]:
                    G[i,j] = Gm[i,j]
                else:
                    G[i,j] = 0
    return G.astype(np.uint8)

def umbrales(G):
    z = np.ones(G.shape)
    r,c = G.shape

    Umax=16
    Umin=14

    for i in range(1,r-1):
        for j in range(1,c-1):
            if G[i,j]<Umax and G[i,j]>Umin:
                if G[i-1,j-1]==0 or G[i-1,j]==0 or G[i-1,j+1]==0 or G[i,j-1]==0 or G[i,j+1]==0 or G[i+1,j-1]==0 or G[i+1,j]==0 or G[i+1,j+1]==0:
                    z[i,j]=0
                else:
                    z[i,j]=255
            elif G[i,j]>Umax:
                z[i,j]=255

            elif G[i,j]<Umin:
                z[i,j]=0
    z=G*z
    return z

def Transformacion_hough(canny,copia_canny,umbral=84):  #83,50
    # global x1,y1,x2,y2

    r,c=np.shape(canny)
    diagonal = np.sqrt(np.square(r) + np.square(c))
    pasos_rhos=len(np.arange(-diagonal,diagonal))/180
    

    thetas=np.arange(0,360,step=1)
    rhos = np.arange(-diagonal, diagonal,step=pasos_rhos)
    Acumulador=np.zeros((256,256))

    for i in range(r):
        for j in range(c):
            if canny[i,j]!=0:
                punto = [i - 128, j - 128]            #####
                
                for ang_min in range (180):
                    rho=punto[1]*np.cos(np.deg2rad(ang_min))+punto[0]*np.sin(np.deg2rad(ang_min))
                    theta=ang_min
                    rho_idx = np.argmin(np.abs(rhos - rho))

                    Acumulador[rho_idx][ang_min] += 1

    for y in range(Acumulador.shape[0]):
        for x in range(Acumulador.shape[1]):
            if Acumulador[y][x] > umbral:

                rho = rhos[y]
                theta = thetas[x]
                a = np.cos(np.deg2rad(theta))
                b = np.sin(np.deg2rad(theta))
                x0 = (a * rho) + 128
                y0 = (b * rho) + 128
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                
                cv2.imshow('canny',copia_canny)
                cv2.line(canny,(x1,y1),(x2,y2),(255,0,0),thickness=1)
                cv2.imshow('image',canny)
                
    
def main():
    global frame,clone

    cap =cv2.VideoCapture(0)
    cv2.namedWindow('image') 
    
    while(True):
        ret, frame = cap.read()
        if(ret): 
            clone=frame.copy()
            clone=cv2.resize(clone,(256,256))
            image = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
            kernel = get_gaussian_kernel(5,1)
            filtered = convolve2d(image, kernel)
            Gx = get_sobel_x_gradient(filtered)
            Gy = get_sobel_y_gradient(filtered)
            Gm, Ga = get_sobel_mag_angle(filtered)
            G = supress_non_maximum(Gm, Ga)
            canny=umbrales(G)
            copia_canny=canny.copy()
           
            Transformacion_hough(canny,copia_canny)
            
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
        else:
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()
