#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 17:27:21 2023

@author: cv
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 22:30:26 2023

@author: cv
"""

import cv2
import numpy
import math

def convolve2d(A, kernel): #se convoluciona para obtener el gradiente
    return cv2.convertScaleAbs(cv2.filter2D(A, cv2.CV_16S, kernel))

def get_gaussian_kernel(k,sigma): # Genera un kernel gaussiano k ancho del kernel
    #sigma es la varianza, los filtros generalmente son simetricos 
    #Se recorre la imagen 
    k = k//2
    H = numpy.zeros((2*k+1, 2*k+1))
    r,c = H.shape
    for i in range(r):
        for j in range(c):
            H[i,j] = 1.0/(2*math.pi*sigma*sigma)*math.exp(-((i-k)**2 + (j-k)**2)/(2*sigma*sigma))
    H /= numpy.sum(H)  # Es para normalizar para no escalar la imagen original
    return H

def get_sobel_x_gradient(A):
    Gx = numpy.asarray([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])/8.0
    #Genera el kernel 
    return convolve2d(A, Gx)

def get_sobel_y_gradient(A):
    Gy = numpy.asarray([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])/8.0
    return convolve2d(A, Gy)

def eigenvalores(M):
    x,v=numpy.linalg.eig(M)
    return x

def get_harris_response(x,k):
    global r
    r=numpy.linalg.det(M)-k*((x[0]+x[1])**2)
    return r
def supresion(rmatriz,img):
    image_norm = cv2.normalize(rmatriz, None, alpha=0,beta=1, norm_type=cv2.NORM_MINMAX)
    r,c=image_norm.shape
    plana =  [x for y in image_norm for x in y ] 
    Max= max(plana)
    for k1 in range( 0,r):
        for k2 in range( 0, c):
            if image_norm[k1,k2]<0.09:
               cir=cv2.circle(img,(k2,k1),3,(0,0,255))
               img=cir
               
    return img 

def matriz(A):
    global M,Gx,rmatriz
    w =2
    Gx = get_sobel_x_gradient(A);
    Gy = get_sobel_y_gradient(A);
    M = numpy.zeros([2,2])
    rmatriz= numpy.zeros((len(A),len(A)))
    ta=len(A)-w-1
    for i in range( 0,ta):
        for j in range( 0,ta):
                  M[0,0] = Gx[i,j]*Gx[i,j]+Gx[i,j+1]*Gx[i,j+1]+Gx[i+1,j]*Gx[i+1,j]+Gx[i+1,j+1]*Gx[i+1,j+1]
                  M[0,1] = Gx[i,j]*Gy[i,j]+Gx[i,j+1]*Gy[i,j+1]+Gx[i+1,j]*Gy[i+1,j]+Gx[i+1,j+1]*Gy[i+1,j+1]
                  M[1,0] = Gx[i,j]*Gy[i,j]+Gx[i,j+1]*Gy[i,j+1]+Gx[i+1,j]*Gy[i+1,j]+Gx[i+1,j+1]*Gy[i+1,j+1]
                  M[1,1] = Gy[i,j]*Gy[i,j]+Gy[i,j+1]*Gy[i,j+1]+Gy[i+1,j]*Gy[i+1,j]+Gy[i+1,j+1]*Gy[i+1,j+1]
                  valores=eigenvalores(M)
                  rmatriz[i,j]=get_harris_response(valores,k)
           
    return M,rmatriz


def main():
    
    
    global k,img
    k=0.5
    cap  = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, img = cap.read()
        img=cv2.resize(img,(256,256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #frame2= frame1.copy()
     
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
         
        kernel = get_gaussian_kernel(5,1)
        img= convolve2d(img, kernel)
        #cv2.imshow("Filtered", img)
   
        Gx = get_sobel_x_gradient(img);
        Gy = get_sobel_y_gradient(img);
        #cv2.imshow("sobel x", Gx)
  
        #cv2.imshow("sobel y", Gy)
        matriz(img)
        final=supresion(rmatriz,img)
        cv2.imshow("esquinas", final)

        

        if cv2.waitKey(10) & 0xFF == 27:
            break
    cap.release()
    #out.release()
    cv2.destroyAllWindows()
    
    
   

if __name__ == '__main__':
    main()
    cap  = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, img = cap.read()
        img=cv2.resize(img,(256,256))
        img= img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break           
    cap.release()


       
