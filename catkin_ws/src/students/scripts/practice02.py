#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 19:51:46 2023

@author: root
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 12:13:25 2023

@author: root
"""

import cv2
import numpy
import math

def convolve2d(A, kernel): #se convoluciona para obtener el gradiente
    return cv2.convertScaleAbs(cv2.filter2D(A, cv2.CV_16S, kernel))
#Filter 2D es para convolucion 
#cv2._CV16s se pasan a 16 bits con signo para los posibles negativos  
#convertScaleAbs  es para regresar al formato original, no se necesita el signo
#del gradiente, saca valor absoluto     

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

def get_sobel_mag_angle(A):  #magnitud y angulo 
    Gx = get_sobel_x_gradient(A)
    Gy = get_sobel_y_gradient(A)
    Gm = numpy.zeros(Gx.shape)
    Ga = numpy.zeros(Gx.shape)
    r,c = Gx.shape
    for i in range(r):
        for j in range(c):
            Gm[i,j] = math.sqrt(Gx[i,j]**2 + Gy[i,j]**2) #
            Ga[i,j] = math.atan2(Gy[i,j],Gx[i,j])
            if Ga[i,j] < 0:
                Ga[i,j] += math.pi
            Ga[i,j] = int(Ga[i,j]/math.pi*180)  # esto es para cuantizar  y pasar a grados
            # se limita los angulos positivos , porque se compara con el mismo
    return Gm.astype(numpy.uint8), Ga.astype(numpy.uint8) #se convierte la magnitud
# para que este entre 0 y 255 

def supress_non_maximum(Gm, Ga): #suprimir no maximos 
    G = numpy.zeros(Gm.shape)
    r,c = Gm.shape                                    
    for i in range(1,r-1):#se ignora todo el borde 
        for j in range(1,c-1):
            #cuantizar el angulo 
                #di, dj = 0, 1
            if  Ga[i,j] <= 22 or Ga[i,j] > 157:     
               if Gm[i,j] >= Gm[i, j+1] and Gm[i,j] > Gm[i, j-1]:
                  G[i,j] = Gm[i,j]
               else:
                   G[i,j] = 0
               
            if Ga[i,j] > 22 and Ga[i,j] <= 67:
                #di, dj = 1, 1
                if Gm[i,j] >= Gm[i+1, j+1] and Gm[i,j] > Gm[i-1, j-1]:
                   G[i,j] = Gm[i,j]
                else:
                    G[i,j] = 0
            if Ga[i,j] > 67 and Ga[i,j] <= 112:
                #di, dj = 1, 0
                if Gm[i,j] >= Gm[i+1, j] and Gm[i,j] > Gm[i-1, j]:
                   G[i,j] = Gm[i,j]
                else:
                   G[i,j] = 0
            if Ga[i,j] > 112 and Ga[i,j] <= 157:
                #di, dj = 1, -1
                 if Gm[i,j] >= Gm[i+1, j-1] and Gm[i,j] > Gm[i+1, j-1]:
                    G[i,j] = Gm[i,j]
                 else:
                    G[i,j] = 0
         
    return G.astype(numpy.uint8)

def umbral(G): #suprimir no maximos 
    global Max, Min
    U = numpy.zeros(G.shape)
    r,c = G.shape
    um1= Max
    um2= Min
    for i in range(1,r-1):#se ignora todo el borde 
        for j in range(1,c-1):
            #cuantizar el angulo 
            if G[i,j] >= um1 : #pixel fuerte 
                U[i,j]=255
            if G[i,j]<= um2:
                U[i,j]=0  #pixel debil 
            if G[i,j] > um2 and G[i,j] < um1:
                if G[i+1,j] >= um1 or G[i-1,j] >= um1 or G[i-1,j+1] >=um1 or G[i-1,j-1]  >=um1  or G[i+1,j+1] >=um1 or G[i+1,j-1] >=um1 or G[i,j+1] >=um1 or G[i,j-1] >=um1: 
                    U[i,j]=255
                else:
                    U[i,j]=0
 
    return U.astype(numpy.uint8)

def trackbar_callback(val):
    global Max
    Max = val
def trackbar_callback2(val2):
    global Min
    Min = val2    
  
    
def canny_border_detector(A, k_size, sigma, low_th, high_th):
    filtered = convolve2d(A, get_gaussian_kernel(k_size, sigma)) 
    #filtro gaussiano pone borrosita la imagen 
    Gm, Ga = get_sobel_mag_angle(filtered)
    #se obtiene magnitud y angulo 
    G = supress_non_maximum(Gm, Ga)
    plana =  [x for y in Gm for x in y ] 
    Max= max(plana)
    Min= min(plana)
    U= umbral(G)

    

def main():
    global Max ,G,Min,Gm,Gx,Ga,U
    Max= 30
    Min= 0
    #img = cv2.imread("chica.jpg")
    #cap  = cv2.VideoCapture(0)
    
    cv2.namedWindow('umbral2') 
    cv2.createTrackbar('max','umbral2',0, Max, trackbar_callback)
    cv2.createTrackbar('min','umbral2',Min, Max, trackbar_callback2)
    
    cap  = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame1 = cap.read()
        frame1=cv2.resize(frame1,(256,256))
        #frame2= frame1.copy(2
     
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        img = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        kernel = get_gaussian_kernel(5,1)
        filtered = convolve2d(img, kernel)
        Gx = get_sobel_x_gradient(filtered)
        Gy = get_sobel_y_gradient(filtered)
        Gm, Ga = get_sobel_mag_angle(filtered)
        G = supress_non_maximum(Gm, Ga)
        cv2.imshow("Original", img)
        cv2.imshow("Filtered", filtered)
        cv2.imshow("Gx", Gx)
        cv2.imshow("Gy", Gy)
        cv2.imshow("G mag", Gm)
        cv2.imshow("G ang", Ga)
        cv2.imshow("G supressed", G)
        U= umbral(G)
        U_copy = U.copy()
        cv2.imshow("umbral2", U_copy)  
        #cv2.imshow('My Video', frame1)

        

        if cv2.waitKey(10) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


 

if __name__ == '__main__':
    main()
