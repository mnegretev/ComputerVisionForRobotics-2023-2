#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 16:37:35 2023

@author: cv
"""


import numpy as np
import cv2 as cv


def mouse_callback(event,x,y,flags,param):
    global  start_point,end_point,C1,C2,C3,C4,img_roi
    #if event==cv.EVENT_MOUSEMOVE:,C1 
    if event == cv.EVENT_LBUTTONDOWN:
        start_point=(x, y)
        C1=start_point[0]
        C2=start_point[1]
        print( start_point)
    
    if event == cv.EVENT_LBUTTONUP:
        end_point=(x, y)
        cv.rectangle(img = img_copy,pt1 = start_point,pt2 = (x,y), color = color, thickness = thickness)
        print(end_point)
        C3=end_point[0]
        C4=end_point[1]
        promedio()
        umbrales(circle_radius)
  
def trackbar_callback(val):
    global circle_radius
    circle_radius = val 
    
def promedio ():
    global b,g,r,meanb,meang,meanr
    img_roi=img_baboon[C2:C4, C1:C3]  
    mean2 = cv.mean(img_roi) #promedio de una dimension
    print(mean2)
    b,g,r= cv.split(img_roi)
    meanb = cv.mean(b)[0] 
    meang = cv.mean(g)[0] 
    meanr = cv.mean(r)[0] 
    
    
def umbrales (circle_radius):
    global  umbralb,umbralg,umbralr,Numbralb,Numbralg, Numbralr,mask,low,upp
    global  img_and, imagenRGB
    umbralb= meanb+circle_radius
    umbralg= meang+circle_radius
    umbralr= meanr+circle_radius   
    
    Numbralb= meanb-circle_radius
    Numbralg= meang-circle_radius
    Numbralr= meanr-circle_radius   
   
    low=np.array([umbralb,umbralg,umbralr],np.uint8)
    upp=np.array([Numbralb,Numbralg,Numbralr],np.uint8)

    mask=cv.inRange(img_baboon,upp,low)
    mask=mask.astype(np.uint8)
    #imagenRGB=cv.cvtColor(img_baboon, cv.COLOR_BGR2RGB)
    cv.namedWindow('mask') #nombra la ventana 
    cv.imshow("mask", mask)
    cv.waitKey(0)# va a estar abierta 
    maskinv=cv.bitwise_not(mask)
    img_and = cv.bitwise_and( img_baboon,img_baboon, mask=maskinv)
    cv.namedWindow('and') #nombra la ventana 
    cv.imshow("and", img_and)
    cv.waitKey(0)
    fondo()
  
def fondo():
     global tama1,tama2,dim,retam,fondoinv,maskinv,final,img_and
     tama1= img_baboon.shape[1]
     tama2= img_baboon.shape[0]
     dim= (tama1,tama2)
     retam=cv.resize(img_fondo,dim)
     cv.namedWindow('tama') #nombra la ventana 
     cv.imshow("tama", retam)
     cv.waitKey(0)
     fondoinv=cv.bitwise_and( retam,retam, mask=mask)
     final=cv.add(fondoinv,img_and)
     cv.namedWindow('fondoinvertido') #nombra la ventana 
     cv.imshow("fondoinvertido", final)
     cv.waitKey(0)
     
         
def main():
    global img_baboon, img_copy, thickness, color,mean,mean2,circle_radius,low,upp,mask,img_roi
    global  img_and, img_fondo
    #ancho = 512
    #largo= 512
    #dim1= (largo,ancho)
    img_baboon =cv.imread('baboon.jpg')
    #img_baboon=cv.resize(img_baboon,dim1)
    img_fondo= cv.imread('cielo.jpeg')
    mean = cv.mean(img_baboon) #promedio de una dimension
    print(mean)
    img_copy = img_baboon.copy()
    color = (0, 0, 255)
    thickness = 3
    circle_radius = 10
    cv.namedWindow('Baboon') #nombra la ventana 
   #low cv.namedWindow('Rectangulo')
    #cv.namedWindow('Mask')
    cv.setMouseCallback('Baboon', mouse_callback)
    #cv.setMouseCallback('Rectangulo', mouse_callback)
    cv.createTrackbar('r','Baboon',circle_radius, 100, trackbar_callback)
    #mask=cv.inRange(img_baboon,low,upp)
  
   
    while True:
       
        cv.imshow("Baboon", img_copy)
        if cv.waitKey(100) & 0xFF == 27: #esc en ascii es 27 
            break
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
     
img = cv.imread('baboon.jpg')        
cv.waitKey(0)


