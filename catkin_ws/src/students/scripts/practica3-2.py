#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 20:57:30 2023

@author: cv
"""

import cv2
import numpy as np
import math


def line(image,edge_image, num_thetas=180):
  global rhos,d,drho,thetas,dtheta,accumulator,cos_thetas,sin_thetas,p,edge_points,x1,y1,x2,y2,pos
  global k,Max
  edge_height, edge_width = edge_image.shape[:2]
  #
  d = np.sqrt(np.square(edge_height) + np.square(edge_width)) #linea diagnoal
  d=np.ceil(d)
  dtheta = 180 / num_thetas
  drho=2 #resolucion de rhos

  #inicializando los intervalos de theta y rho
  thetas = np.arange(0, 180, step=dtheta) 
  rhos = np.arange(0, d, step=drho)
  accumulator = np.zeros((len(thetas),len(rhos)))#iniciamos el acomulador 
  #
  for i in range(0,edge_height):
        for j in range(0,edge_width):
           if edge_image[i,j] != 0:
              for k in range(0,len(thetas)):
                  rad=np.deg2rad(k)
                  p=i*np.cos(rad)+j*np.sin(rad)
                  p=np.ceil(p)
                  pos=int(np.ceil(p/drho))
                  if pos>0 and pos<len(rhos):
                     accumulator[k,pos]=accumulator[k,pos]+1
                     edge_points = np.argwhere(accumulator != 0)
#tranformando si el umbral es mayor 
  for l in range(0,len(thetas)):
        for m in range(0,len(rhos)):
           if accumulator[l,m]>Max:  
               rho= rhos[m]
               theta=thetas[l]
               rad2=np.deg2rad(theta)
               a=np.cos(rad2)
               b=np.sin(rad2)
               x0=a*rho
               y0=b*rho
               x1 = int(x0 + 1000 * (-b))
               y1 = int(y0 + 1000 * (a))
               x2 = int(x0 - 1000 * (-b))
               y2 = int(y0 - 1000 * (a))
               color=(0,255,0)
               punto1=(x1,y1)
               punto2=(x2,y2)
               lineas=cv2.line(image,punto1,punto2,color)
               image=lineas


def trackbar_callback(val):
    global Max
    Max = val               
   

if __name__ == '__main__':
    global rhos,x1,y1,x2,y2,Max
    x1=0
    x2=0
    y1=0
    y2=0
    Max=50
    num_thetas=180
    t_count=50
    image = cv2.imread("TestLines.png")
    image=cv2.resize(image,(256,256))
    cv2.imshow('image1',image)
    cv2.waitKey(0)
    edge_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_image = cv2.GaussianBlur(edge_image, (3, 3), 1)
    edge_image = cv2.Canny(edge_image, 100, 200)
    cv2.namedWindow('TH') 
    cv2.createTrackbar('UMBRAL','TH',30, 200, trackbar_callback)
    line(image,edge_image,num_thetas)
    punto1=(x1,y1)
    punto2=(x2,y2)
    #color=(0,255,0)
    cv2.imshow('TH',image)
    #lineas=cv2.line(image,punto1,punto2,color)
    #cv2.imshow('linea',lineas)
    cv2.waitKey(0)
