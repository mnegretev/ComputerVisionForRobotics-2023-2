#
# COMPUTER VISION FOR ROBOTICS - FI-UNAM - 2023-2
# PRACTICE 01 - THE OPENCV LIBRARY
#

import numpy as np
import cv2
from statistics import mean

def Trackbar(val):
    global circle_radius
    circle_radius = val

def mouse_callback(event, x, y, flags, param):
    global cap,output,image_coordinates,image_coordinates2,clone

    if event == cv2.EVENT_LBUTTONDOWN:   
        image_coordinates =np.array([x,y])
        print('coordenadas1 :',image_coordinates)
            

    if event == cv2.EVENT_LBUTTONUP: 
        image_coordinates2=np.array([x,y])
        print('coordenadas2 :',image_coordinates2)

               

def main():
    global image_coordinates,image_coordinates2,clone
    circle_radius=10
    playa = cv2.imread('playa.jpg')
    
    cap = cv2.VideoCapture(0)
    output = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'MPEG'), 30, (512, 512))
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',mouse_callback)
    cv2.createTrackbar('r','image',circle_radius, 120, Trackbar)

    image_coordinates=tuple([-1,-1])
    image_coordinates2=tuple([-1,-1])
    
    while(True):
        
        cv2.setMouseCallback('image',mouse_callback)
        ret, frame = cap.read()
        clone=frame.copy()
        clone=cv2.resize(clone,(512,512))
        if(ret):
            ValorTrackbar= int(cv2.getTrackbarPos('r','image'))
           
            cv2.rectangle(clone, tuple(image_coordinates), tuple(image_coordinates2),(0, 255, 0), 2)
           
            Rectangulo=clone[image_coordinates[0]:image_coordinates[1],image_coordinates2[0]:image_coordinates2[1]]
            p0=cv2.mean(Rectangulo)[0]; p1=cv2.mean(Rectangulo)[1];  p2=cv2.mean(Rectangulo)[2]     ###HAY PROBLEMAS


            Umin1=p0-ValorTrackbar   
            Umax1=p0+ValorTrackbar

            Umin2=p1-ValorTrackbar  
            Umax2=p1+ValorTrackbar

            Umin3=p2-ValorTrackbar    
            Umax3=p2+ValorTrackbar

            rBajo = np.array([Umin1, Umin2, Umin3], np.uint8)
            rAlto = np.array([Umax1, Umax2, Umax3], np.uint8)
        

            BGR=cv2.inRange(clone,rBajo, rAlto)
            BGR=BGR.astype(np.uint8)
          
            img_not    = cv2.bitwise_not(BGR)
            resize=cv2.resize(playa,(512,512))
            fondo=cv2.bitwise_and(resize,resize,mask=BGR) 
            invertida=cv2.bitwise_and(clone,clone,mask=img_not)

            final=cv2.add(fondo,invertida)
            cv2.imshow('clone',final)
            cv2.imshow('image',clone)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
        else:
            break
    cv2.destroyAllWindows()
    output.release()
    cap.release()
  
  
if __name__ == "__main__":
    global image_coordinates
   
    main()
