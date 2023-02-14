#
# COMPUTER VISION FOR ROBOTICS - FI-UNAM - 2023-2
# Exercise 03 - GUI Controls
#

import cv2
import numpy as np

def mouse_callback(event, x, y, flags, param):
    global img_baboon, img_copy, circle_radius
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img_copy, (x,y), circle_radius, (0,255,0),-1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        img_copy = img_baboon.copy()

def trackbar_callback(val):
    global circle_radius
    circle_radius = val
        
def main():
    global img_baboon, img_copy, circle_radius
    img_baboon = cv2.imread('baboon.jpg')
    img_copy   = img_baboon.copy() #Si se ejecuta imig_copy=img_baboon
    #En ese caso se copia la referencia del apuntador
    #Usando .copy() se puede crear una copia real de cada pixel
    #De forma que al modificar la copia no se altera el original
    circle_radius = 10
    cv2.namedWindow('Baboon')
    cv2.setMouseCallback('Baboon', mouse_callback) #captura el mouse
    #Dispara la función
    cv2.createTrackbar('r','Baboon',circle_radius, 100, trackbar_callback)
    #circle_radius es la varibale que guarda el valor de la barra
    while True:
        cv2.imshow("Baboon", img_copy)
        if cv2.waitKey(100) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

