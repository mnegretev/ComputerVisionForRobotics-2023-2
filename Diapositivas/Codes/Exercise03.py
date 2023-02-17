#
# COMPUTER VISION FOR ROBOTICS - FI-UNAM - 2023-2
# Exercise 03 - GUI Controls
#

import cv2
import numpy as np

def mouse_callback(event, x, y, flags, param):  # cuando hizo click dibuja un circulo en las coordenadas que capturo
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
    img_copy   = img_baboon.copy()# <-----crea una copia     # img_baboon 
    circle_radius = 10
    cv2.namedWindow('Baboon')       # Crea una ventana
    cv2.setMouseCallback('Baboon', mouse_callback)  # dispara esa funcion en cada evento apuntador a funcion  porque no lleva parentesis
    cv2.createTrackbar('r','Baboon',circle_radius, 100, trackbar_callback)  # Pone una barra sobre la ventana baboon y 'r'
    while True:
        cv2.imshow("Baboon", img_copy)
        if cv2.waitKey(100) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

