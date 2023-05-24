import cv2
import numpy as np


def shirt_color(pixels_head):   # Recibe las coordenadas de pixeles donde debe realizar ls detección de color
    # Establece los limites en valores HSV para cada color
    # Rojo
    redmin1 = np.array([0, 100, 20],      np.uint8)
    redmax1 = np.array([11, 255, 255],    np.uint8)
    redmin2 = np.array([175, 100, 20],    np.uint8)
    redmax2 = np.array([179, 255, 255],   np.uint8)
    # naranja
    orangemin = np.array([13, 100, 20],  np.uint8)
    orangemax = np.array([20, 255, 255], np.uint8)
    # amarillo
    yellowmin = np.array([25, 100, 20],  np.uint8)
    yellowmax = np.array([35, 255, 255], np.uint8)
    # verde
    greenmin = np.array([36, 100, 20],   np.uint8)
    greenmax = np.array([70, 255, 255],  np.uint8)
    # azul
    bluemin = np.array([80, 100, 20],    np.uint8)
    bluemax = np.array([125, 255, 255],   np.uint8)
    # morado
    purplemin = np.array([130, 100, 50],  np.uint8)
    purplemax = np.array([145, 255, 255], np.uint8)
    # rosa
    pinkmin = np.array([146, 100, 100],    np.uint8)
    pinkmax = np.array([160, 255, 255],   np.uint8)
    # gris
    graymin = np.array([0, 0, 0],    np.uint8)
    graymax = np.array([0, 0, 140],   np.uint8)
    # negro
    blackmin = np.array([0, 0, 0],   np.uint8)
    blackmax = np.array([0, 0, 45],  np.uint8)
    # blanco
    whitemin = np.array([0, 0, 0],   np.uint8)
    whitemax = np.array([180, 10, 255],  np.uint8)

    offsetx, offsety = 10,10
    pixels_shirt= (pixels_head[0]+offsetx , pixels_head[1]+offsety )
    img_bgr = cv2.imread('mujer_con_playera.jpg')
    mask = np.zeros(img_bgr.shape, dtype=np.uint8)

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    radio_5cm = 30 
    cv2.circle(mask, pixels_shirt, radio_5cm, (250), -1)    # circulo en la mascara

    # - Cambia el espacio de color de BGR a HSV.

    img_hsv = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2HSV)
    img_with_mask = cv2.bitwise_and(img_bgr,img_bgr, mask=mask)

    # - Determinar los pixeles cuyo color esta en el rango de color seleccionado.
    #   Deteccion del color: creamos una mascara que contiene solo los colores definidos en los limites
    #   Regresa imagen binaria: pixeles blancos si entro en el rango, sino pixeles negros.
    img_bin = cv2.inRange(img_with_mask, greenmin , greenmax)

    if img_bin[0,0] :
        print("Is green")

    cv2.imshow("BGR Image", img_bgr)
    cv2.imshow("mask", mask) 
    cv2.imshow("bin", img_bin)   
    cv2.imshow("shirt color", img_with_mask)

    cv2.waitKey(0)




def main():
    pixels_head = [320,120] # 640, 480 ancho, largo
    shirt_color(pixels_head)

if __name__ == '__main__':
    main()

