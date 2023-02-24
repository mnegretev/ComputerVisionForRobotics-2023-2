import cv2
import numpy as np

def main():
    img_baboon = cv2.imread('gui.png')
    print(type(img_baboon))
    print(img_baboon.shape)
    img_blank  = np.zeros((512, 512, 3), np.uint8)
    img_blank[300,300] = [0,0,255]
    img_blank[128:256,0:256] = 255*np.ones((128, 256, 3), np.uint8)
    img_roi    = img_baboon[120:360, 160:480]
    cv2.imshow("BGR Image", img_baboon)
    cv2.imshow("Blank image", img_blank)
    cv2.imshow("Region of Interest", img_roi)
    cv2.waitKey(0) #Importante,
    #espera una tecla del usuario por cierto tiempo
    #En este caso al escribir 0 espera un tiempo infinito

if __name__ == '__main__':
    main()

