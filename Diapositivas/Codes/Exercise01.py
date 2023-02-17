import cv2
import numpy as np

def main():
    img_baboon = cv2.imread('baboon.jpg')
    img_blank  = np.zeros((512, 512, 3), np.uint8)
    #img_blank[255,255] = [255,0,255]
    # Incluye de 128 a 256,   de 0 a 256
    # genera matriz de menor tamanio y cada pixel es blanco (255)
    img_blank[128:256,0:256] = 255*np.ones((128, 256, 3), np.uint8)
    img_roi    = img_baboon[120:360, 160:480]
    cv2.imshow("BGR Image", img_baboon)
    cv2.imshow("Blank image", img_blank)
    cv2.imshow("Region of Interest", img_roi)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()

