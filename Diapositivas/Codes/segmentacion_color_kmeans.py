#!/usr/bin/env python3

import cv2
import numpy as np

    


def main():

    img_bgr = cv2.imread('obj1.jpg')

    # ***************************************
    cv2.imshow('Imagen original', img_bgr)
    cv2.waitKey(0)

    img_copy = np.copy(img_bgr)
    

    # Convertiremos la imagen en un arreglo de ternas, las cuales representan el valor de cada pixel. En pocas palabras,
    # estamos aplanando la imagen, volviéndola un vector de puntos en un espacio 3D.
    pixel_values = img_bgr.reshape((-1,3)) 
    pixel_values = np.float32(pixel_values)


    # Image color quantification with k-medias
    criteria= ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,100,0.1)    # centroid convergence criterion
    k=7     # num of colors in image
    _ , labels , cc =cv2.kmeans(pixel_values ,
                                k ,
                                None,
                                criteria,
                                30,
                                cv2.KMEANS_RANDOM_CENTERS) 
    
    # convert into uint8, and make original image
    cc=np.uint8(cc) # Image with reduced colors, conversion to the format with which cv2 works
    segmented_image= cc[labels.flatten()]
    img_sgm = segmented_image.reshape(img_copy.shape)
    
    # Mostramos la imagen segmentada resultante.
    cv2.imwrite("obj2.jpeg",img_sgm)
    cv2.imshow('Imagen segmentada', img_sgm)
    cv2.waitKey(0)
    



if __name__ == '__main__':
    main()
