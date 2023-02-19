import cv2
import numpy as np


#---------------------------------------------
#----------------- Parte 2 -------------------
#____________ tolerancia y mouse______________

def mouse_callback(event, x, y, flags, param):
    global img_baboon, img_blank, img_copy, t_r, xi,yi,xf,yf,contador
    if event == cv2.EVENT_LBUTTONDOWN and contador == 0:
        xi = x
        yi = y
        contador = 1
    elif event == cv2.EVENT_LBUTTONDOWN and contador == 1:
        xf = x
        yf = y
        cv2.rectangle(img_copy,(xi,yi),(xf,yf),(255,255,255),-1)
        contador = 2
    elif event == cv2.EVENT_RBUTTONDOWN:
        img_copy = img_blank.copy()
        contador = 0

def trackbar_callback(val):
    global t_r
    t_r = val
#------------------------------------------------
#-------------Parte 3----------------------------
def Puntos():
    global xi,yi,xf,yf,pxi,pyi,pxf,pyf
    
    if xi < xf:
        if yi < yf:
            pxi = xi
            pyi = yi
            pxf = xf
            pyf = yf
        elif yi > yf:
            pxi = xi
            pyf = yi
            pxf = xf
            pyi = yf
    elif xi > xf:
        if yi < yf:
            pxf = xi
            pyi = yi
            pxi = xf
            pyf = yf
        elif yi > yf:
            pxf = xi
            pyf = yi
            pxi = xf
            pyi = yf
#-----------------------------------------------

def mascara ():
    global img_baboon, img_mask, t_r, xi,yi,xf,yf,pxi,pyi,pxf,pyf,contador, B_m, G_m, R_m
    rows = img_baboon.shape[0]
    cols = img_baboon.shape[1]
    
    tr = t_r
    for row in range(rows):
        for col in range(cols):

            if img_baboon[row,col][0] <= (B_m + tr) and img_baboon[row,col][0] >= (B_m - tr):
            	
            	if img_baboon[row,col][1] <= (G_m + tr):

                    if img_baboon[row,col][2] <= (R_m+ tr):
                        img_mask[row,col][0] = 255
                        img_mask[row,col][1] = 255
                        img_mask[row,col][2] = 255

#------------------------------------------------
#------------------Parte 1----------------------
def main():
    global img_baboon, img_blank, img_copy, img_mask, t_r, xi,yi,xf,yf,pxi,pyi,pxf,pyf,contador, B_m, G_m, R_m
   
    contador = 0
    t_r = 50
    
    img_baboonn = cv2.imread('baboonn.jpg')
    
    img_blank  = np.zeros((480,640,3), np.uint8)
    img_copy   = img_blank.copy()
    img_mask   = img_blank.copy()
    
    
    
    cv2.namedWindow('Baboon')
    cv2.setMouseCallback('Baboon', mouse_callback)
    cv2.createTrackbar('r','Baboon',t_r, 100, trackbar_callback)
    
    cap  = cv2.VideoCapture(0)
    while cap.isOpened():   
        ret, img_baboon = cap.read()
        if not ret:
        	print("Can't receive frame (stream end?). Exiting ...")
        	break
        if contador <= 1:
            cv2.imshow('Baboon', img_baboon)
        elif contador == 2:
             Puntos()
             img_roi    = img_baboon[pyi:pyf,pxi:pxf]
             (B, G, R) = cv2.split(img_roi)
             Valor = cv2.split(img_roi)
             B_m= np.median(B)
             G_m= np.median(G)
             R_m= np.median(R)
             ArrA = np.array ([B_m - t_r,G_m - t_r,R_m - t_r],np.uint8)
             ArrB = np.array ([B_m + t_r,G_m + t_r,R_m + t_r],np.uint8)       
             contador = 3
        elif contador == 3:
               ArrA = np.array ([B_m - t_r,G_m - t_r,R_m - t_r],np.uint8)
               ArrB = np.array ([B_m + t_r,G_m + t_r,R_m + t_r],np.uint8) 
               img_mask = cv2.inRange(img_baboon, ArrA, ArrB )
               img_not    = cv2.bitwise_not(img_mask)
               img_andd    = cv2.bitwise_and(img_baboon, img_baboon, mask = img_not)
               img_and    = cv2.bitwise_and(img_baboonn,img_baboonn, mask = img_mask)
               img_or     = cv2.bitwise_or (img_andd, img_and) 
               img_xor     = cv2.bitwise_xor (img_andd, img_and)
               cv2.imshow('Baboon', img_baboon)
               cv2.imshow("Imagen Binaria", img_mask )
               cv2.imshow("Solucion", img_xor )
        if cv2.waitKey(100) & 0xFF == 27:
                break
    cap.release() 
    cv2.destroyAllWindows()
#------------------------------------------------

if __name__ == '__main__':
    main()

