#
# COMPUTER VISION FOR ROBOTICS - FI-UNAM - 2023-2
# PRACTICE 02 - BORDER DETECTION
#

import cv2
import numpy
import math

def convolve2d(A, kernel):
    return cv2.convertScaleAbs(cv2.filter2D(A, cv2.CV_16S, kernel))

def get_gaussian_kernel(k,sigma):
    k = k//2
    H = numpy.zeros((2*k+1, 2*k+1))
    r,c = H.shape
    for i in range(r):
        for j in range(c):
            H[i,j] = 1.0/(2*math.pi*sigma*sigma)*math.exp(-((i-k)**2 + (j-k)**2)/(2*sigma*sigma))
    H /= numpy.sum(H)
    return H

def get_sobel_x_gradient(A):
    Gx = numpy.asarray([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])/8.0
    return convolve2d(A, Gx)

def get_sobel_y_gradient(A):
    Gy = numpy.asarray([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])/8.0
    return convolve2d(A, Gy)

def get_sobel_mag_angle(A):
    Gx = get_sobel_x_gradient(A)
    Gy = get_sobel_y_gradient(A)
    Gm = numpy.zeros(Gx.shape)
    Ga = numpy.zeros(Gx.shape)
    r,c = Gx.shape
    for i in range(r):
        for j in range(c):
            Gm[i,j] = math.sqrt(Gx[i,j]**2 + Gy[i,j]**2)
            Ga[i,j] = math.atan2(Gy[i,j],Gx[i,j])
            if Ga[i,j] < 0:
                Ga[i,j] += math.pi
            Ga[i,j] = int(Ga[i,j]/math.pi*180)
    return Gm.astype(numpy.uint8), Ga.astype(numpy.uint8)

def supress_non_maximum(Gm, Ga):
    G = numpy.zeros(Gm.shape)
    r,c = Gm.shape
    for i in range(1,r-1):
        for j in range(1,c-1):
            if Ga[i,j] <= 22 or Ga[i,j] > 157:
                if Gm[i,j] >= Gm[i, j+1] and Gm[i,j] > Gm[i, j-1]:
                    G[i,j] = Gm[i,j]
                else:
                    G[i,j]=0

            elif Ga[i,j] > 22 and Ga[i,j] <= 67:
                if Gm[i,j] >= Gm[i-1, j+1] and Gm[i,j] > Gm[i+1, j-1]:
                    G[i,j] = Gm[i,j]
                else:
                    G[i,j]=0

            elif Ga[i,j] > 67 and Ga[i,j] <= 112:
                if Gm[i,j] >= Gm[i-1, j] and Gm[i,j] > Gm[i+1,j]:
                    G[i,j] = Gm[i,j]
                else:
                    G[i,j]=0

            else:       
                if Gm[i,j] >= Gm[i-1, j-1] and Gm[i,j] > Gm[i+1, j+1]:
                    G[i,j] = Gm[i,j]
                else:
                    G[i,j] = 0
    return G.astype(numpy.uint8)

def trackbar_callback(val):
    global circle_radius
    circle_radius = val

def trackbar_callback2(val2):
    global circle_radius2
    circle_radius2 = val2

def umbrales(G):
    z = numpy.ones(G.shape)
    r,c = G.shape

    Umax=int(cv2.getTrackbarPos('Umbral_Maximo','image'))
    Umin=int(cv2.getTrackbarPos('Umbral_Minimo','image'))

    for i in range(1,r-1):
        for j in range(1,c-1):
            if G[i,j]<Umax and G[i,j]>Umin:
                if G[i-1,j-1]==0 or G[i-1,j]==0 or G[i-1,j+1]==0 or G[i,j-1]==0 or G[i,j+1]==0 or G[i+1,j-1]==0 or G[i+1,j]==0 or G[i+1,j+1]==0:
                    z[i,j]=0
                else:
                    z[i,j]=255
            elif G[i,j]>Umax:
                z[i,j]=255

            elif G[i,j]<Umin:
                z[i,j]=0
    z=G*z
    return z
    
def main():
    global frame,clone, circle_radius,circle_radius2
    cap =cv2.VideoCapture(0)
    cv2.namedWindow('image') 
    circle_radius=1
    circle_radius2=2
    cv2.createTrackbar('Umbral_Minimo','image',circle_radius, 64, trackbar_callback)
    cv2.createTrackbar('Umbral_Maximo','image',circle_radius, 64, trackbar_callback2)
    
    while(True):
        ret, frame = cap.read()
        if(ret):
            
            clone=frame.copy()
            clone=cv2.resize(clone,(300,200))
            img = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
            kernel = get_gaussian_kernel(5,1)
            filtered = convolve2d(img, kernel)
            Gx = get_sobel_x_gradient(filtered)
            Gy = get_sobel_y_gradient(filtered)
            Gm, Ga = get_sobel_mag_angle(filtered)
            G = supress_non_maximum(Gm, Ga)
            z=umbrales(G)
            
            cv2.imshow("clone",clone)
            cv2.imshow("image",z)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
        else:
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()
