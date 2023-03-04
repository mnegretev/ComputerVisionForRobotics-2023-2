import cv2
import numpy
import math

def trackbar_callback_3(val):
    global k_size
    k_size=val
    return

def trackbar_callback_2(val):
    global t2
    t2 = val
    return

def trackbar_callback_1(val):
    global t1
    t1 = val
    return


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
                di, dj = 0, 1
            elif Ga[i,j] > 22 and Ga[i,j] <= 67:
                di, dj = 1, 1
            elif Ga[i,j] > 67 and Ga[i,j] <= 112:
                di, dj = 1, 0
            else:
                di, dj = 1, -1
            if Gm[i,j] >= Gm[i+di, j+dj] and Gm[i,j] > Gm[i-di, j-dj]:
                G[i,j] = Gm[i,j]
            else:
                G[i,j] = 0
    return G.astype(numpy.uint8)

def threshold(G, t_1, t_2):
    T = numpy.zeros(G.shape)
    r,c = G.shape
    for i in range(1,r-1):
        for j in range(1,c-1):
            if G[i,j]>t_1 and G[i,j]<=t_2:
                T[i,j]=60
            elif G[i,j] > t_2 :
                T[i,j]=255
    return T.astype(numpy.uint8)

def final_supress(T):
    F = numpy.zeros(T.shape)
    r,c = T.shape
    for i in range(1,r-1):
        for j in range(1,c-1):
            if T[i,j]==60:
                if(T[i+1,j]==255 or T[i+1,j+1]==255 or T[i+1,j-1]==255 or T[i-1,j]==255 or T[i-1,j+1]==255 or T[i-1,j-1]==255 or T[i,j+1]==255 or T[i,j-1]==255 ):
                    F[i,j]=255
                else:
                    F[i,j]=0
            elif T[i,j]==255 :
                F[i,j]=255
    return F.astype(numpy.uint8)


def canny_border_detector(A, k_size, sigma, low_th, high_th):
    filtered = convolve2d(A, get_gaussian_kernel(k_size, sigma))
    Gm, Ga = get_sobel_mag_angle(filtered)
    G = supress_non_maximum(Gm, Ga)
    return

def main():
    global k_size, t1, t2
    k_size=7
    t1=1
    t2=4


    while True:
        cv2.createTrackbar('Umbral 1','Original', t1, 10, trackbar_callback_1)
        cv2.createTrackbar('Umbral 2','Original', t2, 20, trackbar_callback_2)
        cv2.createTrackbar('Kernel size','Original', k_size, 20, trackbar_callback_3)
        img = cv2.imread("Lizard.jpg")
        #img =cv2.imread("tag.PNG")
        img = cv2.resize(img, None, fx = 0.30, fy = 0.30, interpolation = cv2.INTER_CUBIC)
        img_2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img_2 = cv2.resize(img, None, fx = 0.20, fy = 0.20, interpolation = cv2.INTER_CUBIC)
        kernel = get_gaussian_kernel(k_size,1.4)
        filtered = convolve2d(img_2, kernel)
        Gx = get_sobel_x_gradient(filtered)
        Gy = get_sobel_y_gradient(filtered)
        Gm, Ga = get_sobel_mag_angle(filtered)
        G = supress_non_maximum(Gm, Ga)
        T=threshold(G,t1,t2)
        F=final_supress(T)
        cv2.imshow("Original", img)
        #cv2.imshow("Filtered", filtered)
        #cv2.imshow("Gx", Gx)
        #cv2.imshow("Gy", Gy)
        #cv2.imshow("G mag", Gm)
        #cv2.imshow("G ang", Ga)
        #cv2.imshow("G supressed", G)
        #cv2.imshow("threshold",T)
        cv2.imshow("Final cany",F)
        cv2.waitKey(10)
        if cv2.waitKey(100) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
