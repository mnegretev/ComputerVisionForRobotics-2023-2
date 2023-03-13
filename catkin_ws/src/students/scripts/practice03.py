import cv2
import numpy
import math


#-----------------------------------FILTRO GAUSSIANO-----------------------------------------------------
def convolve2d(A, kernel):
    return cv2.convertScaleAbs(cv2.filter2D(A, cv2.CV_16S, kernel))
#--------------------------------------------------------------------------------------------------------


    
#------------------------------------------GENERAR EL KERNEL (Se proponen 4)-----------------------------
def get_gaussian_kernel(k,sigma):
    k = k//2
    H = numpy.zeros((2*k+1, 2*k+1))
    r,c = H.shape
    for i in range(r):
        for j in range(c):
            H[i,j] = 1.0/(2*math.pi*sigma*sigma)*math.exp(-((i-k)**2 + (j-k)**2)/(2*sigma*sigma))
    H /= numpy.sum(H)
    return H         
#-------------------------------------------------------------------------------------------------------



#------------------------------------------Filtro Sobel x , y and (x,y) --------------------------------
def get_sobel_x_gradient(A):# gradiente en x
    Gx = numpy.asarray([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])/8.0
    return convolve2d(A, Gx)

def get_sobel_y_gradient(A):# Gradiente en y
    Gy = numpy.asarray([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])/8.0
    return convolve2d(A, Gy)

def get_sobel_x_and_y_gradient(A):# Gradiente en y
    Gx = numpy.asarray([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])/8.0
    Gy = numpy.asarray([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])/8.0
    return convolve2d(A, Gx), convolve2d(A, Gy)
#-------------------------------------------------------------------------------------------------------


#------------------------------------------ Magnitud y angulo del gradiente ----------------------------
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
#-------------------------------------------------------------------------------------------------------


#------------------------------- Supresion de no maximos and Histeresis -------------------------------
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
    
def doble_Umbral(Gm,Umin,Umax):
    G = numpy.zeros(Gm.shape)
    r,c = Gm.shape
    for i in range(1,r-1):
        for j in range(1,c-1):
            if Gm[i,j] > Umax:
                G[i,j] = Gm[i,j] 
            elif Gm[i,j] >= Umin and Gm[i,j] <= Umax:
                if Gm[i+1, j] > 0  or Gm[i-1, j] > 0 :
                    G[i,j] = Gm[i,j] 
                elif Gm[i, j+1] > 0  or Gm[i, j-1] > 0 :
                    G[i,j] = Gm[i,j] 
                elif Gm[i+1, j+1] > 0  or Gm[i-1, j-1] > 0 :
                    G[i,j] = Gm[i,j] 
                else:
                    G[i,j] = 0
            else:
                G[i,j] = 0
    return G.astype(numpy.uint8)

def doble_Umbral_Saturado(Gm,Umin,Umax):
    G = numpy.zeros(Gm.shape)
    r,c = Gm.shape
    for i in range(1,r-1):
        for j in range(1,c-1):
            if Gm[i,j] > Umax:
                G[i,j] = 255
            elif Gm[i,j] >= Umin and Gm[i,j] <= Umax:
                if Gm[i+1, j] > 0  or Gm[i-1, j] > 0 :
                    G[i,j] = 220
                elif Gm[i, j+1] > 0  or Gm[i, j-1] > 0 :
                    G[i,j] = 220
                elif Gm[i+1, j+1] > 0  or Gm[i-1, j-1] > 0 :
                    G[i,j] = 220
                else:
                    G[i,j] = 0
            else:
                G[i,j] = 0
    return G.astype(numpy.uint8)
#-------------------------------------------------------------------------------------------------------




#------------------------------------------------ Filtro CANNY -----------------------------------------
def canny_border_detector(filtered, Umin, Umax):
    Gm, Ga = get_sobel_mag_angle(filtered)
    Gsnm = supress_non_maximum(Gm, Ga)
    G = doble_Umbral(Gsnm,Umin,Umax)
    return G
def canny_border_detector_Saturado(filtered, Umin, Umax):
    Gm, Ga = get_sobel_mag_angle(filtered)
    Gsnm = supress_non_maximum(Gm, Ga)
    G = doble_Umbral_Saturado(Gsnm,Umin,Umax)
    return G
#-------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------
#-----------------------------Transfomada Hough-------------------------------------------
    
def transformada_hough(img, umbral, d_min, d_res, theta_min, theta_max, theta_res):
    
    alto, ancho = img.shape
    
    d_max = int(math.sqrt(alto**2 + ancho**2))
    d_n = numpy.arange(d_min, d_max, d_res)
    theta_n = numpy.arange(theta_min, theta_max, theta_res)
    accumulator = numpy.zeros((len(d_n), len(theta_n)))

    for i in range(alto):
        for j in range(ancho):
            if img[i,j] > 0:
                for k in range(len(theta_n)):
                    d = int(((j * math.cos(theta_n[k]) + i * math.sin(theta_n[k])) - d_min)//d_res)
                    if(d >= 0 and d < len(d_n)):
                        accumulator[d, k] += 1
    lineas = []
    for i in range(len(d_n)):
        for j in range(len(theta_n)):
            if accumulator[i,j] > umbral:
                d = d_n[i]
                theta = theta_n[j]
                lineas.append((d, theta))
    return lineas 
#-------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------
#-----------------------------Trazar Lineas-------------------------------------------
    
def trazar_lineas(G, lines):
    img = numpy.zeros(G.shape)
    for x in range(0,len(lines)-1):
        rho, theta = lines[x]
        a = numpy.cos(theta)
        b = numpy.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return img
#-------------------------------------------------------------------------------------------------------


def Constante():
    return 0

#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------

def main():

    #----------------------------------------
    #--------- Video ------------------------
    cap  = cv2.VideoCapture(0)
    
    #------ Kernel y umbrales ---------------
    kernel = get_gaussian_kernel(5,1)
    kernel_1 = get_gaussian_kernel(7,1)
    kernel_2 = get_gaussian_kernel(3,1)
    
    pv = cv2.imread("Prueba.png");
    Pv= cv2.cvtColor(pv, cv2.COLOR_BGR2GRAY)
    PV = numpy.zeros(pv.shape)
    
    Umin = 5
    Umax = 255
    #------ Porcentaje de escala ------------
    scale_percent = 30
    
    #------ Variable auxiliar ---------------
    aux1 = 0
    Umbral = 50
    cv2.namedWindow('Transformada Hough')
    cv2.createTrackbar('Umbral','Transformada Hough',Umbral, 100, Constante)
   #-----------------------------------------
   
    while cap.isOpened():
    
    
    
        #--------- Video -----------------------------------------------------------------
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
       #------------------------------------------------------------------------------------
       
       #--------- Escala la imagen --------------------------------------------------------
        if aux1 == 0:
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            dsize = (width, height)
            aux1 = 1
        output = cv2.resize(frame, dsize)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        #------------------------------------------------------------------------------------
        
        
        #--------- Escala la imagen --------------------------------------------------------
        if aux1 == 1:
            filtered = convolve2d(output, kernel)
            Gx = get_sobel_x_gradient(filtered)
            Gy = get_sobel_y_gradient(filtered)
            cv2.imshow("Filtro Sobel en x Gx", Gx)
            cv2.imshow("Filtro Sobel en y Gy", Gy)
        elif aux1 == 2:
            filtered = convolve2d(output, kernel)
            Gm, Ga = get_sobel_mag_angle(filtered )
            cv2.imshow("Angulo del gradiente Ga", Ga)
            cv2.imshow("Magnitud del gradiente Gm", Gm)
        elif aux1 == 3:
            filtered = convolve2d(output, kernel)
            Gm, Ga = get_sobel_mag_angle(filtered )
            cv2.imshow("Angulo del gradiente Ga", Ga)
            cv2.imshow("Magnitud del gradiente Gm", Gm)
        elif aux1 == 4:
            filtered = convolve2d(output, kernel)
            Gm, Ga = get_sobel_mag_angle(filtered )
            G = supress_non_maximum(Gm, Ga)
            cv2.imshow("Supresion de no maximos", G)
        elif aux1 == 4:
            filtered = convolve2d(output, kernel)
            Gm, Ga = get_sobel_mag_angle(filtered )
            G = supress_non_maximum(Gm, Ga)
            Gf = doble_Umbral(G,Umin,Umax)
            cv2.imshow("Histeresis", Gf)
        elif aux1 == 5:
            filtered = convolve2d(output, kernel)
            G = canny_border_detector(filtered,Umin,Umax)
            cv2.imshow("Filtro Kanny", G)       
        elif aux1 == 6:
            filtered = convolve2d(output, kernel)
            G = canny_border_detector_Saturado(filtered,Umin,Umax)
            cv2.imshow("Filtro Kanny Saturado", G) 
        elif aux1 == 7:
            lines = transformada_hough(Pv, 200, 0,3, -numpy.pi/2, numpy.pi/2, numpy.pi/180)
            PV = trazar_lineas(PV,lines)
            cv2.imshow("PruebaHough", PV)
        elif aux1 >= 8:
            filtered = convolve2d(output, kernel)
            G = canny_border_detector_Saturado(filtered,10,255)
            GG = numpy.zeros(G.shape)
            lines = transformada_hough(G, 80, 0,3, -numpy.pi/2, numpy.pi/2, numpy.pi/180)
            GG = trazar_lineas(PV,lines)
            cv2.imshow("Transformada Hough", GG)
            
        else:
            cv2.imshow("Video", output)
        
        if aux1 <= 9:
           aux1 += 1
        #------------------------------------------------------------------------------------
        
        
        #--------- Fin  del programa------------------------------------------------------------ 
        if cv2.waitKey(10) & 0xFF == 27:
           break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
#-------------------------------------------------------------------------------------------------------

