import cv2
import numpy
import math


#-----------------------------------FILTRO GAUSSIANO-----------------------------------------------------
def convolve2d(A, kernel):
    return cv2.convertScaleAbs(cv2.filter2D(A, cv2.CV_32F, kernel))

#--------------------------------------------------------------------------------------------------------

#------------------------------------------GENERAR EL KERNEL --------------------------------------------
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

#------------------------------------------Filtro Sobel x , y  -----------------------------------------
def get_sobel_x_gradient(A):# gradiente en x
    Gx = numpy.zeros((3, 3, cv2.CV_32FC1))
    Gx = numpy.asarray([[-1.0, 0.0, 1.0],[-2.0, 0.0, 2.0],[-1.0, 0.0, 1.0]])/8.0
    return convolve2d(A, Gx)

def get_sobel_y_gradient(A):# Gradiente en y
    Gy = numpy.zeros((3, 3, cv2.CV_32FC1))
    Gy = numpy.asarray([[-1.0, -2.0, -1],[0.0, 0.0, 0.0],[1.0, 2.0, 1.0]])/8.0
    return convolve2d(A, Gy)
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
                    G[i,j] = 255
                elif Gm[i, j+1] > 0  or Gm[i, j-1] > 0 :
                    G[i,j] = 255
                elif Gm[i+1, j+1] > 0  or Gm[i-1, j-1] > 0 :
                    G[i,j] = 255
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
#-----------------------------Transfomada Hough---------------------------------------------------------
    
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
#--------------------------------------- Trazar Lineas -------------------------------------------------
    
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



#-------------------------------------------------------------------------------------------------------
#------------------------ Detector de esquinas de Harry ------------------------------------------------

#-------------------------------------------------------------------------------------------------------
def convolve2dh(A, kernel):
    return cv2.filter2D(A, cv2.CV_32F, kernel)
#-------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------
def get_sobel_x_gradienth(A):# gradiente en x
    Gx = numpy.asarray([[-1.0, 0.0, 1.0],[-2.0, 0.0, 2.0],[-1.0, 0.0, 1.0]], numpy.float32)
    return convolve2dh(A, Gx)
#-------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------
def get_sobel_y_gradienth(A):# Gradiente en y
    Gy = numpy.asarray([[-1.0, -2.0, -1],[0.0, 0.0, 0.0],[1.0, 2.0, 1.0]], numpy.float32)
    return convolve2dh(A, Gy)
#-------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------
def matrix_segundo_momento(A,W_size):
    
    w = W_size//2
    Gx = get_sobel_x_gradienth(A)
    Gy = get_sobel_y_gradienth(A)
   
    r,c = A.shape
    M = numpy.zeros((r,c,4))
    M = numpy.float32(M)
    
    for i in range(w, r-w):
        for j in range(w, c-w):
            for k1 in range(i-w, i+w+1):
                for k2 in range(j-w, j+w+1):
                    M[i,j,0] += Gx [k1,k2]**2
                    M[i,j,1] += Gx [k1,k2]*Gy [k1,k2]
                    M[i,j,2] += Gx [k1,k2]*Gy [k1,k2]
                    M[i,j,3] += Gy [k1,k2]**2
    cv2.imshow("Derivada en x", Gx)
    cv2.imshow("Derivada en y", Gy)
    return M
#-------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------
def valores_propios(M):
    
    r,cc,e = M.shape
    lambdaa = numpy.zeros((r,cc,2))
    lambdaa = numpy.float32(lambdaa)
    b = 0.00
    c = 0.00
    b = numpy.float32(b)
    c = numpy.float32(c)
    
    for i in range(0, r):
        for j in range(0, cc):
            
            m = M[i,j]
            
            b = - m[0] - m[3]  
            c =  (m[0] * m[3] ) + (m[1] * m[2] )
            
            lambdaa[i,j,0] = (-b + numpy.sqrt(b**2 - 4*c))/2
            lambdaa[i,j,1] = (-b - numpy.sqrt(b**2 - 4*c))/2
    return lambdaa
#-------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------
def respuesta_Harris(lambdaa, k):
    
    r,c,e= lambdaa.shape
    R =  numpy.zeros((r,c))
    R = numpy.float32(R)
    
    for i in range(0, r):
        for j in range(0, c):
            l1 = lambdaa[i,j,0]
            l2 = lambdaa[i,j,1]
            R[i,j] = l1 * l2 - k * (l1 + l2) ** 2
    return R
#-------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------
def supresion_de_no_maximos_harris(R, W_size):
    
    w = W_size//2
    r,c= R.shape
    H = numpy.zeros(R.shape)
    for i in range(0, r):
        for j in range(0, c):
            
            if R[i,j] < 0.5:
                continue
            max_v = -999999
            max_v = numpy.float32(max_v)
            for k1 in range(i-w, i+w+1):
                for k2 in range(j-w, j+w+1):
                    if k1 >= 0  and k1< r and k2 >= 0  and k2 < c: 
                        if  R[k1,k2] > max_v :
                            max_v = R[k1,k2]
            if  max_v == R[i,j] :
                H[i,j] = 255
            else :
                H[i,j] = 0
    return H
#-------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------
def corners_harris(A, W_size,k):

    M = matrix_segundo_momento(A,W_size)
    lambdaa = valores_propios(M)
    R = respuesta_Harris(lambdaa, k)
    H = supresion_de_no_maximos_harris(R, W_size)
    
    cv2.imshow("2do Momento", M)
    cv2.imshow("respuesta H", R)
    cv2.imshow("Sup no max", H)
    
    corner = cv2.findNonZero(H)
    
    return corner
#-------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------
def draw_corners(img, corners):
    
    H = numpy.zeros(img.shape)
    for i in range(len(corners)):
       cv2.circle(img, tuple(corners[i][0]),1,(0,0,255),-1)
    H = img
    return H

def draw_cornersG(img, corners):
    
    H = numpy.zeros(img.shape)
    for i in range(len(corners)):
       cv2.circle(img, tuple(corners[i][0]),4,(0,0,255),-1)
    H = img
    return H
#-------------------------------------------------------------------------------------------------------  
    

def main():

    #----------------------------------------
    #--------- Video ------------------------
    cap  = cv2.VideoCapture(0)
    #------ Kernel y umbrales ---------------
    kernel = get_gaussian_kernel(5,1)
    #----------------------------------------
    Umin = 5
    Umax = 255
    #------ Porcentaje de escala ------------
    scale_percent = 30
    #------ Variable auxiliar ---------------
    aux1 = 0
    #----------------------------------------
    Prueba = cv2.imread("TestCorners.png");
    PruebaCopy = Prueba.copy()
    Pruebaceros = numpy.zeros(Prueba.shape)
    
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
            aux1 = 7
        output = cv2.resize(frame, dsize)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        #------------------------------------------------------------------------------------
        
        
        #--------- Escala la imagen --------------------------------------------------------
        if aux1 == 7:
            img = cv2.cvtColor(PruebaCopy, cv2.COLOR_BGR2GRAY)      
            img = numpy.float32(img)
            img = img/255.0
            corner = corners_harris(img, 3,50/1000.0)
            H = draw_cornersG(Prueba, corner)
            cv2.imshow("Prueba Harris", H)
        elif aux1 >= 8:
            cv2.imshow("Video", output)
            
            filtered = convolve2d(output, kernel)
            G = canny_border_detector_Saturado(filtered,10,255)
            cv2.imshow("Kanny Saturado", G)
            
            GG = numpy.zeros(G.shape)
            lines = transformada_hough(G, 40, 0,3, -numpy.pi/2, numpy.pi/2, numpy.pi/180)
            GG = trazar_lineas(GG,lines)
            GG = GG.astype(numpy.uint8)
            cv2.imshow("Lineas", GG)
              
            Gp = G.copy()
            img_1 = cv2.cvtColor(G, cv2.COLOR_GRAY2BGR)      
            img = numpy.float32(Gp)
            cv2.imshow("Imagen en escala de grises", img)
            img = img/255.0
            corner = corners_harris(img, 3, 246/1000.0)
            H = draw_corners(img_1, corner)
            cv2.imshow("Sol Harris", H)
            
        else:
            cv2.imshow("Video", output)
        
        if aux1 <= 8:
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
#-------------------------------------------------------------------------------------------------------
