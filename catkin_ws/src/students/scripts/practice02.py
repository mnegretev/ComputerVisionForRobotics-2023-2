import cv2
import numpy as np
import math

# Función para el filtro Gaussiano
def convolve2d(A, kernel):
    return cv2.convertScaleAbs(cv2.filter2D(A, -1, kernel))

# Función para obtener el kernel Gaussiano
def get_gaussian_kernel(k, sigma):
    k = k // 2
    H = np.zeros((2 * k + 1, 2 * k + 1))
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            H[i + k, j + k] = (1.0 / (2 * math.pi * sigma ** 2)) * math.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2))
    H /= np.sum(H)
    return H

# Función para obtener los gradientes x, y
def sobel_filters(img):
    #Kx = np.array([[-1/8, 0, 1/8], [-2/8, 0, 2/8], [-1/8, 0, 1/8]], np.float32)
    #Ky = np.array([[1/8, 2/8, 1/8], [0, 0, 0], [-1/8, -2/8, -1/8]], np.float32)
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Gx = convolve2d(img, Kx)
    Gy = convolve2d(img, Ky)
    
    return Gx, Gy

# Función para obtener la magnitud y el ángulo
def get_magnitude_and_angle(Gx, Gy):
    Gm = np.hypot(Gx, Gy)
    Ga = np.arctan2(Gy, Gx)
    return Gm, Ga

# Función para aplicar la supresión de no máximo
def non_max_suppression(Gm, Ga):
    M, N = Gm.shape
    Ga = Ga * 180. / np.pi
    Ga[Ga < 0] += 180
    edgeImg = np.zeros((M, N), dtype=np.int32)
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255
                
               #angle 0
                if (0 <= Ga[i,j] < 22.5) or (157.5 <= Ga[i,j] <= 180):
                    q = Gm[i, j+1]
                    r = Gm[i, j-1]
                #angle 45
                elif 22.5 <= Ga[i,j] < 67.5:
                    q = Gm[i+1, j-1]
                    r = Gm[i-1, j+1]
                #angle 90
                elif 67.5 <= Ga[i,j] < 112.5:
                    q = Gm[i+1, j]
                    r = Gm[i-1, j]
                #angle 135
                elif 112.5 <= Ga[i,j] < 157.5:
                    q = Gm[i-1, j-1]
                    r = Gm[i+1, j+1]

                if (Gm[i,j] >= q) and (Gm[i,j] >= r):
                    edgeImg[i, j] = Gm[i, j]
                else:
                    edgeImg[i, j] = 0

            except IndexError as e:
                pass
    return edgeImg

# Función para doble umbral y detección de bordes
def threshold(img, lowThresholdRatio=0.1, highThresholdRatio=0.05):
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(25)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)

# Rastreo de bordes por histéresis
def hysteresis(img, weak, strong=255):
    M, N = img.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

# Función sin hacer nada para la inicialización de las Trackbars
def nothing(x):
    pass

# Función principal
def main():
    # Captura de video desde la cámara
    cap = cv2.VideoCapture(0)

    # Obtener la resolución original del fotograma
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Definir la nueva resolución más baja
    reduced_width = int(frame_width/3)
    reduced_height = int(frame_height/3)

    # Crear las trackbars para los parámetros
    cv2.namedWindow('Control', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('K', 'Control', 1, 10, nothing)
    cv2.createTrackbar('Sigma', 'Control', 1, 10, nothing)
    cv2.createTrackbar('Lower Threshold Ratio', 'Control', 5, 100, nothing)
    cv2.createTrackbar('Upper Threshold Ratio', 'Control', 9, 100, nothing)

    while True:
        # Leer el video frame por frame
        ret, frame = cap.read()
        if not ret:
            break
        # Redimensionar el fotograma a una resolución más baja
        resized_frame = cv2.resize(frame, (reduced_width, reduced_height))
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        # Leer los valores de las trackbars
        k = round(cv2.getTrackbarPos('K', 'Control') / 2)
        sigma = round((cv2.getTrackbarPos('Sigma', 'Control') + 1) / 2)
        sigma = sigma if sigma != 0 else 0.1  # Asegurarse de que sigma nunca sea cero
        lower_threshold_ratio = (cv2.getTrackbarPos('Lower Threshold Ratio', 'Control') / 100) / 2
        upper_threshold_ratio = (cv2.getTrackbarPos('Upper Threshold Ratio', 'Control') / 100) / 2

        # Garantizar que K sea impar
        k = 2 * k + 1

        # Obtener el kernel Gaussiano
        kernel = get_gaussian_kernel(k, sigma)

        # Obtener la imagen filtrada
        filtered_image = convolve2d(gray, kernel)

        # Obtener los gradientes x, y
        Gx, Gy = sobel_filters(filtered_image)

        Gx_resized = cv2.resize(Gx, (reduced_width, reduced_height))
        Gy_resized = cv2.resize(Gy, (reduced_width, reduced_height))

        # Obtener la magnitud y el ángulo
        Gm, Ga = get_magnitude_and_angle(Gx, Gy)

        # Aplicar la supresión de no máximo
        edges = non_max_suppression(Gm, Ga)

        # Aplicar el umbral doble y la detección de bordes
        thresholded_edges, weak, strong = threshold(edges, lower_threshold_ratio, upper_threshold_ratio)
        hysteresis_edges = hysteresis(thresholded_edges, weak, strong)

        hysteresis_edges_resized = cv2.resize(hysteresis_edges, (reduced_width, reduced_height))

        # Mostrar las imágenes
        cv2.imshow('Original', resized_frame)
        cv2.imshow('Filtrada', filtered_image)
        cv2.imshow('Gradiente X', cv2.convertScaleAbs(Gx_resized))
        cv2.imshow('Gradiente Y', cv2.convertScaleAbs(Gy_resized))
        cv2.imshow('Bordes', cv2.convertScaleAbs(hysteresis_edges_resized))
        
        # Si se presiona la tecla Esc, terminar el programa
        if cv2.waitKey(1) == 27:
            break

    # Liberar la cámara y cerrar todas las ventanas
    cap.release()
    cv2.destroyAllWindows()

# Llamar a la función principal
if __name__ == "__main__":
    main()
