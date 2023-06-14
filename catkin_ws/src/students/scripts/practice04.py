import cv2
import numpy
import math

# Función para convolucionar una matriz con un kernel
def convolve2d(A, kernel):
    return cv2.convertScaleAbs(cv2.filter2D(A, cv2.CV_32F, kernel))


# Función para obtener el kernel de la función Gaussiana
def get_gaussian_kernel(k, sigma):
    k = k // 2
    H = numpy.zeros((2 * k + 1, 2 * k + 1))
    r, c = H.shape
    for i in range(r):
        for j in range(c):
            H[i, j] = 1.0 / (2 * math.pi * sigma * sigma) * math.exp(-((i - k) ** 2 + (j - k) ** 2) / (2 * sigma * sigma))
    H /= numpy.sum(H)
    return H


# Funciones para obtener el gradiente de la imagen usando los operadores de Sobel
def get_sobel_x_gradient(A):
    Gx = numpy.zeros((3, 3, cv2.CV_32FC1))
    Gx = numpy.asarray([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
    return convolve2d(A, Gx)


def get_sobel_y_gradient(A):
    Gy = numpy.zeros((3, 3, cv2.CV_32FC1))
    Gy = numpy.asarray([[-1.0, -2.0, -1], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]) / 8.0
    return convolve2d(A, Gy)


# Función para obtener la magnitud y el ángulo del gradiente de la imagen
def get_magnitude_and_angle(A):
    Gx = get_sobel_x_gradient(A)
    Gy = get_sobel_y_gradient(A)
    Gm = numpy.zeros(Gx.shape)
    Ga = numpy.zeros(Gx.shape)
    r, c = Gx.shape
    for i in range(r):
        for j in range(c):
            Gm[i, j] = math.sqrt(Gx[i, j] ** 2 + Gy[i, j] ** 2)
            Ga[i, j] = math.atan2(Gy[i, j], Gx[i, j])
            if Ga[i, j] < 0:
                Ga[i, j] += math.pi
            Ga[i, j] = int(Ga[i, j] / math.pi * 180)
    return Gm.astype(numpy.uint8), Ga.astype(numpy.uint8)


# Función para suprimir los no máximos en la imagen del gradiente
def non_max_suppression(Gm, Ga):
    G = numpy.zeros(Gm.shape)
    r, c = Gm.shape
    for i in range(1, r - 1):
        for j in range(1, c - 1):
            if Ga[i, j] <= 22 or Ga[i, j] > 157:
                di, dj = 0, 1
            elif Ga[i, j] > 22 and Ga[i, j] <= 67:
                di, dj = 1, 1
            elif Ga[i, j] > 67 and Ga[i, j] <= 112:
                di, dj = 1, 0
            else:
                di, dj = 1, -1
            if Gm[i, j] >= Gm[i + di, j + dj] and Gm[i, j] > Gm[i - di, j - dj]:
                G[i, j] = Gm[i, j]
            else:
                G[i, j] = 0
    return G.astype(numpy.uint8)


# Funciones para aplicar el umbral doble en la imagen del gradiente
def doble_Umbral(Gm, Umin, Umax):
    G = numpy.zeros(Gm.shape)
    r, c = Gm.shape
    for i in range(1, r - 1):
        for j in range(1, c - 1):
            if Gm[i, j] > Umax:
                G[i, j] = Gm[i, j]
            elif Gm[i, j] >= Umin and Gm[i, j] <= Umax:
                if Gm[i + 1, j] > 0 or Gm[i - 1, j] > 0:
                    G[i, j] = Gm[i, j]
                elif Gm[i, j + 1] > 0 or Gm[i, j - 1] > 0:
                    G[i, j] = Gm[i, j]
                elif Gm[i + 1, j + 1] > 0 or Gm[i - 1, j - 1] > 0:
                    G[i, j] = Gm[i, j]
                else:
                    G[i, j] = 0
            else:
                G[i, j] = 0
    return G.astype(numpy.uint8)


def umbral_config(Gm, Umin, Umax):
    G = numpy.zeros(Gm.shape)
    r, c = Gm.shape
    for i in range(1, r - 1):
        for j in range(1, c - 1):
            if Gm[i, j] > Umax:
                G[i, j] = 255
            elif Gm[i, j] >= Umin and Gm[i, j] <= Umax:
                if Gm[i + 1, j] > 0 or Gm[i - 1, j] > 0:
                    G[i, j] = 255
                elif Gm[i, j + 1] > 0 or Gm[i, j - 1] > 0:
                    G[i, j] = 255
                elif Gm[i + 1, j + 1] > 0 or Gm[i - 1, j - 1] > 0:
                    G[i, j] = 255
                else:
                    G[i, j] = 0
            else:
                G[i, j] = 0
    return G.astype(numpy.uint8)


# Función para aplicar el detector de bordes de Canny
def canny_border_detector(filtered, Umin, Umax):
    Gm, Ga = get_magnitude_and_angle(filtered)
    Gsnm = non_max_suppression(Gm, Ga)
    G = doble_Umbral(Gsnm, Umin, Umax)
    return G


def Canny_config(filtered, Umin, Umax):
    Gm, Ga = get_magnitude_and_angle(filtered)
    Gsnm = non_max_suppression(Gm, Ga)
    G = umbral_config(Gsnm, Umin, Umax)
    return G

# Funciones para aplicar el detector de esquinas de Harris
def convolve2dh(A, kernel):
    return cv2.filter2D(A, cv2.CV_32F, kernel)


def get_sobel_x(A):
    Gx = numpy.asarray([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], numpy.float32)
    return convolve2dh(A, Gx)


def get_sobel_y(A):
    Gy = numpy.asarray([[-1.0, -2.0, -1], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], numpy.float32)
    return convolve2dh(A, Gy)


def second_moment_matrix(A, W_size):
    w = W_size // 2
    Gx = get_sobel_x(A)
    Gy = get_sobel_y(A)

    r, c = A.shape
    M = numpy.zeros((r, c, 4))
    M = numpy.float32(M)

    for i in range(w, r - w):
        for j in range(w, c - w):
            for k1 in range(i - w, i + w + 1):
                for k2 in range(j - w, j + w + 1):
                    M[i, j, 0] += Gx[k1, k2] ** 2
                    M[i, j, 1] += Gx[k1, k2] * Gy[k1, k2]
                    M[i, j, 2] += Gx[k1, k2] * Gy[k1, k2]
                    M[i, j, 3] += Gy[k1, k2] ** 2
    return M


def eigenvalues(M):
    r, cc, e = M.shape
    lambdaa = numpy.zeros((r, cc, 2))
    lambdaa = numpy.float32(lambdaa)
    b = 0.00
    c = 0.00
    b = numpy.float32(b)
    c = numpy.float32(c)

    for i in range(0, r):
        for j in range(0, cc):
            m = M[i, j]
            b = -m[0] - m[3]
            c = (m[0] * m[3]) + (m[1] * m[2])
            lambdaa[i, j, 0] = (-b + numpy.sqrt(b ** 2 - 4 * c)) / 2
            lambdaa[i, j, 1] = (-b - numpy.sqrt(b ** 2 - 4 * c)) / 2
    return lambdaa


def Harris_response(lambdaa, k):
    r, c, e = lambdaa.shape
    R = numpy.zeros((r, c))
    R = numpy.float32(R)
    for i in range(0, r):
        for j in range(0, c):
            l1 = lambdaa[i, j, 0]
            l2 = lambdaa[i, j, 1]
            R[i, j] = l1 * l2 - k * (l1 + l2) ** 2
    return R


def suppress_non_maxima_harris(R, W_size):
    w = W_size // 2
    r, c = R.shape
    H = numpy.zeros(R.shape)
    for i in range(0, r):
        for j in range(0, c):
            if R[i, j] < 0.5:
                continue
            max_v = -999999
            max_v = numpy.float32(max_v)
            for k1 in range(i - w, i + w + 1):
                for k2 in range(j - w, j + w + 1):
                    if k1 >= 0 and k1 < r and k2 >= 0 and k2 < c:
                        if R[k1, k2] > max_v:
                            max_v = R[k1, k2]
            if max_v == R[i, j]:
                H[i, j] = 255
            else:
                H[i, j] = 0
    return H


def corners_harris(A, W_size, k):
    M = second_moment_matrix(A, W_size)
    lambdaa = eigenvalues(M)
    R = Harris_response(lambdaa, k)
    H = suppress_non_maxima_harris(R, W_size)
    corner = cv2.findNonZero(H)
    return corner


def draw_corners(img, corners):
    for i in range(len(corners)):
        cv2.circle(img, tuple(corners[i][0]), 1, (0, 0, 255), -1)
    return img


def draw_cornersG(img, corners):
    for i in range(len(corners)):
        cv2.circle(img, tuple(corners[i][0]), 4, (0, 0, 255), -1)
    return img

# Función de callback para la trackbar
def change_window_size(value):
    global window_size
    window_size = value
    
def main():
  # Creación de ventana y trackbar para modificar el tamaño de la ventana de Harris
    cv2.namedWindow("Configuracion")
    cv2.createTrackbar("Tamaño de ventana", "Configuracion", window_size, 15, change_window_size)

    cap = cv2.VideoCapture(0)
    kernel = get_gaussian_kernel(5, 1)
    Umin = 5
    Umax = 255
    scale_percent = 30
    aux1 = 0
    Prueba = cv2.imread("TestCorners.png")
    PruebaCopy = Prueba.copy()
    Pruebaceros = numpy.zeros(Prueba.shape)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if aux1 == 0:
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            dsize = (width, height)
            aux1 = 7
        output = cv2.resize(frame, dsize)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

        if aux1 == 7:
            img = cv2.cvtColor(PruebaCopy, cv2.COLOR_BGR2GRAY)
            img = numpy.float32(img)
            img = img / 255.0
            corner = corners_harris(img, window_size, 50 / 1000.0)
            H = draw_cornersG(Prueba, corner)
            cv2.imshow("Prueba Harris", H)
        elif aux1 >= 8:
            cv2.imshow("Video", output)
            filtered = convolve2d(output, kernel)
            G = Canny_config(filtered, 10, 255)
            cv2.imshow("Canny", G)

            Gp = G.copy()
            img_1 = cv2.cvtColor(G, cv2.COLOR_GRAY2BGR)
            img = numpy.float32(Gp)
            corner = corners_harris(img, window_size, 246 / 1000.0)
            H = draw_corners(img_1, corner)
            cv2.imshow("Sol Harris", H)

        else:
            cv2.imshow("Video", output)

        if aux1 <= 8:
            aux1 += 1

        if cv2.waitKey(10) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    window_size = 3
    main()
