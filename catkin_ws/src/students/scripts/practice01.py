import cv2
import numpy as np

# Variables globales
rectangulo = False  # Si se está haciendo clic con el botón izquierdo del ratón
x1, y1 = -1, -1
cuadro = None  # El cuadro actual de la cámara web
roi = None  # La región de interés (el rectángulo seleccionado)

# Función de devolución de llamada del ratón para el evento de hacer clic y arrastrar el ratón
def dibujar_rectangulo(event, x, y, flags, param):
    global x1, y1, rectangulo, roi, cuadro
    if event == cv2.EVENT_LBUTTONDOWN:
        rectangulo = True
        x1, y1 = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if rectangulo:
            copia_cuadro = cuadro.copy()
            cv2.rectangle(copia_cuadro, (min(x1, x), min(y1, y)), (max(x1, x), max(y1, y)), (255, 0, 0), 1)
            cv2.imshow('Practica1', copia_cuadro)
    elif event == cv2.EVENT_LBUTTONUP:
        rectangulo = False
        roi = cuadro[min(y1, y):max(y1, y), min(x1, x):max(x1, x)]

# Función a ser llamada cuando cambia el control deslizante
def al_cambiar(x):
    pass

# Abrir la cámara web
cam = cv2.VideoCapture(0)

# Obtener la resolución del video
ancho_cuadro = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
alto_cuadro = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Cargar y redimensionar la imagen de fondo
fondo = cv2.imread('Fondo.png')
fondo = cv2.resize(fondo, (ancho_cuadro, alto_cuadro))

# Crear una ventana y adjuntar a ella la función de devolución de llamada del ratón
cv2.namedWindow('Practica1')
cv2.setMouseCallback('Practica1', dibujar_rectangulo)

# Crear un control deslizante
cv2.createTrackbar('Tolerancia', 'Practica1', 0, 255, al_cambiar)

while True:
    ret, cuadro = cam.read()
    if ret == False:
        break
    if roi is not None:
        tolerancia = cv2.getTrackbarPos('Tolerancia', 'Practica1')
        color_promedio = np.average(np.average(roi, axis=0), axis=0)
        limite_inferior = np.array([color_promedio[0]-tolerancia, color_promedio[1]-tolerancia, color_promedio[2]-tolerancia])
        limite_superior = np.array([color_promedio[0]+tolerancia, color_promedio[1]+tolerancia, color_promedio[2]+tolerancia])
        mascara = cv2.inRange(cuadro, limite_inferior, limite_superior)
        cuadro = cv2.bitwise_and(cuadro, cuadro, mask=cv2.bitwise_not(mascara))
        cuadro = cv2.add(cuadro, cv2.bitwise_and(fondo, fondo, mask=mascara))
    cv2.imshow('Practica1', cuadro)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

