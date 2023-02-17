import numpy as np
import cv2

cap  = cv2.VideoCapture(0)
while cap.isOpened(): # mientras la camara este abierta
    ret, frame = cap.read()     # devuelve si se pudo ono leer y los datos del cuadro 480x640
    if not ret:     # si no pudo abrir
        print("Can't receive frame (stream end?). Exiting ...")
        break
    cv2.imshow('My Video', frame)       # si si pudo obtener el frame actualiza la imagen 30x seg
    if cv2.waitKey(10) & 0xFF == 27:    # si se presiono la tecla sale, operacion bit a bit, pone a cero el bit mas significativo y se queda con el menos significativo (Esc)
        break
cap.release()    # 
cv2.destroyAllWindows()
