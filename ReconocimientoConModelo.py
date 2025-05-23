import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import pygame
import time
import tensorflow as tf
import numpy as np
import math

# Cargar modelo entrenado (CNN bioinspirado)
model = tf.keras.models.load_model('modelo_clasificacion_ojos.keras')

# Inicializar cámara
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Inicializar pygame para alerta
pygame.init()
pygame.mixer.init()

def playaudio():
    pygame.mixer.music.load("alerta-sonido.mp3")
    pygame.mixer.music.play(-1)

def stopaudio():
    pygame.mixer.music.stop()

# Inicializar detector facial
detector = FaceMeshDetector(maxFaces=1)

# Puntos para ojos
ojos_derecho_ids = [33, 160, 158, 133, 153, 144]
ojos_izquierdo_ids = [362, 385, 387, 263, 373, 380]

# Función para extraer ojo de una imagen
def extraer_ojo(frame, puntos):
    x_coords = [p[0] for p in puntos]
    y_coords = [p[1] for p in puntos]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    padding = 5
    x_min = max(x_min - padding, 0)
    y_min = max(y_min - padding, 0)
    x_max = min(x_max + padding, frame.shape[1])
    y_max = min(y_max + padding, frame.shape[0])
    ojo = frame[y_min:y_max, x_min:x_max]
    return ojo

# Función para preprocesar ojo
def preprocess_eye(ojo):
    try:
        ojo = cv2.resize(ojo, (64, 64))
        ojo = cv2.cvtColor(ojo, cv2.COLOR_BGR2GRAY)
        ojo = ojo / 255.0
        ojo = np.expand_dims(ojo, axis=-1)
        ojo = np.expand_dims(ojo, axis=0)
        return ojo
    except:
        return None

# Variables de control
inicio = 0
parpadeo = False
alarma_activada = False
conteo_sue = 0

# Loop principal
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame, faces = detector.findFaceMesh(frame, draw=False)

    if faces:
        face = faces[0]
        ojo_der = [face[i] for i in ojos_derecho_ids]
        ojo_izq = [face[i] for i in ojos_izquierdo_ids]

        eye_right_img = preprocess_eye(extraer_ojo(frame, ojo_der))
        eye_left_img = preprocess_eye(extraer_ojo(frame, ojo_izq))

        if eye_right_img is not None and eye_left_img is not None:
            pred_right = model.predict(eye_right_img, verbose=0)[0]
            pred_left = model.predict(eye_left_img, verbose=0)[0]
            pred_r = np.argmax(pred_right)
            pred_l = np.argmax(pred_left)

            cerrado = pred_r == 0 and pred_l == 0  # Ambos ojos cerrados

            if cerrado:
                if not parpadeo:
                    inicio = time.time()
                    parpadeo = True
                else:
                    duracion = time.time() - inicio
                    cv2.putText(frame, f"Cierre: {int(duracion)}s", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    if duracion >= 3 and not alarma_activada:
                        conteo_sue += 1
                        playaudio()
                        alarma_activada = True
                        cv2.putText(frame, "¡MICRO SUENO DETECTADO!", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                if parpadeo:
                    fin = time.time()
                    duracion = fin - inicio
                    if duracion < 3:
                        pass  # Parpadeo normal
                parpadeo = False
                alarma_activada = False
                stopaudio()

            # Mostrar etiquetas
            estado_ojo_d = "CERRADO" if pred_r == 0 else "ABIERTO"
            estado_ojo_i = "CERRADO" if pred_l == 0 else "ABIERTO"

            cv2.putText(frame, f"Ojo D: {estado_ojo_d}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Ojo I: {estado_ojo_i}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Microsuenos: {conteo_sue}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Detección de Somnolencia - CNN", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
stopaudio()
