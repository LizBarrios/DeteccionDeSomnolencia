import cv2
import numpy as np
import tensorflow as tf
from cvzone.FaceMeshModule import FaceMeshDetector
import pygame
import time

model = tf.keras.models.load_model('modelo_clasificacion_ojos.keras')
detector = FaceMeshDetector(maxFaces=1)

pygame.init()
pygame.mixer.init()
pygame.mixer.music.load("alerta-sonido.mp3")

parpadeo = False
inicio = 0
alarma = False
conteo_sue = 0

ojos_derecho_ids = [33, 160, 158, 133, 153, 144]
ojos_izquierdo_ids = [362, 385, 387, 263, 373, 380]

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
    return frame[y_min:y_max, x_min:x_max]

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

def generate():
    global parpadeo, inicio, alarma, conteo_sue
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        success, frame = cap.read()
        if not success:
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

                cerrado = pred_r == 0 and pred_l == 0

                if cerrado:
                    if not parpadeo:
                        inicio = time.time()
                        parpadeo = True
                    else:
                        duracion = time.time() - inicio
                        if duracion >= 3 and not alarma:
                            conteo_sue += 1
                            pygame.mixer.music.play(-1)
                            alarma = True
                        cv2.putText(frame, f"CIERRE: {int(duracion)}s", (30, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    if parpadeo:
                        parpadeo = False
                        alarma = False
                        pygame.mixer.music.stop()

                cv2.putText(frame, f"Microsuenos: {conteo_sue}", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
