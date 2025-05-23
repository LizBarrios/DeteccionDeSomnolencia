# 💤 Sistema de Detección de Somnolencia

Este proyecto detecta signos de somnolencia en tiempo real mediante visión computacional y redes neuronales convolucionales (CNN). Si ambos ojos permanecen cerrados por más de **3 segundos**, se activa una **alarma sonora** como medida de alerta.

---

## 🧠 Descripción del Proyecto

El sistema combina varias tecnologías para lograr la detección de somnolencia:

* 🎥 **OpenCV**: Captura de video en tiempo real desde la webcam.
* 🧱 **MediaPipe (cvzone)**: Detección de la malla facial para localizar los ojos.
* 🤖 **TensorFlow (Keras)**: Clasificación de ojos como abiertos o cerrados usando un modelo CNN.
* 🔊 **Pygame**: Reproducción de una alerta sonora.
* 🌐 **Flask**: Interfaz web que muestra el video procesado en tiempo real.

---

## 📁 Estructura del Proyecto

```
deteccion-somnolencia/
├── app.py                      # Servidor Flask
├── utils.py                   # Lógica para captura y procesamiento de video
├── modelo_clasificacion_ojos.keras  # Modelo CNN entrenado
├── alerta-sonido.mp3          # Sonido de alarma
├── templates/
│   └── index.html             # Interfaz web
├── requirements.txt           # Dependencias del proyecto
├── entrenar_modelo.py         # Script para entrenar el modelo (opcional)
└── README.md                  # Este archivo
```

---
## ⚠️ Advertencia Importante
Este sistema requiere el uso de OpenCV y acceso a una cámara web.
Asegúrate de lo siguiente antes de ejecutar la aplicación:

- Tienes una cámara conectada y funcional (integrada o externa).
- Los permisos de cámara están habilitados (especialmente en navegadores o sistemas operativos con restricciones).
- El entorno tiene instalado opencv-python correctamente.
- No hay otras aplicaciones usando la cámara al mismo tiempo.
---

## ⚙️ Instalación

### 1. Clona el repositorio

```bash
git clone https://github.com/tu_usuario/deteccion-somnolencia.git
cd deteccion-somnolencia
```

### 2. Crear un entorno virtual


```bash
python -m venv venv
```

### 3. Instala las dependencias

```bash
pip install -r requirements.txt
```

---

## 🧪 Entrenamiento del Modelo (opcional)

El archivo `modelo_clasificacion_ojos.keras` fue entrenado con un conjunto de imágenes etiquetadas como **Open** y **Closed**, utilizando una CNN con 3 capas convolucionales.

Si deseas entrenar tu propio modelo:

```bash
python entrenar_modelo.py
```

---

## 🚀 Ejecución

Asegúrate de tener una **webcam conectada**.

Ejecuta el servidor Flask:

```bash
python app.py
```

Luego abre tu navegador y visita:
👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🛠 Requisitos del Sistema

* Python **3.8 o superior**
* Webcam funcional
* Sistema operativo con acceso a hardware de audio (para la alarma)

---

## 👩‍💻 Autores

* **Lizeth Barrios**
* **Daniel Rios**

