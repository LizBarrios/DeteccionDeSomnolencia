# Entrenamiento de modelo para clasificación de ojos (abiertos/cerrados)

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Configuración
data_dir = 'dataset_new/train'  # Cambia esta ruta a la ubicación de tu dataset
categories = ['Closed', 'Open']  # Cambia estas categorías según tu dataset
IMG_SIZE = 64

# Carga y preprocesamiento de imágenes
data = []
labels = []

for i, category in enumerate(categories):
    folder = os.path.join(data_dir, category)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(i)

# Conversión a arrays de NumPy
X = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
y = to_categorical(labels, num_classes=2)

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aumento de datos
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# Definición del modelo CNN
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compilación del modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Entrenamiento del modelo
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=1000,
    callbacks=[early_stopping]
)

# Evaluación
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Pérdida en prueba: {test_loss:.4f}, Precisión en prueba: {test_acc:.4f}')

# Guardar el modelo
model.save('modelo_clasificacion_ojos.keras')
