import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo pre-entrenado
model = load_model('model.h5')

# Cargar el clasificador de caras pre-entrenado de OpenCV
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Iniciar la cámara
cap = cv2.VideoCapture(0)

# Definir un diccionario para las etiquetas de las emociones
emotions = {0: 'angry', 1: 'disgust', 2: 'fear',
            3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    # Capturar un frame de la cámara
    ret, frame = cap.read()
    if not ret:
        continue

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar caras en la imagen
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    # Iterar sobre las caras detectadas
    for (x, y, w, h) in faces:
        # Extraer la región de interés (ROI)
        roi_gray = gray[y:y+h, x:x+w]

        # Redimensionar la imagen a 48x48 y normalizar los valores de los píxeles
        roi_gray = cv2.resize(roi_gray, (48, 48)) / 255.0

        # Añadir una dimensión extra para la cantidad de canales de la imagen (en este caso, 1 canal de escala de grises)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        # Predecir la emoción en base a la ROI
        pred = model.predict(np.array([roi_gray]))
        emotion = emotions[np.argmax(pred)]

        # Dibujar un rectángulo alrededor de la cara y mostrar la emoción predicha
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar el frame
    cv2.imshow('Emotion Recognition', frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar la ventana
cap.release()
cv2.destroyAllWindows()
