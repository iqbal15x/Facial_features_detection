import cv2
import numpy as np
from tensorflow.keras.models import load_model

def detect_features(image_path):
    model = load_model('model/facial_features_model.h5')
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    results = []
    for (x, y, w, h) in faces:
        roi_color = image[y:y+h, x:x+w]
        roi_color = cv2.resize(roi_color, (150, 150))
        roi_color = roi_color.astype('float32') / 255
        roi_color = np.expand_dims(roi_color, axis=0)

        prediction = model.predict(roi_color)
        label = np.argmax(prediction, axis=1)
        results.append(label)

        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imwrite(image_path, image)
    return results
