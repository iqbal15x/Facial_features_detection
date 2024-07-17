from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'model/facial_features_model.h5'

# Define the classes and recommendations
CLASS_NAMES = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
RECOMMENDATIONS = {
    'Heart': [
        "Cat-eye glasses: These frames balance the wider forehead and add width to the chin area.",
        "Oval or round frames: Softens the angles of the face.",
        "Rimless or semi-rimless frames: Lighter frames can minimize the width at the top of the face."
    ],
    'Oblong': [
        "Tall frames: Help create the illusion of a shorter face.",
        "Decorative or contrasting temples: Add width to the face.",
        "Wayfarers or rectangular frames: Create a balanced look by adding width."
    ],
    'Oval': [
        "Almost any frame shape: This versatile face shape can pull off most styles.",
        "Square or rectangular frames: Add structure and contrast.",
        "Aviators: Complement the natural balance of the face."
    ],
    'Round': [
        "Angular and geometric frames: Add definition and balance the roundness.",
        "Rectangular or square frames: Provide a slimming effect.",
        "Clear bridge frames: Widen the eyes and add contrast."
    ],
    'Square': [
        "Round or oval frames: Soften the angular features.",
        "Cat-eye glasses: Draw attention upwards and balance the strong jawline.",
        "Rimless or semi-rimless frames: Provide a lighter look that softens the angles."
    ]
}

try:
    model = load_model(MODEL_PATH, compile=False)
except Exception as e:
    print(f"Error loading the model: {e}")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        filepath = os.path.join('static/uploads', file.filename)
        file.save(filepath)
        
        # Preprocess the image
        img = image.load_img(filepath, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
        recommendations = RECOMMENDATIONS[predicted_class]
        
        return render_template('result.html', prediction=predicted_class, image_path=filepath, recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
