from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import os
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'chext.h5'  # Update this path if needed

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model with error handling
try:
    model = tf.keras.models.load_model('D:\Way to denmark\Projects\Covid-19\models\chext.h5')
    class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read image")
        
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        raise ValueError(f"Image processing error: {e}")

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(image_path)

        # Preprocess and predict
        processed_image = preprocess_image(image_path)
        prediction = model.predict(processed_image)
        class_index = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction, axis=1)[0])

        # Clean up
        os.remove(image_path)

        return jsonify({
            'class': class_names[class_index],
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)