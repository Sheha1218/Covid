from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
import cv2
import io
import re
from PIL import Image
import tensorflow as tf
import pygame
import logging

# Initialize Flask app
app = Flask(__name__)


logging.basicConfig(level=logging.INFO)


try:
    model = tf.keras.models.load_model('mask_model.h5')
    logging.info("Model loaded successfully!")
except Exception as e:
    logging.error(f"Error loading model: {e}")


class_names = ['with_mask', 'without_mask']


try:
    pygame.mixer.init()
    pygame.mixer.music.load('sound.mp3')
    logging.info("Sound system initialized!")
except Exception as e:
    logging.error(f"Error initializing sound: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_mask', methods=['POST'])
def detect_mask():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'Invalid request'}), 400

      
        image_data = data['image']
        image_data = re.sub('^data:image/.+;base64,', '', image_data)
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        
        
        image = image.resize((224, 224))  # Adjust size based on model input
        image = np.array(image) / 255.0   # Normalize
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        
        prediction = model.predict(image)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        
        if predicted_class == 'without_mask' and not pygame.mixer.music.get_busy():
            pygame.mixer.music.play()

        return jsonify({
            'result': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        logging.error(f"Detection error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)