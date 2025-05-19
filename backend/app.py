from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import cv2
import tensorflow as tf
from werkzeug.utils import secure_filename
from datetime import datetime

# Initialisation de l'application
app = Flask(__name__)
CORS(app)  # Permet les requêtes depuis React

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Charger le modèle
model = tf.keras.models.load_model('faciales.h5')
emotion_labels = ['Colère', 'Dégoût', 'Peur', 'Heureux', 'Triste', 'Surpris', 'Neutre']

# 🔍 Prétraitement image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 48, 48, 1))
    return reshaped

# 📤 Route de prédiction
@app.route('/predict-emotion', methods=['POST'])
def predict_emotion():
    if 'image' not in request.files:
        return jsonify({'error': 'Aucune image reçue'}), 400

    image = request.files['image']
    filename = secure_filename(f"{datetime.now().timestamp()}_webcam.jpg")
    path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(path)

    try:
        processed = preprocess_image(path)
        prediction = model.predict(processed)[0]
        emotion = emotion_labels[np.argmax(prediction)]
        score = float(np.max(prediction))
        os.remove(path)  # Nettoyage
        return jsonify({'emotion': emotion, 'score': round(score, 4)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Lancer le serveur
if __name__ == '__main__':
    app.run(debug=True, port=5000)
