from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import cv2
import tensorflow as tf
import base64 # Importation nécessaire pour décoder l'image base64
from werkzeug.utils import secure_filename
from datetime import datetime

# Initialisation de l'application
app = Flask(__name__)
CORS(app)  # Permet les requêtes depuis React

# Configuration du dossier d'upload (peut être utile pour le débogage, mais pas strictement nécessaire si on traite l'image en mémoire)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Charger le modèle
# Assurez-vous que 'faciales.h5' est dans le même répertoire que votre script Flask
try:
    model = tf.keras.models.load_model('faciales.h5')
    print("Modèle TensorFlow chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    # Gérer l'erreur, par exemple en arrêtant l'application ou en désactivant la fonctionnalité de prédiction
    model = None # S'assurer que le modèle est None si le chargement échoue

# ✅ Liste des émotions reconnues (doit correspondre à l'ordre dans le modèle et au frontend)
# Assurez-vous que cet ordre correspond à l'ordre de sortie de votre modèle et à 'emotionsList' dans React
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


# 🔍 Prétraitement image
def preprocess_image_from_base64(base64_string):
    # Supprimer le préfixe "data:image/jpeg;base64," si présent
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]

    # Décoder la chaîne base64 en bytes
    img_bytes = base64.b64decode(base64_string)

    # Convertir les bytes en tableau numpy
    np_arr = np.frombuffer(img_bytes, np.uint8)

    # Lire l'image avec OpenCV
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Impossible de décoder l'image. Le format est-il correct ?")

    # Convertir en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Redimensionner à 48x48 pixels
    resized = cv2.resize(gray, (48, 48))

    # Normaliser les pixels entre 0 et 1
    normalized = resized / 255.0

    # Ajouter les dimensions nécessaires pour le modèle (batch_size, height, width, channels)
    reshaped = np.reshape(normalized, (1, 48, 48, 1))
    return reshaped

# 📤 Route de détection d'émotion
# Le frontend envoie à '/detect_emotion'
@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    if model is None:
        return jsonify({'error': 'Le modèle de détection d\'émotion n\'a pas pu être chargé.'}), 500

    # Vérifier si la requête contient des données JSON
    if not request.is_json:
        return jsonify({'error': 'La requête doit être au format JSON.'}), 400

    data = request.get_json()

    # Vérifier si la clé 'image' est présente dans les données JSON
    if 'image' not in data:
        return jsonify({'error': 'Aucune image (base64) reçue dans le corps de la requête JSON.'}), 400

    image_base64 = data['image']

    try:
        # Prétraiter l'image
        processed_image = preprocess_image_from_base64(image_base64)

        # Effectuer la prédiction
        predictions = model.predict(processed_image)[0] # Obtenir les scores de probabilité pour chaque émotion

        # Trouver l'émotion avec le score le plus élevé
        detected_emotion_index = np.argmax(predictions)
        detected_emotion_label = emotion_labels[detected_emotion_index]

        # Convertir les scores numpy en liste Python pour jsonify
        scores_list = predictions.tolist()

        # Retourner l'émotion détectée et tous les scores
        return jsonify({
            'emotion': detected_emotion_label,
            'scores': scores_list # Retourne la liste complète des scores
        })

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        print(f"Erreur inattendue lors du traitement de l'image : {e}")
        return jsonify({'error': f'Erreur interne du serveur lors du traitement de l\'image: {e}'}), 500

# Lancer le serveur
if __name__ == '__main__':
    # Utilisez host='0.0.0.0' pour rendre le serveur accessible depuis d'autres machines sur le réseau local
    # (utile si votre frontend React n'est pas sur le même localhost que Flask)
    app.run(debug=True, host='127.0.0.1', port=5000)

