from flask import Flask, render_template, request, jsonify
import numpy as np
import base64
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('doodle_model.keras')  # Your trained model
classes = ['car', 'house', 'tree']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'image' not in data:
        return jsonify({'error': 'No image data received'}), 400

    try:
        img_data = data['image']
        img_data = img_data.split(',')[1]  # Remove base64 prefix
        img = np.frombuffer(base64.b64decode(img_data), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)

        pred = model.predict(img)
        pred_class = classes[np.argmax(pred)]

        return jsonify({'prediction': pred_class})
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({'error': 'Failed to process image'}), 500

if __name__ == '__main__':
    app.run(debug=True)



