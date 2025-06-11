import os
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load model
model_path = os.path.join(os.path.dirname(__file__), 'pakde_folder_model.keras')
model = load_model(model_path)

# Fungsi untuk preprocessing gambar
def preprocess_image(img, target_size=(224, 224)):
    # Konversi ke RGB
    img = img.convert('RGB')
    # Resize gambar
    img = img.resize(target_size)
    # Konversi ke array
    img_array = np.array(img).astype(np.float32)
    # Preprocessing MobileNetV2: dari [0,255] ke [-1,1]
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Route untuk halaman utama
@app.route('/')
def home():
    return render_template('home.html')

# Route untuk halaman tentang penyakit Parkinson
@app.route('/about')
def about():
    return render_template('about.html')

# Route untuk halaman jurnal referensi
@app.route('/journal')
def journal():
    return render_template('journal.html')

# Route untuk halaman tim capstone
@app.route('/team')
def team():
    return render_template('team.html')

# Route untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diunggah'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'})
    
    try:
        # Baca gambar
        img = Image.open(io.BytesIO(file.read()))
        
        # Preprocess gambar
        processed_img = preprocess_image(img)
        
        # Prediksi
        prediction = model.predict(processed_img)
        
        # Dapatkan hasil prediksi
        prediction_value = float(prediction[0][0])
        
        # Tentukan hasil klasifikasi
        if prediction_value > 0.5:
            result = "Parkinson Terdeteksi"
        else:
            result = "Sehat"
        
        # Konversi confidence score ke persentase
        confidence = prediction_value if prediction_value > 0.5 else 1 - prediction_value
        confidence_percentage = round(confidence * 100, 2)
        
        return jsonify({
            'result': result,
            'confidence': confidence_percentage
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)