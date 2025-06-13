import os
import time
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# --- MODEL LOADING ---
MODEL_FILENAME = 'pakde_folder_model.keras'  # Or .tflite if you want to test TFLite
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

model = None
model_type = None  # "keras" or "tflite"

try:
    print(f"Trying to load model from {MODEL_PATH}", flush=True)
    # For Keras model
    model = tf.keras.models.load_model(MODEL_PATH)
    model_type = "keras"
    print("Keras model loaded successfully.", flush=True)

    # # For TFLite model (uncomment to test tflite)
    # interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    # interpreter.allocate_tensors()
    # model = interpreter
    # model_type = "tflite"
    # print("TFLite model loaded successfully.", flush=True)

except Exception as e:
    print(f"Model failed to load: {e}", flush=True)
    model = None
    model_type = None


def preprocess_image(img, target_size=(224, 224)):
    img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img).astype(np.float32)
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/journal')
def journal():
    return render_template('journal.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    print("Predict endpoint hit", flush=True)

    # Check if model is loaded
    if model is None:
        print("Model not loaded.", flush=True)
        return jsonify({'error': 'Model not loaded', 'status': 'fail'}), 500

    if 'file' not in request.files:
        print("No file part in request", flush=True)
        return jsonify({'error': 'Tidak ada file yang diunggah', 'status': 'fail'}), 400

    file = request.files['file']
    if file.filename == '':
        print("Empty filename", flush=True)
        return jsonify({'error': 'Tidak ada file yang dipilih', 'status': 'fail'}), 400

    try:
        print("Reading image...", flush=True)
        img = Image.open(io.BytesIO(file.read()))
        print("Image opened.", flush=True)

        print("Preprocessing image...", flush=True)
        processed_img = preprocess_image(img)
        print("Image preprocessed.", flush=True)

        print("Running model.predict...", flush=True)
        t_pred_start = time.time()

        if model_type == "keras":
            prediction = model.predict(processed_img)
            prediction_value = float(prediction[0][0])
        elif model_type == "tflite":
            input_details = model.get_input_details()
            output_details = model.get_output_details()
            model.set_tensor(input_details[0]['index'], processed_img.astype(np.float32))
            model.invoke()
            prediction_value = float(model.get_tensor(output_details[0]['index'])[0][0])
        else:
            raise Exception("Unknown model type.")

        t_pred_end = time.time()
        print(f"Prediction itself took {t_pred_end - t_pred_start:.3f} seconds", flush=True)

        result = "Parkinson Terdeteksi" if prediction_value > 0.5 else "Sehat"
        confidence = prediction_value if prediction_value > 0.5 else 1 - prediction_value
        confidence_percentage = round(confidence * 100, 2)

        elapsed = time.time() - start_time
        print(f"Prediction completed in {elapsed:.3f} seconds", flush=True)
        return jsonify({
            'result': result,
            'confidence': confidence_percentage,
            'status': 'success',
            'model_type': model_type
        })

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Prediction failed after {elapsed:.3f} seconds: {str(e)}", flush=True)
        return jsonify({'error': str(e), 'status': 'fail'}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
