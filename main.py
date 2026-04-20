from flask import Flask, render_template, request, send_from_directory
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
import json
from datetime import datetime
from PIL import Image as PILImage
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import io
import base64

# ─────────────────────────────────────────────
# Initialize Flask app
# ─────────────────────────────────────────────
app = Flask(__name__)

# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
model = load_model(r'D:\brain-tumor-detection\models\model.h5')

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMAGE_SIZE = 224
UPLOAD_FOLDER = 'uploads'
LOG_FILE = 'prediction_log.json'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# ─────────────────────────────────────────────
# Helper: File validation
# ─────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_valid_image(path):
    try:
        img = PILImage.open(path)
        img.verify()
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────
# Helper: Prediction logging
# ─────────────────────────────────────────────
def log_prediction(filename, result, confidence):
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "result": result,
        "confidence": f"{confidence * 100:.2f}%"
    }
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            logs = json.load(f)
    logs.append(entry)
    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=2)


# ─────────────────────────────────────────────
# Helper: Grad-CAM heatmap generation
# ─────────────────────────────────────────────
def generate_gradcam_b64(img_path, model, image_size=224):
    try:
        img = load_img(img_path, target_size=(image_size, image_size))
        img_array = preprocess_input(np.expand_dims(img_to_array(img), axis=0))

        vgg16 = model.get_layer("vgg16")

        full_grad_model = tf.keras.models.Model(
            inputs=vgg16.input,
            outputs=[vgg16.get_layer("block5_conv3").output, vgg16.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, vgg_output = full_grad_model(img_array)
            x = vgg_output
            for layer in model.layers:
                if layer.name not in ["vgg16"]:
                    x = layer(x)
            pred_index = tf.argmax(x[0])
            class_channel = x[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.squeeze(conv_outputs[0] @ pooled[..., tf.newaxis])
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        heatmap = np.uint8(255 * heatmap.numpy())

        colored = np.uint8(cm.get_cmap("jet")(heatmap)[:, :, :3] * 255)
        colored_r = np.array(PILImage.fromarray(colored).resize((image_size, image_size)))
        overlay = np.uint8(colored_r * 0.4 + np.array(img))

        buf = io.BytesIO()
        PILImage.fromarray(overlay).save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None


# ─────────────────────────────────────────────
# Helper: Tumor prediction
# ─────────────────────────────────────────────
def predict_tumor(image_path):
    try:
        img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img_array = preprocess_input(np.expand_dims(img_to_array(img), axis=0))

        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))
        label = CLASS_LABELS[predicted_index]

        # Build per-class probabilities for the chart
        class_probs = {CLASS_LABELS[i]: float(predictions[0][i]) * 100 for i in range(len(CLASS_LABELS))}

        if label == 'notumor':
            return "No Tumor Detected", confidence, class_probs
        else:
            return f"Tumor Detected: {label.capitalize()}", confidence, class_probs

    except Exception as e:
        return f"Error: {str(e)}", 0, {}


# ─────────────────────────────────────────────
# Route: Main detection page
# ─────────────────────────────────────────────
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded.")

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error="No file selected.")

        if not allowed_file(file.filename):
            return render_template('index.html', error="Invalid file type. Please upload PNG, JPG, or JPEG.")

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        if not is_valid_image(file_path):
            os.remove(file_path)
            return render_template('index.html', error="Corrupted or unreadable image. Please try another file.")

        result, confidence, class_probs = predict_tumor(file_path)
        log_prediction(file.filename, result, confidence)
        gradcam_img = generate_gradcam_b64(file_path, model)

        return render_template(
            'index.html',
            result=result,
            confidence=f"{confidence * 100:.2f}%",
            confidence_val=round(confidence * 100, 2),
            file_path=f'/uploads/{file.filename}',
            gradcam_img=gradcam_img,
            class_probs=class_probs
        )

    return render_template('index.html')


# ─────────────────────────────────────────────
# Route: Prediction history
# ─────────────────────────────────────────────
@app.route('/history')
def history():
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            logs = json.load(f)
    return render_template('history.html', logs=logs[::-1])


# ─────────────────────────────────────────────
# Route: Serve uploaded images
# ─────────────────────────────────────────────
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)
