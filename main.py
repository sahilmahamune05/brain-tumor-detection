from flask import Flask, render_template, request, send_from_directory
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16 # ✅ FIXED: VGG16 preprocessing
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
 
# Initialize Flask app
app = Flask(__name__)
 
# ✅ Load the trained model
model = load_model(r'D:\BrainTumorusingVGG16\models\model.h5')
 
# ✅ FIXED: Class labels match EXACT training order from your notebook
# Notebook training used: CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
# So: index 0=glioma, 1=meningioma, 2=notumor, 3=pituitary
# Your old main.py had ['glioma', 'meningioma', 'pituitary', 'notumor'] — pituitary & notumor SWAPPED!
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
 
# VGG16 requires 224x224
IMAGE_SIZE = 224
 
# Define uploads folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
 
# -------------------------------
# Prediction Function
# -------------------------------
def predict_tumor(image_path):
    try:
        # Load image with correct size
        img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
 
        # Convert to array
        img_array = img_to_array(img)
 
        # ✅ FIXED: Use VGG16's preprocess_input — NOT img / 255.0
        # Training used preprocess_input; inference MUST use the same normalization
        img_array = preprocess_input(img_array)
 
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
 
        # Predict
        predictions = model.predict(img_array)
 
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence_score = float(np.max(predictions))
 
        label = class_labels[predicted_class_index]
 
        if label == 'notumor':
            return "No Tumor", confidence_score
        else:
            return f"Tumor: {label.capitalize()}", confidence_score
 
    except Exception as e:
        return f"Error: {str(e)}", 0


# -------------------------------
# Main Route
# -------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
 
        if 'file' not in request.files:
            return render_template('index.html', result="No file uploaded")
 
        file = request.files['file']
 
        if file.filename == '':
            return render_template('index.html', result="No file selected")
 
        # Save file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
 
        # Predict
        result, confidence = predict_tumor(file_path)
 
        return render_template(
            'index.html',
            result=result,
            confidence=f"{confidence * 100:.2f}%",
            file_path=f'/uploads/{file.filename}'
        )
 
    return render_template('index.html')
 
 
# -------------------------------
# Serve uploaded images
# -------------------------------
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
 
 
# -------------------------------
# Run App
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)