"""
Flask Web Application for Brain MRI Classification
Provides a web interface for doctors to upload and analyze MRI images
"""

import os
import sys
import uuid
from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import cv2

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Class names
CLASS_NAMES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
CLASS_COLORS = {
    'Glioma Tumor': '#dc3545',  # Red
    'Meningioma Tumor': '#fd7e14',  # Orange
    'No Tumor': '#28a745',  # Green
    'Pituitary Tumor': '#6f42c1'  # Purple
}

# Load model
model = None

def load_model():
    """Load the trained model"""
    global model
    model_path = 'models/final_brain_mri_model.h5'
    
    # Also check for best model if final doesn't exist
    if not os.path.exists(model_path):
        model_path = 'models/best_brain_mri_model.h5'
    
    if os.path.exists(model_path):
        print(f"✅ Loading model from {model_path}")
        model = tf.keras.models.load_model(model_path)
        return True
    else:
        print(f"❌ Model not found at {model_path}")
        print("Please run 'python src/train.py' first to train the model")
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    # Read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size (224x224)
    image = cv2.resize(image, (224, 224))
    
    # Normalize
    image = image / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

def predict_image(image_path):
    """Make prediction on uploaded image"""
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image, verbose=0)[0]
    
    predicted_class_idx = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_class_idx]
    confidence = float(predictions[predicted_class_idx])
    
    # Get all class probabilities
    probabilities = {
        CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))
    }
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': probabilities,
        'color': CLASS_COLORS[predicted_class]
    }

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # Check if file is empty
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file type
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Use JPG, PNG, or JPEG'}), 400
    
    # Save file
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)
    
    try:
        # Make prediction
        result = predict_image(filepath)
        result['image_path'] = url_for('uploaded_file', filename=unique_filename)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up - remove file after prediction
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    # Load model before starting server
    if load_model():
        print("\n" + "=" * 60)
        print("🚀 Starting Brain MRI Classification Web App")
        print("=" * 60)
        print("\n📍 Access the application at: http://127.0.0.1:5000")
        print("📍 Press CTRL+C to stop the server")
        print("\n💡 Tip: Upload a brain MRI image to get a diagnosis")
        print("=" * 60 + "\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Failed to load model. Please train the model first:")
        print("   python src/train.py")