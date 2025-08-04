import os
import logging
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables to suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the pre-trained model
try:
    model = load_model('tomate_model.keras')  # Replace with your model's path
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'svg'}

# Check if file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Predict the disease from the image
def predict_disease(img_path):
    try:
        logger.info("Processing image for prediction...")
        
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))  # Updated size to match model input
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize

        logger.info(f"Preprocessed image shape: {img_array.shape}")
        
        # Predict using the model
        prediction = model.predict(img_array)
        logger.info(f"Raw prediction output: {prediction}")
        
        # Get the class index with the highest probability
        max_prob = np.max(prediction)
        predicted_class = np.argmax(prediction, axis=1)
        confidence_score = round(max_prob * 100, 2)
        logger.info(f"Predicted class index: {predicted_class}")
        logger.info(f"Predicted confidence: {confidence_score}%")

        #Confidence threshold
        if max_prob < 0.5:
            predicted_disease = "Unknown Image"
        else:
        # Map the class index to the class label

            class_labels = {
                0: "Tomato Bacterial Spot",
                1: "Tomato Early Blight",
                2: "Tomato Late Blight",
                3: "Tomato Leaf Mold",
                4: "Tomato Septoria Leaf Spot",
                5: "Tomato Spider Mites (Two Spotted Spider Mite)",
                6: "Tomato Target Spot",
                7: "Tomato Tomato Yellow Leaf Curl Virus",
                8: "Tomato Mosaic Virus",
                9: "Tomato Healthy"
            }


            predicted_disease = class_labels.get(predicted_class[0], "Unknown Disease")

        logger.info(f"Predicted disease: {predicted_disease} | Confidence: {confidence_score}%")
        
        return f"{predicted_disease} | {confidence_score}%"
    except Exception as e:
        logger.error(f"Error during disease prediction: {e}")
        return "Prediction Error"

@app.route('/')
def index():
    return render_template('index.html')  # Make sure index.html is in the templates folder

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)  # Save image in 'uploads' directory

        try:
            file.save(filepath)
            logger.info(f"File saved at: {filepath}")

            # Make prediction on the uploaded image
            disease = predict_disease(filepath)
            logger.info(f"Prediction: {disease}")

            # Optionally, remove the file after processing
            os.remove(filepath)

            return jsonify({'prediction': disease})
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return jsonify({'error': 'An error occurred during prediction'}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    # Ensure 'uploads' directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(debug=True)