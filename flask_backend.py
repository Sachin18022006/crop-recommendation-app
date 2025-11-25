import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import joblib
import numpy as np

# Initialize the Flask application
app = Flask(__name__)
# Enable CORS for the Streamlit app running on a different port
CORS(app)

# ---------------------------
# GLOBAL DATA/MODEL SETUP
# ---------------------------

# Define paths to model files
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
CROP_MODEL_PATH = os.path.join(MODEL_DIR, "crop_model.pkl")
CROP_ENCODER_PATH = os.path.join(MODEL_DIR, "crop_encoder.pkl")
MODEL_FEATURES_PATH = os.path.join(MODEL_DIR, "model_features.pkl")
FERTILIZER_RATIOS_PATH = os.path.join(MODEL_DIR, "fertilizer_ratios.pkl")

# Initialize variables
CROP_MODEL = None
CROP_ENCODER = None
MODEL_FEATURES = []
FERTILIZER_RATIOS = {}

def load_models():
    """Load all necessary ML models and data using joblib."""
    global CROP_MODEL, CROP_ENCODER, MODEL_FEATURES, FERTILIZER_RATIOS
    try:
        # Load Crop Recommendation Model (RandomForestClassifier)
        CROP_MODEL = joblib.load(CROP_MODEL_PATH)
        print("Crop Model loaded successfully.")
        
        # Load Crop Label Encoder (for decoding predictions)
        CROP_ENCODER = joblib.load(CROP_ENCODER_PATH)
        print("Crop Encoder loaded successfully.")

        # Load Model Features (to ensure consistent column order for prediction)
        MODEL_FEATURES = joblib.load(MODEL_FEATURES_PATH)
        print(f"Model Features loaded: {len(MODEL_FEATURES)} features.")

        # Load Fertilizer Ratios (for the lookup table)
        FERTILIZER_RATIOS = joblib.load(FERTILIZER_RATIOS_PATH)
        print("Fertilizer Ratios loaded successfully.")

    except FileNotFoundError as e:
        print(f"ERROR: Could not find model file: {e}")
    except Exception as e:
        print(f"ERROR: An error occurred during model loading: {e}")
        
# Load models when the application starts
load_models()

# ---------------------------
# API Endpoints
# ---------------------------

@app.route('/', methods=['GET'])
def index():
    # This simple response confirms the server is live and working
    return "Agri-Tech ML API is running successfully!", 200

@app.route('/predict', methods=['POST'])
def predict_crop():
    """
    Endpoint for Crop Recommendation prediction.
    Inputs: N, P, K, temperature, humidity, ph, rainfall, soil_type
    Output: recommended_crop
    """
    # Check if models are loaded before proceeding
    if not CROP_MODEL or not CROP_ENCODER or not MODEL_FEATURES:
        return jsonify({"recommended_crop": None, "error": "ML models are not loaded. Server configuration error."}), 500

    try:
        data = request.get_json()
        
        # 1. Input Validation and Extraction
        required_numeric_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        soil_key = 'soil_type'
        
        input_data = {}
        for feature in required_numeric_features:
            value = data.get(feature)
            if value is None:
                return jsonify({"recommended_crop": None, "error": f"Missing required feature: {feature}"}), 400
            try:
                # Convert to float
                input_data[feature] = float(value)
            except ValueError:
                return jsonify({"recommended_crop": None, "error": f"Invalid numeric value for feature: {feature}"}), 400
        
        soil_type = data.get(soil_key)
        if not soil_type:
            return jsonify({"recommended_crop": None, "error": f"Missing required feature: {soil_key}"}), 400
        input_data[soil_key] = str(soil_type)

        
        # 2. Prepare the Final Feature Vector (Ensure correct order for prediction)
        
        # Initialize a dictionary to hold all features in the correct order
        final_features_dict = {}
        
        # Iterate through the model's expected features and populate the dictionary
        # This handles both numeric and OHE features correctly.
        
        # A. Populate numeric features
        for feature in required_numeric_features:
            final_features_dict[feature] = input_data[feature]
            
        # B. Populate OHE soil type features, setting all to 0 initially
        ohe_cols = [f for f in MODEL_FEATURES if f.startswith('soil_type_')]
        for col in ohe_cols:
            final_features_dict[col] = 0

        # C. Set the corresponding soil type OHE feature to 1
        ohe_col_name = f'soil_type_{input_data[soil_key]}'

        if ohe_col_name in final_features_dict:
            final_features_dict[ohe_col_name] = 1
        else:
            # Handle unrecognized soil types
            return jsonify({"recommended_crop": None, "error": f"Soil type '{input_data[soil_key]}' is not recognized by the model."}), 400

        # D. Create the final feature vector array in the strict order defined by MODEL_FEATURES
        final_features = np.array([final_features_dict[f] for f in MODEL_FEATURES]).reshape(1, -1)
        
        # 3. Make Prediction
        prediction_encoded = CROP_MODEL.predict(final_features)[0]
        
        # 4. Decode Prediction
        prediction = CROP_ENCODER.inverse_transform([prediction_encoded])[0]
            
        return jsonify({
            "recommended_crop": prediction,
            "error": None
        })

    except Exception as e:
        # Catch any unexpected errors during processing
        print(f"Prediction Error: {e}")
        return jsonify({"recommended_crop": None, "error": f"An internal error occurred during prediction: {str(e)}"}), 500

@app.route('/fertilizer_recommendation', methods=['POST'])
def fertilizer_recommendation():
    """
    Endpoint for Fertilizer Recommendation (Lookup/Prediction)
    Inputs: crop
    Output: recommended_ratio (N, P, K)
    """
    if not FERTILIZER_RATIOS:
        return jsonify({"recommended_ratio": None, "error": "Fertilizer ratio data is not loaded."}), 500

    try:
        data = request.get_json()
        crop_name = data.get('crop', '').lower()

        if not crop_name:
            return jsonify({"error": "Missing 'crop' name"}), 400
        
        # --- LOOKUP LOGIC ---
        ratio = FERTILIZER_RATIOS.get(crop_name)
        
        if ratio:
            # Ensure the ratio keys are returned exactly as expected by the frontend (N, P, K)
            return jsonify({
                "recommended_ratio": {
                    "N": ratio.get("N"),
                    "P": ratio.get("P"),
                    "K": ratio.get("K")
                },
                "error": None
            })
        else:
            return jsonify({"recommended_ratio": None, "error": f"No fertilizer data found for crop: {crop_name}"}), 404

    except Exception as e:
        print(f"Fertilizer Lookup Error: {e}")
        return jsonify({"recommended_ratio": None, "error": str(e)}), 500

if __name__ == '__main__':
    # Running locally for debugging
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000)
