from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# ---------------- Load Model ----------------
try:
    best_model = joblib.load('best_cci_model.pkl')
    le = joblib.load('cci_label_encoder.pkl')
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    best_model = None
    le = None

# ---------------- Routes ----------------

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['age', 'bmi', 'oa_severity', 'activity', 'smoking', 'pain_score']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Prepare input data
        input_data = pd.DataFrame([[
            data['age'],
            data['bmi'],
            data['oa_severity'],
            data['activity'],
            data['smoking'],
            data['pain_score']
        ]], columns=['age', 'bmi', 'oa_severity', 'activity', 'smoking', 'pain_score'])
        
        # Make prediction
        pred_encoded = best_model.predict(input_data)[0]
        pred_label = le.inverse_transform([pred_encoded])[0]
        
        # Get probabilities
        probabilities = {}
        if hasattr(best_model, "predict_proba"):
            probs = best_model.predict_proba(input_data)[0]
            for i, class_label in enumerate(le.classes_):
                probabilities[class_label] = float(probs[i])
        else:
            # If model doesn't support probability, return 100% for predicted class
            probabilities[pred_label] = 1.0
        
        # Prepare response
        response = {
            'predicted_category': pred_label,
            'probabilities': probabilities,
            'input_data': {
                'age': data['age'],
                'bmi': data['bmi'],
                'oa_severity': data['oa_severity'],
                'activity': data['activity'],
                'smoking': data['smoking'],
                'pain_score': data['pain_score']
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_status = "loaded" if best_model is not None else "not loaded"
    return jsonify({
        'status': 'healthy',
        'model_status': model_status
    })

# ---------------- Error Handlers ----------------

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ---------------- Run App ----------------

if __name__ == '__main__':
    # Check if static and templates folders exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("Created 'templates' folder")
    
    if not os.path.exists('static'):
        os.makedirs('static')
        print("Created 'static' folder")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)