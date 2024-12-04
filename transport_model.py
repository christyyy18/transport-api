import pandas as pd
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

app = Flask(__name__)

# Load pre-trained model and scalers
model = tf.keras.models.load_model('transport_model.h5')
scaler = StandardScaler()
scaler_y = MinMaxScaler()

# Define categories
categories = ['Bike', 'Motorcycle', 'Private', 'Public', 'Walk']

@app.route('/predict_transport', methods=['POST'])
def predict_transport():
    """Predict carbon emissions for a transport mode."""
    try:
        data = request.get_json()
        category = data.get('category')
        distance_km = data.get('distance_km')

        if category not in categories:
            return jsonify({'error': f"Invalid category! Choose from: {', '.join(categories)}"}), 400
        if not isinstance(distance_km, (int, float)) or distance_km <= 0:
            return jsonify({'error': 'Distance must be a positive number'}), 400

        # Encode category
        category_encoded = [0, 0, 0, 0, 0]
        category_encoded[categories.index(category)] = 1

        # Prepare input data
        input_data = category_encoded + [distance_km]
        input_df = pd.DataFrame([input_data], columns=['Bike', 'Motorcycle', 'Private', 'Public', 'Walk', 'Distance (km)'])
        input_data_scaled = scaler.transform(input_df)

        # Predict emission
        predicted_emission_scaled = model.predict(input_data_scaled)
        predicted_emission = scaler_y.inverse_transform(predicted_emission_scaled.reshape(-1, 1))[0][0]

        return jsonify({
            'category': category,
            'distance_km': distance_km,
            'predicted_emission': round(predicted_emission, 2)
        }), 200

    except Exception as e:
        app.logger.error(f"Error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)