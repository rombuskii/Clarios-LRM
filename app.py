from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

port = int(os.environ.get('PORT', 8000))
app = Flask(__name__)

cors = CORS(app, resources={
    r"/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Load the trained model and encoder
model = joblib.load('logistic_regression_model.pkl')
encoder = joblib.load('encoder.pkl')

def preprocess_data(sku, quantity, date_placed, actual_shipment_date):
    # Convert inputs to DataFrame
    data = pd.DataFrame({
        'sku': [sku],
        'quantity': [quantity],
        'days_to_ship': [(pd.to_datetime(actual_shipment_date) - pd.to_datetime(date_placed)).days]
    })
    
    # One-hot encode the 'sku' column
    sku_encoded = encoder.transform(data[['sku']])
    sku_encoded_df = pd.DataFrame(sku_encoded.toarray(), columns=encoder.get_feature_names_out(['sku']))
    
    # Combine encoded features with the original features
    data_final = pd.concat([data.drop(columns=['sku']), sku_encoded_df], axis=1)
    
    return data_final

@app.route('/', methods=['GET'])
def hello():
    return "Clarios Logistic Regression Model"

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.json
    sku = data.get('sku')
    quantity = data.get('quantity')
    date_placed = data.get('datePlaced')
    actual_shipment_date = data.get('actualShipmentDate')

    if(not sku):
        sku = 'HE-FE'
    
    if(not quantity):
        quantity = 151218

    # Preprocess the data
    processed_data = preprocess_data(sku, quantity, date_placed, actual_shipment_date)
    
    # Make prediction and get probability
    probability = model.predict_proba(processed_data)[0][1]  # Probability of being 'Completed'
    
    # Convert probability to percentage
    probability_percentage = probability * 100
    
    return jsonify({'completed_probability_percentage': probability_percentage})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port)
