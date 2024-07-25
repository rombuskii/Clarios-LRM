from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

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

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.json
    sku = data.get('sku')
    quantity = data.get('quantity')
    date_placed = data.get('datePlaced')
    actual_shipment_date = data.get('actualShipmentDate')

    # Preprocess the data
    processed_data = preprocess_data(sku, quantity, date_placed, actual_shipment_date)
    
    # Make prediction and get probability
    probability = model.predict_proba(processed_data)[0][1]  # Probability of being 'Completed'
    
    # Convert probability to percentage
    probability_percentage = probability * 100
    
    return jsonify({'completed_probability_percentage': probability_percentage})

if __name__ == '__main__':
    app.run(debug=True)