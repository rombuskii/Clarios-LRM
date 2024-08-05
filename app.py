from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
import math

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
prob_encoder = joblib.load('encoder.pkl')
prob_model = joblib.load('logistic_regression_model.pkl')
date_encoder =  joblib.load('encoder_dt.pkl')
date_model = joblib.load('decision_tree_model.pkl')


def preprocess_data(sku, quantity, date_placed, actual_shipment_date):
    # Convert inputs to DataFrame
    data = pd.DataFrame({
        'sku': [sku],
        'quantity': [quantity],
        'days_to_ship': [(pd.to_datetime(actual_shipment_date) - pd.to_datetime(date_placed)).days]
    })
    
    # One-hot encode the 'sku' column
    sku_encoded = prob_encoder.transform(data[['sku']])
    sku_encoded_df = pd.DataFrame(sku_encoded.toarray(), columns=prob_encoder.get_feature_names_out(['sku']))
    
    # Combine encoded features with the original features
    data_final = pd.concat([data.drop(columns=['sku']), sku_encoded_df], axis=1)
    
    return data_final

def preprocess_getDate(sku, quantity, actual_shipment_date):
    data = pd.DataFrame({
        'sku': [sku],
        'quantity': [quantity],
        'actualShipmentDate': [actual_shipment_date]
    })
    print(data)

    # One-hot encode 'sku'
    df_encoded = date_encoder.transform(data[['sku']])
    df_encoded_df = pd.DataFrame(df_encoded.toarray(), columns=date_encoder.get_feature_names_out(['sku']))

    # Combine the encoded features with other features
    data_final = pd.concat([data.drop(columns=['sku']).reset_index(drop=True), df_encoded_df.reset_index(drop=True)], axis=1)

    # Convert 'actualShipmentDate' to numeric (UNIX timestamp)
    data_final['actualShipmentDate'] = pd.to_datetime(data_final['actualShipmentDate']).astype(int) / 10**9
    
    return data_final

@app.route('/', methods=['GET'])
def hello():
    return "Clarios Logistic Regression Model"


@app.route('/getDate', methods=['POST'])
def getDate():
    # Get data from request
    data = request.json
    sku = data.get('sku')
    quantity = data.get('quantity')
    print(quantity)
    actual_shipment_date = data.get('actualShipmentDate')

    if not sku:
        sku = 'HE-FE'
    
    if not quantity:
        quantity = 151218

    # Preprocess the data
    processed_data = preprocess_getDate(sku, quantity, actual_shipment_date)
    
    # Make prediction and get probability
    days_to_ship = date_model.predict(processed_data)

    if (quantity and int(quantity) > 100000):
        quantity = int(quantity)
        prediction = pd.to_datetime(actual_shipment_date) - pd.Timedelta(days=days_to_ship[0] + int(quantity / 100000))
    elif (quantity and int(quantity) < 1000):
        quantity = int(quantity)
        prediction = pd.to_datetime(actual_shipment_date) - pd.Timedelta(days=days_to_ship[0] - (7 - int(7 * (quantity / 100000))))
    else:
        prediction = pd.to_datetime(actual_shipment_date) - pd.Timedelta(days=days_to_ship[0])

    return jsonify({'days_to_ship': prediction.strftime('%m/%d/%Y')})


@app.route('/predict', methods=['POST'])
def predict():
    model = joblib.load('logistic_regression_model.pkl')
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
    probability = prob_model.predict_proba(processed_data)[0][1]  # Probability of being 'Completed'
    
    # Convert probability to percentage
    probability_percentage = probability * 100
    
    return jsonify({'completed_probability_percentage': probability_percentage})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
