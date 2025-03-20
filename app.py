from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from azure.storage.blob import BlobServiceClient
import os

app = Flask(__name__)

# Azure Blob Storage Configuration
AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=carpricepredictiondata;AccountKey=b7BMz7ai4Ym/qc2SzmHLpOwmD/ZzlTqGgK/A+QSiByAYzIWkkBovJuyr03y+tB6KxZpV0/lcb1ja+AStW7Th9g==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "datasets"
BLOB_NAME = "used_cars.csv"

# Function to download dataset from Azure Blob Storage
def download_csv_from_blob():
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=BLOB_NAME)

    # Download the CSV file to a local buffer
    csv_data = blob_client.download_blob().readall()

    # Convert to DataFrame
    from io import StringIO
    df = pd.read_csv(StringIO(csv_data.decode("utf-8")))

    return df

# Load trained model
with open("random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load feature names
with open("feature_names.pkl", "rb") as file:
    feature_names = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_features = np.array(data['features']).reshape(1, -1)

        # Download dataset from Azure Blob Storage
        df = download_csv_from_blob()

        # Convert input to DataFrame and align columns
        input_df = pd.DataFrame([input_features[0]], columns=feature_names[:len(input_features[0])])
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # Predict price
        prediction = model.predict(input_df)

        return jsonify({'predicted_price': float(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
