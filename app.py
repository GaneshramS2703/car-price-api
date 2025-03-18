from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

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

        # Convert input to DataFrame and align columns
        input_df = pd.DataFrame([input_features[0]], columns=feature_names[:len(input_features[0])])
        input_df = input_df.reindex(columns=feature_names, fill_value=0)  # Fill missing columns

        # Predict price
        prediction = model.predict(input_df)

        return jsonify({'predicted_price': float(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
