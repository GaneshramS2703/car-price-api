import requests

url = "http://192.168.0.175:5000/predict"
data = {
    "features": [2015, 60000, 1, 0, 0, 1, 0, 1, 0]
}

response = requests.post(url, json=data)
print(response.json())  # Should return the predicted price
