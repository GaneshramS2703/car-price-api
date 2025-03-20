import requests

url = "https://flask-car-price-api-bgdzgdc7faaheada.canadacentral-01.azurewebsites.net/predict"
data = {"features": [2015, 60000, 1, 0, 0, 1, 0, 1, 0]}

response = requests.post(url, json=data)
print(response.json())  # Should return the predicted price
