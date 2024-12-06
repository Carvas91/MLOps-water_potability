import json
import requests

url = "https://mlops-water-potability.onrender.com/predict"

x_new = dict(
    ph = 1,
    Hardness = 1,
    Solids = 0,
    Chloramines = 1,
    Sulfate = 0,
    Conductivity = 0,
    Organic_carbon = 0,
    Trihalomethanes = 0,
    Turbidity = 0
)

x_new_json = json.dumps(x_new)

response = requests.post(url=url, data=x_new_json)

print(f"Post response text:{response.text}")

print(f"Post status code:{response.status_code}")