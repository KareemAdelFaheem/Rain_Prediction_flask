from flask import Flask
import joblib
import numpy as np
import pandas as pd
import math
import pickle
import xgboost
from sklearn.preprocessing import StandardScaler
import json
import requests
import time
import threading

app = Flask(__name__)
model=joblib.load(open('xgb_model.pkl','rb'))

# mice_imputer = joblib.load('mice_imputer.pkl')

with open('encoders.pkl', 'rb') as f:
    lencoders = pickle.load(f)

scaler = joblib.load('scaler.pkl')

def encode_data(data):
    for col in data.select_dtypes(include=['object']).columns:
        if col in lencoders:
            # Apply the encoder to the new data
            data[col] = lencoders[col].transform(data[col])
    return data

def preprocessing(input_data):
    encoded = encode_data(input_data)
    final_data = scaler.transform(encoded)
    return final_data

base_url = "https://api.open-meteo.com/v1/forecast"
params = {"latitude":"30.04442","longitude":"31.23571","daily":"sunset,sunrise","hourly":"evapotranspiration","current":"cloud_cover,wind_speed_10m,wind_direction_10m,wind_gusts_10m,precipitation,rain","timezone":"Africa/Cairo","forecast_days":"1"}

def get_weather():

    try:
        response = requests.get(base_url,params=params)
        response.raise_for_status()
        response=response.json()
        return response
    except requests.exceptions.HTTPError as http_error:
        print(f"status code error {http_error}")
    except Exception as e:
        print(f"error Occurred retrieving API {e}")
    return None



@app.route('/predict',methods=['GET','POST'])
def predict():
    # input_data = pd.DataFrame([["Albury", 14.0, 18.0, 10.2, 5.0, 1.2, "S", 37, "SE", "S", 7, 15, 95,
    #                             80, 1005.2, 1003.8, 7, 8, 15.5, 17.2, 1]],
    #                           columns=["Location", "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
    #                                    "WindGustDir", "WindGustSpeed", "WindDir9am",
    #                                    "WindDir3pm", "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm",
    #                                    "Pressure9am", "Pressure3pm", "Cloud9am",
    #                                    "Cloud3pm", "Temp9am", "Temp3pm", "RainToday"])

    # input_data = pd.DataFrame([["Melbourne", 15.5, 22.0, 0.0, 3.0, 8.5, "NE", 20, "N", "E", 5, 10, 50,
    #                             40, 1012.5, 1010.3, 5, 6, 18.0, 21.0, 0]],
    #                           columns=["Location", "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
    #                                    "WindGustDir", "WindGustSpeed", "WindDir9am", "WindDir3pm", "WindSpeed9am",
    #                                    "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
    #                                    "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm", "RainToday"])

    input_data = pd.DataFrame([["Albury", 18.0, 25.5, 0.0, 4.2, 9.0, "E", 15, "NNE", "NE", 4, 7, 45,
                                35, 1015.0, 1013.5, 4, 5, 20.0, 23.0, 0]],
                              columns=["Location", "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
                                       "WindGustDir", "WindGustSpeed", "WindDir9am", "WindDir3pm", "WindSpeed9am",
                                       "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
                                       "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm", "RainToday"])

    # input_data = pd.DataFrame([["Albury", 12.5, 18.0, 2.5, 3.0, 5.0, "NE", 20, "E", "ENE", 8, 12, 72,
    #                             80, 1010.5, 1008.9, 6, 7, 15.0, 16.5, 1]],
    #                           columns=["Location", "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
    #                                    "WindGustDir", "WindGustSpeed", "WindDir9am", "WindDir3pm", "WindSpeed9am",
    #                                    "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
    #                                    "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm", "RainToday"])

    # Encode the data
    # encoded = encode_data(input_data)
    # imputed_data = mice_imputer.transform(encoded)
    # final_data= scaler.transform(encoded)

    # returned_value=get_temp()
    final_data=preprocessing(input_data)
    prediction= model.predict(final_data)
    weather_data = get_weather()
    print(weather_data["current"]["cloud_cover"])
    return "prediction result %s "% prediction


if __name__ == '__main__':
    app.run(debug=True)

# def get_temp():
#     # Simulating WiFi scan (This part will vary depending on the platform)
#     # For example, using a library to scan WiFi networks (like wifi or pywifi)
#     networks_found = scan_wifi_networks()
#
#     if networks_found == 0:
#         print("No networks found.")
#         return
#
#     # Prepare JSON payload
#     wifi_access_points = []
#     for i in range(min(networks_found, 3)):  # Only process up to 3 networks
#         network = {
#             "macAddress": get_bssid(i),  # You'd need to implement get_bssid()
#             "signalStrength": get_rssi(i)  # You'd need to implement get_rssi()
#         }
#         wifi_access_points.append(network)
#
#     # Create JSON payload
#     payload = {
#         "wifiAccessPoints": wifi_access_points
    # }
    #
    # # Convert to JSON string
    # json_payload = json.dumps(payload)
    #
    # # Send HTTP POST request to OpenWeatherMap API
    # # url = "http://api.openweathermap.org/geo/1.0/direct?q=London&limit=5&appid=f91cb1579a0d0ea623150046a5d9e10c"
    #
    # # headers = {"Content-Type": "application/json"}
    #
    # try:
    #     # response = requests.get(url, headers=headers, data=json_payload)
    #     # response = requests.get(f'https://api.openweathermap.org/data/2.5/weather?lat=30.0443879&lon=31.2357257&appid=d10dd0ebe1faa44c412c735463f8c225')
    #     response = requests.get(f'https://api.open-meteo.com/v1/forecast?latitude=30.04442&longitude=31.23571&hourly=cloud_cover,evapotranspiration,wind_speed_10m,wind_speed_80m,wind_direction_10m,wind_speed_120m,wind_speed_180m,wind_direction_80m,wind_direction_120m,wind_direction_180m,wind_gusts_10m,precipitation,rain&forecast_days=1')
    #     #
        # if response.status_code == 200:
        #     data = response.json()
        #     # print(response.status_code)
        #     print("wind speed is",data["hourly"]["wind_speed_10m"][21])
        #     print("clouds are",data["hourly"]["cloud_cover"][21])
        #
        #     return "well"
        # else:
        #     print(f"Request failed with status code {response.status_code}")
#             return "error"
#
#     except Exception as e:
#         print(f"Exception occurred: {e}")
#
# def scan_wifi_networks():
#     # Implement WiFi scanning logic using platform-specific libraries.
#     # Return the number of networks found.
#     return 3  # Example placeholder, modify according to actual scanning method
#
# def get_bssid(index):
#     # Return BSSID for a given index (stub function for illustration)
#     return "00:14:22:01:23:45"
#
# def get_rssi(index):
#     # Return RSSI for a given index (stub function for illustration)
#     return -50  # Example RSSI value

