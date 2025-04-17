from flask import Flask, jsonify, request
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
params = {"latitude":"30.04442","longitude":"31.23571","daily":"sunset,sunrise,sunshine_duration","hourly":"evapotranspiration,rain,precipitation,wind_speed_10m,wind_direction_10m,wind_gusts_10m,cloud_cover","current":"cloud_cover,wind_speed_10m,wind_direction_10m,wind_gusts_10m,rain,precipitation","timezone":"Africa/Cairo","forecast_days":"1"}

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



@app.route('/',methods=['GET','POST'])
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
    #
    # input_data = pd.DataFrame([["Albury", 18.0, 25.5, 0.0, 4.2, 9.0, "E", 15, "NNE", "NE", 4, 7, 45,
    #                             35, 1015.0, 1013.5, 4, 5, 20.0, 23.0, 0]],
    #                           columns=["Location", "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
    #                                    "WindGustDir", "WindGustSpeed", "WindDir9am", "WindDir3pm", "WindSpeed9am",
    #                                    "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
    #                                    "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm", "RainToday"])

    # input_data = pd.DataFrame([["Albury", 12.5, 18.0, 2.5, 3.0, 5.0, "NE", 20, "E", "ENE", 8, 12, 72,
    #                             80, 1010.5, 1008.9, 6, 7, 15.0, 16.5, 1]],
    #                           columns=["Location", "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
    #                                    "WindGustDir", "WindGustSpeed", "WindDir9am", "WindDir3pm", "WindSpeed9am",
    #                                    "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
    #                                    "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm", "RainToday"])


    try:
        # for getting json from ESP32
        try:
            response_data = request.get_json()
            mintemp = response_data.get('mintemp')
            maxtemp = response_data.get('maxtemp')
            hum9am = response_data.get('hum9am')
            hum3pm = response_data.get('hum3pm')
            pressure9am = response_data.get('pressure9am')
            pressure3pm = response_data.get('pressure3pm')
            temp9am = response_data.get('temp9am')
            temp3pm = response_data.get('temp3pm')
            raintoday = response_data.get('raintoday')


        except Exception as e:
            return "exception from ESP32 is %s"%e


        # for getting data from params
        # mintemp = request.args.get('mintemp', type=float)
        # maxtemp = request.args.get('maxtemp', type=float)
        # hum9am = request.args.get('hum9am', type=float)
        # hum3pm = request.args.get('hum3pm', type=float)
        # pressure9am = request.args.get('pressure9am', type=float)
        # pressure3pm = request.args.get('pressure3pm', type=float)
        # temp9am = request.args.get('temp9am', type=float)
        # temp3pm = request.args.get('temp3pm', type=float)
        # raintoday = request.args.get('raintoday', type=int)

        # for 1 prediction result
        # mintemp = 12.5
        # maxtemp = 18
        # hum9am =72
        # hum3pm = 80
        # pressure9am = 1010.5
        # pressure3pm = 1008.9
        # temp9am = 15
        # temp3pm = 16.5
        # raintoday = 1



        # #for getting data values from API
        # weather = get_weather()
        # Rainfall= weather["current"]["rain"]
        # Evaporation= weather["hourly"]["evapotranspiration"][21]
        # Sunshine= weather["daily"]["sunshine_duration"][0]
        # WindGustDir= weather["current"]["wind_gusts_10m"]
        # WindGustSpeed= weather["current"]["wind_speed_10m"]
        # WindDir9am= weather["hourly"]["wind_direction_10m"][21]
        # WindDir3pm= weather["hourly"]["wind_direction_10m"][15]
        # WindSpeed9am= weather["hourly"]["wind_speed_10m"][21]
        # WindSpeed3pm= weather["hourly"]["wind_speed_10m"][15]
        # Cloud9am= weather["hourly"]["cloud_cover"][21]
        # Cloud3pm= weather["hourly"]["cloud_cover"][15]
        # currentCloud= weather["current"]["cloud_cover"]

        #for 1 prediction result
        Rainfall = 2.5
        Evaporation = 3
        Sunshine = 5
        WindGustDir = "NE"
        WindGustSpeed = 20
        WindDir9am = "E"
        WindDir3pm = "ENE"
        WindSpeed9am = 8
        WindSpeed3pm = 12
        Cloud9am = 6
        Cloud3pm = 7
        # currentCloud = weather["current"]["cloud_cover"]

        columns = ["Location", "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
                   "WindGustDir", "WindGustSpeed", "WindDir9am", "WindDir3pm", "WindSpeed9am",
                   "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
                   "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm", "RainToday"]
        input_data = pd.DataFrame([[
            "Albury",
            mintemp,
            maxtemp,
            Rainfall,
            Evaporation,
            Sunshine,
            WindGustDir,
            WindGustSpeed,
            WindDir9am,
            WindDir3pm,
            WindSpeed9am,
            WindSpeed3pm,
            hum9am,
            hum3pm,
            pressure9am,
            pressure3pm,
            Cloud9am,
            Cloud3pm,
            temp9am,
            temp3pm,
            raintoday

        ]], columns=columns)



        final_data = preprocessing(input_data)
        final_data_json = final_data.tolist()
        prediction = model.predict(final_data)


        return jsonify({
            "status": "success",
            "final_data": final_data_json,
            "prediction": float(prediction[0]),
            "weather values": {
                "mintemp":mintemp,
                "maxtemp":maxtemp ,
                "hum9am":hum9am,
                "hum3pm":hum3pm,
                "pressure9am":pressure9am,
                "pressure3pm":pressure3pm,
                "temp9am":temp9am,
                "temp3pm":temp3pm,
                "raintoday":raintoday,
                "Rainfall": Rainfall,
                "Evaporation":Evaporation,
                "Sunshine":Sunshine,
                "WindGustDir":WindGustDir ,
                "WindGustSpeed":WindGustSpeed ,
                "WindDir9am":WindDir9am ,
                "WindDir3pm":WindDir3pm ,
                "WindSpeed9am": WindSpeed9am,
                "WindSpeed3pm":WindSpeed3pm ,
                "Cloud9am":Cloud9am,
                "Cloud3pm":Cloud3pm ,

            }
        })
    except Exception as e:
        return jsonify({"status":"error","message":f"{e}"})


if __name__ == '__main__':
    app.run(debug=True)