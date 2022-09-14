import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import datetime as dt

# Create flask app
app = Flask(__name__)


@app.route("/", methods = ["POST","GET"])
def Home():
    return jsonify("Hello World")

@app.route("/predict", methods = ["GET","POST"])
def predict():

    x_future_date = pd.date_range(start ="2022-08-01", end = "2023-01-31")

    x_future_dates = pd.DataFrame()

    x_future_dates["Dates"] = pd.to_datetime(x_future_date)

    x_future_dates.index = x_future_dates["Dates"]
    df1 = x_future_dates

    def create_features(df1, label=None):
        df1['date'] = df1.index
        df1['hour'] = df1['date'].dt.hour
        df1['dayofweek'] = df1['date'].dt.dayofweek
        df1['quarter'] = df1['date'].dt.quarter
        df1['month'] = df1['date'].dt.month
        df1['year'] = df1['date'].dt.year
        df1['dayofyear'] = df1['date'].dt.dayofyear
        df1['dayofmonth'] = df1['date'].dt.day
        df1['weekofyear'] = df1['date'].dt.weekofyear
        
        X = df1[['hour','dayofweek','quarter','month','year',
            'dayofyear','dayofmonth','weekofyear']]
        if label:
            y = df1[label]
            return X, y
        return X

    X, y = create_features(x_future_dates, label='Dates')

    # First model pickle

    model_xg = pickle.load(open("xgmodel.pkl", "rb"))

    y_future_total_tickets = model_xg.predict(X)

    #print(y_future_total_tickets)

    # 2nd model pickle
    x_future_dates["Predicted Tickets"] = y_future_total_tickets
    x_future_dates.drop("Dates", inplace = True, axis = 1)
    model_mr = pickle.load(open("mrmodel.pkl", "rb"))
    y_future_prediction = model_mr.predict(np.array(x_future_dates["Predicted Tickets"]).reshape(184,1))
    #print(y_future_prediction)


    #converting nparray to list to finally convert it into json
    future_pred = y_future_prediction.tolist()
    return json.dumps(future_pred)

if __name__ == "__main__":
        app.run()
    #final_jsonformat = json.dumps(future_pred)

    #print(final_jsonformat)

