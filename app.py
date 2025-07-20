import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from datetime import datetime


flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    number_of_weekend_nights = float(request.form['number_of_weekend_nights'])
    number_of_week_nights = float(request.form['number_of_week_nights'])
    lead_time = float(request.form['lead_time'])
    average_price = float(request.form['average_price'])
    special_requests = float(request.form['special_requests'])
    type_of_meal = request.form['type_of_meal']
    room_type = request.form['room_type']
    market_segment_type = request.form['market_segment_type']
    reservation_season = None 

    reservation_date = request.form['reservation_date']
    date_obj = datetime.strptime(reservation_date, "%Y-%m-%d")
    reservation_day = date_obj.day
    reservation_month = date_obj.month
    reservation_weekday = date_obj.weekday()
    reservation_week = date_obj.isocalendar()[1]
    reservation_is_weekend = 1 if reservation_weekday >= 5 else 0
    
    if reservation_month in [12, 1, 2]:
        reservation_season = 'Winter'
    elif reservation_month in [3, 4, 5]:
        reservation_season = 'Spring'
    elif reservation_month in [6, 7, 8]:
        reservation_season = 'Summer'
    else:
        reservation_season = 'Fall'

    meal_dummies = [
        1 if type_of_meal == 'Meal Plan 2' else 0,
        1 if type_of_meal == 'Meal Plan 3' else 0,
        1 if type_of_meal == 'Not Selected' else 0
    ]
    room_dummies = [
        1 if room_type == 'Room_Type 2' else 0,
        1 if room_type == 'Room_Type 3' else 0,
        1 if room_type == 'Room_Type 4' else 0,
        1 if room_type == 'Room_Type 5' else 0,
        1 if room_type == 'Room_Type 6' else 0,
        1 if room_type == 'Room_Type 7' else 0
    ]
    market_dummies = [
        1 if market_segment_type == 'Complementary' else 0,
        1 if market_segment_type == 'Corporate' else 0,
        1 if market_segment_type == 'Offline' else 0,
        1 if market_segment_type == 'Online' else 0
    ]
    season_dummies = [
        1 if reservation_season == 'Spring' else 0,
        1 if reservation_season == 'Summer' else 0,
        1 if reservation_season == 'Winter' else 0
    ]

    input_features = [
        number_of_weekend_nights,
        number_of_week_nights,
        lead_time,
        average_price,
        special_requests,
        reservation_day,
        reservation_month,
        reservation_weekday,
        reservation_week,
        reservation_is_weekend,
        *meal_dummies,
        *room_dummies,
        *market_dummies,
        *season_dummies
    ]

    prediction = model.predict([input_features])[0]
    if prediction == 1:
        result = "Booking Not Cancelled"
        prediction_class = "success"
    else:
        result = "Booking Cancelled"
        prediction_class = "danger"
    return render_template("index.html", prediction_text=f"{result}", prediction_class=prediction_class)

if __name__ == "__main__":
    flask_app.run(debug=True)