import csv
import time
import random
from datetime import datetime, timedelta
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def read_predicted_consumption(file_path):
    """Reads predicted hourly consumption from a CSV file."""
    consumption = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            consumption.append(float(row['Global_active_power']))
    return consumption

def generate_real_time_data(predicted_consumption):
    real_time_data = []
    for predicted_value in predicted_consumption:
        variation = random.uniform(-0.2, 0.5)  
        real_time_value = predicted_value + variation
        real_time_data.append(max(0, real_time_value))
    return real_time_data

def compare_consumption(predicted_consumption, real_time_consumption):
    start_time = datetime.now()
    for minute, (predicted, real) in enumerate(zip(predicted_consumption, real_time_consumption)):
        current_time = start_time + timedelta(seconds=minute*60)

        if real > predicted * 1.5:
            print(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} - Consumption is much higher: Predicted {predicted:.2f}, Real {real:.2f}   ALERT")
            return {'color': '#ff4847'}

        else:
            print(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} - Consumption is normal: Predicted {predicted:.2f}, Real {real:.2f}")
            return {'color': '#3cb371'}

        time.sleep(0.2)  

@app.route('/')
def get_color():
    file_path = '../Backend/outputs/hourly_consumption.csv'
    predicted_consumption = read_predicted_consumption(file_path)
    real_time_consumption = generate_real_time_data(predicted_consumption)
    return jsonify(compare_consumption(predicted_consumption, real_time_consumption))

if __name__ == "__main__":
    app.run()