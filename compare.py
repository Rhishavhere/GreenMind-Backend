import csv
import time
import random
from datetime import datetime, timedelta

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
        variation = random.uniform(-0.2, 1)  
        real_time_value = predicted_value + variation
        real_time_data.append(max(0, real_time_value))
    return real_time_data

def compare_consumption(predicted_consumption, real_time_consumption):
    start_time = datetime.now()
    for hour, (predicted, real) in enumerate(zip(predicted_consumption, real_time_consumption)):
        current_time = start_time + timedelta(seconds=hour)

        if real > predicted * 1.5:
            print(f"Alert! {current_time.strftime('%Y-%m-%d %H:%M:%S')} - Consumption is much higher: Predicted {predicted:.2f}, Real {real:.2f}")
        else:
            print(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} - Consumption is normal: Predicted {predicted:.2f}, Real {real:.2f}")
        time.sleep(1)  

if __name__ == "__main__":
    file_path = '../Backend/outputs/hourly_consumption.csv'
    predicted_consumption = read_predicted_consumption(file_path)
    real_time_consumption = generate_real_time_data(predicted_consumption)
    compare_consumption(predicted_consumption, real_time_consumption)