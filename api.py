from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS


model = joblib.load("your_trained_model.pkl")

app = Flask(__name__)
CORS(app)  

@app.route('/predict', methods=['POST'])
def predict():

    input_data = request.json
    

    df = pd.DataFrame([input_data])
    

    predictions = model.predict(df)
    

    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
