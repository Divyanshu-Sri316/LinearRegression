from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the pickled model
with open('model.pkl', 'rb') as file:
    pickled_model = pickle.load(file)

# Simple linear regression model using the pickled model
def linear_regression(x):
    df_train = pd.DataFrame({'x': [x]})
    return pickled_model.predict((df_train[['x']].values).reshape(-1, 1))[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    if not data or 'input_value' not in data:
        return jsonify({"error": "Invalid input. Please provide 'input_value' in the request body."}), 400

    try:
        input_value = float(data['input_value'])
    except ValueError:
        return jsonify({"error": "Invalid input. 'input_value' must be a number."}), 400

    result = linear_regression(input_value)
    return jsonify({"input": input_value, "output": result})

if __name__ == '__main__':
    app.run(debug=True)