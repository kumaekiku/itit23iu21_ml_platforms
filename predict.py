#!/usr/bin/env python
# coding: utf-8

import pickle

import pandas as pd
from flask import Flask, jsonify, request

input = 'wrap_model.bin'
with open(input, 'rb') as f_in:
    wrapper = pickle.load(f_in)

app = Flask('RetentionPredict')

@app.route('/predict', methods=['POST'])
def predict():
    employee = pd.DataFrame([request.get_json()])

    y_pred = wrapper.model.predict(employee)
    y_pred_proba = wrapper.model.predict_proba(employee)[:,1]

    result = {
        'retention': int(y_pred),
        'retention_proba': float(y_pred_proba)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
