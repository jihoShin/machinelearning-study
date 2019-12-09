# Import libraries
import numpy as np
from flask import Flask, request, jsonify
from sklearn.externals import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('C://Users//jiho87.shin//Documents//GitHub//machinelearning-study//dicisiontree_model.joblib')

@app.route('/api',methods=['GET'])
def predict():
    # Get the data from the POST request.
    print("predict")

    sampleData = [[0,1,3,4]]

    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict(sampleData)

    # Take the first value of prediction
    output = prediction[0]
    result = {'prediction':int(output)}
    return jsonify(result)

if __name__ == '__main__':
    try:
    	app.run(port=5000, debug=True)
    except:
    	print("Server is exited unexpectedly. Please contact server admin.")
