from flask import Flask, request, jsonify
import pickle
import numpy as np
import json

app = Flask(__name__) 

global cols
cols = ['age',
 'sex',
 'cp',
 'trestbps',
 'chol',
 'fbs',
 'restecg',
 'thalach',
 'exang',
 'oldpeak',
 'slope',
 'ca',
 'thal']

def open_model(filename = '/home/hanna/Downloads/rfc_heart.pkl'):
    """ Given filename, returns the model 
    param filename: filepath for model
    return : the model from pickle """
    with open(filename, 'rb') as file:
        pickle_model = pickle.load(file)
    return pickle_model

def use_pickle(X):
    
    """ Given X, returns a prediction 
    param X: the feature values to consider
    returns pred_pickle: the prediction """
    if not isinstance(X, np.ndarray):
        X = np.array(eval(str(X)))
    if len(X.shape) == 1:
        X = X[np.newaxis , :]  
    pred_pickle = model.predict(X)
    return str(pred_pickle[0:X.shape[0]])

@app.route('/')
def enter():
    return 'Welcome!'
@app.route('/predict_single')
def predict_single():
    """ Returns prediction for given parameters"""
    pred_array = np.zeros(len(cols))
    
    for i, col in enumerate(cols):
        pred_array[i] = request.args.get(col)
    return use_pickle(pred_array)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = use_pickle(data)

    return jsonify(prediction)


if __name__ == "__main__":
    global model
    model = open_model()
    app.run()
