from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

# loading model
model = pickle.load(open('model.pkl','rb'))

# creating app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    #   cement    blastFurnace    flyAsh    water    superplasticizer    courseAggregate    fineaggregate    age
    cement = float(request.form['cement'])
    blastFurnace = float(request.form['blastFurnace'])
    flyAsh = float(request.form['flyAsh'])
    water = float(request.form['water'])
    superplasticizer = float(request.form['superplasticizer'])
    courseAggregate = float(request.form['courseAggregate'])
    fineaggregate = float(request.form['fineaggregate'])
    age = int(request.form['age'])

    # transform input features
    features = np.array([cement, blastFurnace, flyAsh, water, superplasticizer, courseAggregate, fineaggregate, age]).reshape(1, -1)
    prediction = model.predict(features).reshape(1, -1)

    return render_template('index.html', strength=prediction[0][0])

# python main
if __name__ == "__main__":
    app.run(debug=True)
