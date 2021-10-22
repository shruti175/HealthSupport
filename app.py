from flask import Flask, render_template, request, flash, redirect
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)

def predict(values, dic):
    if len(values) == 8:
        model = pickle.load(open('models/diabetes.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 30:
        model = pickle.load(open('models/cancer.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 13:
        model = pickle.load(open('models/heart.pkl','rb'))
        input_features = [float(x) for x in request.form.values()]
        features_value = [np.array(input_features)]

        features_name = ['sex', 'age', 'cp', 'trestbps',
                         'chol', 'fbs', 'restecg',
                         'thalach', 'exang', 'oldpeak',
                         'slope', 'ca', 'thal']

        df = pd.DataFrame(features_value, columns=features_name)
        output = model.predict(df)
        if output == 0:
            pred = 1
        else:
            pred = 0
        return pred
    elif len(values) == 18:
        model = pickle.load(open('models/kidney.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 10:
        model = pickle.load(open('models/liver.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/heart')
def heart():
    return render_template('heart.html')

@app.route('/liver')
def liver():
    return render_template('liver.html')

@app.route('/kidney')
def kidney():
    return render_template('kidney.html')

@app.route('/cancer')
def cancer():
    return render_template('cancer.html')

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

def name(values, dic):
    if len(values) == 8:
        disease="Diabetes"
        return disease
    elif len(values) == 30:
        disease="Breast Cancer"
        return disease
    elif len(values) == 13:
        disease = "Heart Disease"
        return disease
    elif len(values) == 18:
        disease = "Kidney Disease"
        return disease
    elif len(values) == 10:
        disease = "Liver Disease"
        return disease

@app.route("/predict", methods = ['POST', 'GET'])
def predictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list, to_predict_dict)
            disease = name(to_predict_list, to_predict_dict)
    except:
        message = "Please enter valid Data"
        return render_template("index.html", message = message)

    return render_template('result.html', pred = pred,name=disease)

if __name__ == "__main__":
    app.run(debug=True)
