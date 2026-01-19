from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    features = [float(request.form[i]) for i in ['sepal_length','sepal_width','petal_length','petal_width']]
    final = np.array([features])
    pred = model.predict(final)[0]
    return render_template("index.html",prediction=pred)


if __name__== "__main__":
    app.run(debug=True,port=5000)

