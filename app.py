import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__, template_folder="template")

model = pickle.load(open("xgb_model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    if prediction[0] == 0:
        prediction = "WILL NOT"
    else:
        prediction = "WILL"
    return render_template("index.html", prediction_text = f"The customer {prediction} purchase the offer.")

if __name__ == "__main__":
    app.run(debug=True)