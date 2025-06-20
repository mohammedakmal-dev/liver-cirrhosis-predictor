from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("normalizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            data = [float(x) for x in request.form.values()]
            data = np.array(data).reshape(1, -1)
            scaled = scaler.transform(data)
            result = model.predict(scaled)[0]
            prediction = "Cirrhosis Detected (Class 1)" if result == 1 else "No Cirrhosis Detected (Class 2)"
        except:
            prediction = "Something went wrong. Please check your inputs."
    return render_template("index.html", result=prediction)

if __name__ == "__main__":
    app.run(debug=True)
