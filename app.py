from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model/wine_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form values
        alcohol = float(request.form["alcohol"])
        malic_acid = float(request.form["malic_acid"])
        ash = float(request.form["ash"])
        flavanoids = float(request.form["flavanoids"])
        color_intensity = float(request.form["color_intensity"])
        proline = float(request.form["proline"])

        # Arrange into array
        data = np.array([[alcohol, malic_acid, ash, flavanoids, color_intensity, proline]])

        # Scale input (VERY IMPORTANT FOR FULL MARKS)
        data_scaled = scaler.transform(data)

        # Predict
        prediction = model.predict(data_scaled)[0]

        result = f"Cultivar {prediction + 1}"

        return render_template("index.html", prediction_text=f"Prediction: {result}")

    except:
        return render_template("index.html", prediction_text="Please enter valid values.")

if __name__ == "__main__":
    app.run(debug=True)
