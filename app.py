from flask import Flask, render_template, request
import numpy as np
import joblib
import logging
import os

# ✅ Setup
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ✅ Required model files
required_files = [
    "random_forest_best.pkl",
    "decision_tree_best.pkl",
    "label_encoder.pkl",
    "xgb_distance_model.pkl"
]

for f in required_files:
    if not os.path.exists(f):
        raise FileNotFoundError(f"❌ Missing required file: {f}")

# ✅ Load models
models = {
    "Random Forest": joblib.load("random_forest_best.pkl"),
    "Decision Tree": joblib.load("decision_tree_best.pkl")
}

# ✅ Load encoder and location model
le = joblib.load("label_encoder.pkl")
distance_model = joblib.load("xgb_distance_model.pkl")

# ✅ Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/homepage')
def homepage():
    return render_template('homepage.html')

@app.route('/ourteam')
def ourteam():
    return render_template('ourteam.html')

@app.route('/toolsandtech')
def toolsandtech():
    return render_template('toolsandtech.html')

@app.route('/input')
def input_page():
    return render_template('input.html', models=models.keys())

# ✅ Fault Type Prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features
        v1 = float(request.form['V1'])
        v2 = float(request.form['V2'])
        v3 = float(request.form['V3'])
        i1 = float(request.form['I1'])
        i2 = float(request.form['I2'])
        i3 = float(request.form['I3'])
        v0 = float(request.form['V0'])
        i0 = float(request.form['I0'])
        selected_model = request.form['model']

        features = np.array([[v1, v2, v3, i1, i2, i3, v0, i0]])

        if selected_model not in models:
            raise ValueError("Invalid model selected.")

        model = models[selected_model]
        prediction_encoded = model.predict(features)
        prediction_label = le.inverse_transform(prediction_encoded)[0]

        return render_template('result.html',
                               model=selected_model,
                               prediction=prediction_label,
                               models=models.keys())

    except Exception as e:
        logging.exception("Error during fault prediction")
        return render_template('result.html',
                               error=f"⚠️ Error: {str(e)}",
                               models=models.keys())

# ✅ Fault Location Prediction
@app.route('/predict_distance', methods=['POST'])
def predict_distance():
    try:
        # Get input features
        vs1 = float(request.form['VS1'])
        vs1_angle = float(request.form['VS1_angle'])
        vr1 = float(request.form['VR1'])
        vr1_angle = float(request.form['VR1_angle'])
        is1 = float(request.form['IS1'])
        is1_angle = float(request.form['IS1_angle'])
        ir1 = float(request.form['IR1'])
        ir1_angle = float(request.form['IR1_angle'])

        features = np.array([[vs1, vs1_angle, vr1, vr1_angle, is1, is1_angle, ir1, ir1_angle]])
        prediction = distance_model.predict(features)[0]

        return render_template('result_distance.html', prediction=prediction)

    except Exception as e:
        logging.exception("Error during location prediction")
        return render_template('result_distance.html',
                               error=f"⚠️ Error: {str(e)}")

# ✅ Run the App
if __name__ == '__main__':
    app.run(debug=True)
