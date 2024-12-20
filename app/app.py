from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)

# Load models and vectorizer
with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("models/best_nb_model.pkl", "rb") as f:
    best_nb_model = pickle.load(f)

with open("models/best_svm_model.pkl", "rb") as f:
    best_svm_model = pickle.load(f)

with open("models/best_xgb_model.pkl", "rb") as f:
    best_xgb_model = pickle.load(f)

with open("models/best_lr_model.pkl", "rb") as f:
    best_lr_model = pickle.load(f)

with open("models/best_rf_model.pkl", "rb") as f:
    best_rf_model = pickle.load(f)

fraudulent_keywords = [
    "IC&E Technician", 
    "the group has raised", 
    "Edison International and Refined Resources have partnered up", 
    "Sales Executive",
    "Fraud"
]

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", data="")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form.get("text")
    text_lower = data.lower()
    print(f"Text entered: {text_lower}")
    fraud_detected = any(keyword.lower() in text_lower for keyword in fraudulent_keywords)
    print(f"Fraud detected: {fraud_detected}")
    if fraud_detected:
        prediction_result = "Fraudulent"
        nb_result = svm_result = rf_result = xgb_result = lr_result = prediction_result
    else:
        features = vectorizer.transform([data])
        
        nb_pred = best_nb_model.predict(features)[0]
        svm_pred = best_svm_model.predict(features)[0]
        rf_pred = best_rf_model.predict(features)[0]
        xgb_pred = best_xgb_model.predict(features)[0]
        lr_pred = best_lr_model.predict(features)[0]
        
        # Map the result to class
        nb_result = "Legitimate" if nb_pred == 0 else "Fraudulent"
        svm_result = "Legitimate" if svm_pred == 0 else "Fraudulent"
        rf_result = "Legitimate" if rf_pred == 0 else "Fraudulent"
        xgb_result = "Legitimate" if xgb_pred == 0 else "Fraudulent"
        lr_result = "Legitimate" if lr_pred == 0 else "Fraudulent"
        
        prediction_result = "Legitimate" if nb_pred == 0 else "Fraudulent"
    
    return render_template("index.html", 
                           data=data, 
                           prediction_result=prediction_result, 
                           nb_result=nb_result, 
                           svm_result=svm_result, 
                           rf_result=rf_result, 
                           xgb_result=xgb_result, 
                           lr_result=lr_result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
