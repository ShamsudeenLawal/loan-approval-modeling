import pandas as pd
from flask import Flask, request, render_template, jsonify
from prediction_pipeline import CustomData
from prediction_pipeline import PredictPipeline

# set up prediction pipeline
pred_pipeline = PredictPipeline()

app = Flask(__name__)

# @app.route("/")
# def index():
#     return render_template("index.html")


@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            custom_data = CustomData(
                ApplicantIncome=float(request.form.get("ApplicantIncome")),
                CoapplicantIncome=float(request.form.get("CoapplicantIncome")),
                LoanAmount=float(request.form.get("LoanAmount")),
                LoanAmountTerm=float(request.form.get("LoanAmountTerm")),
                Gender=request.form.get("Gender"),
                Married=request.form.get("Married"),
                Dependents=request.form.get("Dependents"),
                Education=request.form.get("Education"),
                SelfEmployed=request.form.get("SelfEmployed"),
                CreditHistory=request.form.get("CreditHistory"),
                PropertyArea=request.form.get("PropertyArea"),
            )

            df_data = custom_data.get_data_as_df()

            prediction = pred_pipeline.predict(df_data)

            prediction_message = (
                f"Loan Status: {'Approved' if prediction['prediction'] == 'Yes' else 'Rejected'} with {100 * prediction['probability']:.2f}%"
            )

        except Exception as e:
            prediction_message = f"Error: {str(e)}"

        return render_template("home.html", prediction_message=prediction_message)

    return render_template("home.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()

        custom_data = CustomData(
            ApplicantIncome=float(data["ApplicantIncome"]),
            CoapplicantIncome=float(data["CoapplicantIncome"]),
            LoanAmount=float(data["LoanAmount"]),
            LoanAmountTerm=float(data["LoanAmountTerm"]),
            Gender=data["Gender"],
            Married=data["Married"],
            Dependents=data["Dependents"],
            Education=data["Education"],
            SelfEmployed=data["SelfEmployed"],
            CreditHistory=data["CreditHistory"],
            PropertyArea=data["PropertyArea"],
        )

        df_data = custom_data.get_data_as_df()

        prediction_result = pred_pipeline.predict(df_data)

        result = {
            "status": "Approved" if prediction_result["prediction"] == "Yes" else "Rejected",
            "confidence": prediction_result["probability"],
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/batch_predict", methods=["POST"])
def batch_predict():
    try:
        data = request.get_json()

        # Expecting a list of dictionaries
        if not isinstance(data, list):
            return jsonify({"error": "Input must be a list of records"}), 400

        df = pd.DataFrame(data)

        # Optional: type casting (important!)
        numeric_cols = [
            "ApplicantIncome", "CoapplicantIncome",
            "LoanAmount", "LoanAmountTerm", 
        ]
        df[numeric_cols] = df[numeric_cols].astype(float)
        df["CreditHistory"] = df["CreditHistory"].astype("category")

        predictions = pred_pipeline.model.predict(df)

        results = []
        for pred in predictions:
            results.append({
                "prediction": int(pred),
                "status": "Approved" if pred == 1 else "Rejected"
            })

        return jsonify({
            "num_predictions": len(results),
            "results": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # app.run(debug=True) # development environment
    app.run(port=5000, debug=True)
