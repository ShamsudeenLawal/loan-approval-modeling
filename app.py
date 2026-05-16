import pandas as pd
from flask import Flask, request, render_template, jsonify
from utils.prediction_pipeline import CustomData
from utils.prediction_pipeline import PredictPipeline

# set up prediction pipeline
pred_pipeline = PredictPipeline()

app = Flask(__name__)


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
                f"Loan Status: {prediction['prediction']}"
            )

        except Exception as e:
            prediction_message = f"Error: {str(e)}"

        return render_template("home.html", prediction_message=prediction_message)

    return render_template("home.html")


if __name__ == "__main__":
    # app.run(debug=True) # development environment
    app.run(port=5000, debug=True)
