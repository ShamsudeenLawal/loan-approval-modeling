
import os
import joblib
import pandas as pd

class CustomData:
    def __init__(self,
                 ApplicantIncome, CoapplicantIncome, LoanAmount,
                 LoanAmountTerm, Gender, Married, Dependents,
                 Education, SelfEmployed, CreditHistory, PropertyArea):

        self.ApplicantIncome = ApplicantIncome
        self.CoapplicantIncome = CoapplicantIncome
        self.LoanAmount = LoanAmount
        self.LoanAmountTerm = LoanAmountTerm
        self.Gender = Gender
        self.Married = Married
        self.Dependents = Dependents
        self.Education = Education
        self.SelfEmployed = SelfEmployed
        self.CreditHistory = CreditHistory
        self.PropertyArea = PropertyArea

    def get_data_as_df(self):
        data = {
            'ApplicantIncome': [self.ApplicantIncome],
            'CoapplicantIncome': [self.CoapplicantIncome],
            'LoanAmount': [self.LoanAmount],
            'LoanAmountTerm': [self.LoanAmountTerm],
            'Gender': [self.Gender],
            'Married': [self.Married],
            'Dependents': [self.Dependents],
            'Education': [self.Education],
            'SelfEmployed': [self.SelfEmployed],
            'CreditHistory': [self.CreditHistory],
            'PropertyArea': [self.PropertyArea]
            }

        df_data = pd.DataFrame(data=data)
        df_data["CreditHistory"] = df_data["CreditHistory"].astype("category")

        return df_data


class PredictPipeline:
    def __init__(self):
        self.model = joblib.load("models/classifier.joblib")

    def predict(self, features):
        try:
            preds = self.model.predict(features)[0]
            proba = self.model.predict_proba(features)[0][1]

            return {
                "prediction": preds,
                "probability": float(proba)
            }

        except Exception as e:
            raise Exception(f"Prediction failed: {e}")
