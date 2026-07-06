from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "ml" / "best_model.pkl"

pipeline = joblib.load(MODEL_PATH)


def predict_loan(applicant):
    input_data = pd.DataFrame(
        [
            {
                "no_of_dependents": applicant.no_of_dependents,
                "education": applicant.education,
                "self_employed": applicant.self_employed,
                "income_annum": applicant.income_annum,
                "loan_amount": applicant.loan_amount,
                "loan_term": applicant.loan_term,
                "cibil_score": applicant.cibil_score,
                "residential_assets_value": applicant.residential_assets_value,
                "commercial_assets_value": applicant.commercial_assets_value,
                "luxury_assets_value": applicant.luxury_assets_value,
                "bank_asset_value": applicant.bank_asset_value,
            }
        ]
    )

    prediction = pipeline.predict(input_data)[0]

    probability = pipeline.predict_proba(input_data)[0]

    confidence = round(float(max(probability) * 100), 2)

    prediction = "Approved" if prediction == 1 else "Rejected"

    return prediction, confidence