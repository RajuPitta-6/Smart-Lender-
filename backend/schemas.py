from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ApplicantBase(BaseModel):
    applicant_name: str
    gender: str
    education: str
    self_employed: str
    no_of_dependents: int
    income_annum: float
    loan_amount: float
    loan_term: int
    cibil_score: int
    

class ApplicantCreate(ApplicantBase):
    pass


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float


class ApplicantResponse(ApplicantBase):
    id: int
    prediction: str
    confidence: float
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)