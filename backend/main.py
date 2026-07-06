from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware
from backend import crud, models, schemas
from backend.database import Base, engine, get_db
from backend.ml_service import predict_loan

app = FastAPI(
    title="Smart Lender API",
    version="1.0.0",
    description="Loan Approval Prediction using Machine Learning",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)


@app.get("/")
def home():
    return {
        "message": "Welcome to Smart Lender API"
    }


@app.post(
    "/predict",
    response_model=schemas.ApplicantResponse,
)
def predict(
    applicant: schemas.ApplicantCreate,
    db: Session = Depends(get_db),
):
    prediction, confidence = predict_loan(applicant)

    saved_data = crud.create_applicant(
        db=db,
        applicant=applicant,
        prediction=prediction,
        confidence=confidence,
    )

    return saved_data


@app.get(
    "/applications",
    response_model=list[schemas.ApplicantResponse],
)
def get_applications(
    db: Session = Depends(get_db),
):
    return crud.get_all_applicants(db)


@app.get(
    "/applications/{applicant_id}",
    response_model=schemas.ApplicantResponse,
)
def get_application(
    applicant_id: int,
    db: Session = Depends(get_db),
):
    applicant = crud.get_applicant_by_id(
        db,
        applicant_id,
    )

    if applicant is None:
        raise HTTPException(
            status_code=404,
            detail="Applicant not found",
        )

    return applicant


@app.delete("/applications/{applicant_id}")
def delete_application(
    applicant_id: int,
    db: Session = Depends(get_db),
):
    applicant = crud.delete_applicant(
        db,
        applicant_id,
    )

    if applicant is None:
        raise HTTPException(
            status_code=404,
            detail="Applicant not found",
        )

    return {
        "message": "Applicant deleted successfully"
    }